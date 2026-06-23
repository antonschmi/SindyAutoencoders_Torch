"""PyTorch reimplementation of the SindyAutoencoder architecture from
Champion, Lusch, Kutz & Brunton (2019).

Time derivatives are propagated through the encoder/decoder with forward-mode
autodiff (torch.func.jvp) instead of the hand-derived per-activation chain-rule
formulas used in the original TensorFlow 1 code. jvp is exact for any
activation function, so this removes the need to re-derive and transcribe a
separate derivative formula for every activation x {order 1, order 2}
combination - the dominant source of risk in a manual port.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import jvp


ACTIVATIONS = {
    'relu': F.relu,
    'elu': F.elu,
    'sigmoid': torch.sigmoid,
    'linear': lambda x: x,
}


class MLP(nn.Module):
    """Linear -> activation -> ... -> Linear (no activation on the final layer),
    matching build_network_layers() in the original code."""

    def __init__(self, input_dim, output_dim, widths, activation):
        super().__init__()
        self.activation_fn = ACTIVATIONS[activation]
        dims = [input_dim] + list(widths) + [output_dim]
        self.layers = nn.ModuleList(nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1))

    def forward(self, x):
        h = x
        for layer in self.layers[:-1]:
            h = self.activation_fn(layer(h))
        return self.layers[-1](h)

    def forward_jvp(self, x, dx):
        """Returns (output, d(output)/dt) given dx = dx/dt."""
        return jvp(self.forward, (x,), (dx,))

    def forward_jvp2(self, x, dx, ddx):
        """Returns (output, d(output)/dt, d^2(output)/dt^2) given dx, ddx.

        Uses nested jvp (forward-over-forward AD): differentiating
        g(x, v) = jvp(f, x, v) = (f(x), Df(x)v) along the curve (x(t), x'(t))
        gives (Df(x)x', D^2f(x)[x',x'] + Df(x)x'') = (dz, ddz).
        """
        def g(x, v):
            return jvp(self.forward, (x,), (v,))

        (z, dz), (_dz_check, ddz) = jvp(g, (x, dx), (dx, ddx))
        return z, dz, ddz


def sindy_library(z, latent_dim, poly_order, include_sine=False):
    """Order-1 SINDy library. Term ordering matches sindy_library_tf exactly."""
    batch = z.shape[0]
    terms = [torch.ones(batch, dtype=z.dtype, device=z.device)]

    for i in range(latent_dim):
        terms.append(z[:, i])

    if poly_order > 1:
        for i in range(latent_dim):
            for j in range(i, latent_dim):
                terms.append(z[:, i] * z[:, j])

    if poly_order > 2:
        for i in range(latent_dim):
            for j in range(i, latent_dim):
                for k in range(j, latent_dim):
                    terms.append(z[:, i] * z[:, j] * z[:, k])

    if poly_order > 3:
        for i in range(latent_dim):
            for j in range(i, latent_dim):
                for k in range(j, latent_dim):
                    for p in range(k, latent_dim):
                        terms.append(z[:, i] * z[:, j] * z[:, k] * z[:, p])

    if poly_order > 4:
        for i in range(latent_dim):
            for j in range(i, latent_dim):
                for k in range(j, latent_dim):
                    for p in range(k, latent_dim):
                        for q in range(p, latent_dim):
                            terms.append(z[:, i] * z[:, j] * z[:, k] * z[:, p] * z[:, q])

    if include_sine:
        for i in range(latent_dim):
            terms.append(torch.sin(z[:, i]))

    return torch.stack(terms, dim=1)


def sindy_library_order2(z, dz, latent_dim, poly_order, include_sine=False):
    """Order-2 SINDy library: identical term generation to sindy_library, just
    built over [z, dz] concatenated, matching sindy_library_tf_order2."""
    z_combined = torch.cat([z, dz], dim=1)
    return sindy_library(z_combined, 2 * latent_dim, poly_order, include_sine)


class SindyAutoencoder(nn.Module):
    def __init__(self, params):
        super().__init__()
        input_dim = params['input_dim']
        latent_dim = params['latent_dim']
        widths = params['widths']
        activation = params['activation']

        self.latent_dim = latent_dim
        self.model_order = params['model_order']
        self.poly_order = params['poly_order']
        self.include_sine = params.get('include_sine', False)
        library_dim = params['library_dim']

        self.encoder = MLP(input_dim, latent_dim, widths, activation)
        self.decoder = MLP(latent_dim, input_dim, list(reversed(widths)), activation)

        init = params.get('coefficient_initialization', 'constant')
        if init == 'xavier':
            coeffs = nn.init.xavier_uniform_(torch.empty(library_dim, latent_dim))
        elif init == 'specified':
            coeffs = torch.as_tensor(params['init_coefficients'], dtype=torch.float32)
        elif init == 'normal':
            coeffs = nn.init.normal_(torch.empty(library_dim, latent_dim))
        else:
            coeffs = torch.full((library_dim, latent_dim), 1.0)
        self.sindy_coefficients = nn.Parameter(coeffs)
        self.register_buffer('coefficient_mask', torch.ones(library_dim, latent_dim))

    def forward(self, batch):
        x, dx = batch['x'], batch['dx']
        masked_coefficients = self.coefficient_mask * self.sindy_coefficients
        out = {'x': x, 'dx': dx, 'sindy_coefficients': self.sindy_coefficients}

        if self.model_order == 1:
            z, dz = self.encoder.forward_jvp(x, dx)
            Theta = sindy_library(z, self.latent_dim, self.poly_order, self.include_sine)
            dz_predict = Theta @ masked_coefficients
            x_decode, dx_decode = self.decoder.forward_jvp(z, dz_predict)
            out.update(z=z, dz=dz, x_decode=x_decode, dx_decode=dx_decode,
                       Theta=Theta, dz_predict=dz_predict)
        else:
            ddx = batch['ddx']
            z, dz, ddz = self.encoder.forward_jvp2(x, dx, ddx)
            Theta = sindy_library_order2(z, dz, self.latent_dim, self.poly_order, self.include_sine)
            ddz_predict = Theta @ masked_coefficients
            x_decode, dx_decode, ddx_decode = self.decoder.forward_jvp2(z, dz, ddz_predict)
            out.update(ddx=ddx, z=z, dz=dz, ddz=ddz, x_decode=x_decode, dx_decode=dx_decode,
                       ddx_decode=ddx_decode, Theta=Theta, ddz_predict=ddz_predict)
        return out

    def compute_losses(self, outputs, params):
        decoder_loss = F.mse_loss(outputs['x_decode'], outputs['x'])
        if self.model_order == 1:
            sindy_z_loss = F.mse_loss(outputs['dz_predict'], outputs['dz'])
            sindy_x_loss = F.mse_loss(outputs['dx_decode'], outputs['dx'])
        else:
            sindy_z_loss = F.mse_loss(outputs['ddz_predict'], outputs['ddz'])
            sindy_x_loss = F.mse_loss(outputs['ddx_decode'], outputs['ddx'])
        reg_loss = (self.coefficient_mask * self.sindy_coefficients).abs().mean()

        losses = {
            'decoder': decoder_loss,
            'sindy_z': sindy_z_loss,
            'sindy_x': sindy_x_loss,
            'sindy_regularization': reg_loss,
        }
        loss = (params['loss_weight_decoder'] * decoder_loss
                + params['loss_weight_sindy_z'] * sindy_z_loss
                + params['loss_weight_sindy_x'] * sindy_x_loss
                + params['loss_weight_sindy_regularization'] * reg_loss)
        loss_refinement = (params['loss_weight_decoder'] * decoder_loss
                            + params['loss_weight_sindy_z'] * sindy_z_loss
                            + params['loss_weight_sindy_x'] * sindy_x_loss)
        return loss, losses, loss_refinement