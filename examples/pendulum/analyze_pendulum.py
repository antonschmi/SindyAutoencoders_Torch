"""Load a trained pendulum model and check reconstruction / SINDy error on fresh
trajectories, the same checks as the original analyze_pendulum_model*.ipynb notebooks."""
import sys
sys.path.append('../../src')
import argparse

import numpy as np
import torch
from scipy.integrate import odeint

from data import get_pendulum_data, pendulum_to_movie
from model import SindyAutoencoder
from sindy_utils import sindy_simulate_order2


def load_model(save_name, device='cpu'):
    ckpt = torch.load(save_name + '.pt', map_location=device, weights_only=False)
    params = ckpt['params']
    model = SindyAutoencoder(params).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model, params


def evaluate(model, params, data, device='cpu'):
    batch = {k: torch.as_tensor(data[k], dtype=torch.float32, device=device)
              for k in ('x', 'dx', 'ddx') if k in data}
    with torch.no_grad():
        out = model(batch)
    return {k: v.detach().cpu().numpy() for k, v in out.items()}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('save_name', nargs='?', default='model1')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, params = load_model(args.save_name, device=device)

    t = np.arange(0, 20, .02)
    z0s = np.pi / np.array([1.5, 2, 3, 4, 8, 16])
    dz0s = .5 * np.ones(z0s.shape)
    f = lambda z, t: [z[1], -np.sin(z[0])]
    n_ics = z0s.size

    z = np.zeros((n_ics, t.size, 2))
    dz = np.zeros(z.shape)
    for i in range(n_ics):
        z[i] = odeint(f, [z0s[i], dz0s[i]], t)
        dz[i] = np.array([f(z[i, j], t[j]) for j in range(len(t))])

    x, dx, ddx = pendulum_to_movie(z, dz)

    test_data = {}
    test_data['x'] = x.reshape((-1, params['input_dim']))
    test_data['dx'] = dx.reshape((-1, params['input_dim']))
    test_data['ddx'] = ddx.reshape((-1, params['input_dim']))
    test_data['z'] = z[:, :, 0].reshape((-1, params['latent_dim']))
    test_data['dz'] = z[:, :, 1].reshape((-1, params['latent_dim']))
    test_data['ddz'] = dz[:, :, 1].reshape((-1, params['latent_dim']))

    results = evaluate(model, params, test_data, device=device)

    true_coefficients = np.zeros(results['sindy_coefficients'].shape)
    true_coefficients[-2] = -1.

    z_sim = np.zeros((n_ics, t.size, 2))
    pendulum_sim = np.zeros(z_sim.shape)
    for i in range(n_ics):
        z_sim[i] = sindy_simulate_order2(
            results['z'][i * t.size], results['dz'][i * t.size], t,
            params['coefficient_mask'] * results['sindy_coefficients'],
            params['poly_order'], params['include_sine'])
        pendulum_sim[i] = sindy_simulate_order2(
            test_data['z'][i * t.size], test_data['dz'][i * t.size], t,
            true_coefficients, params['poly_order'], params['include_sine'])

    sim_error = np.mean((z_sim - pendulum_sim) ** 2) / np.mean(pendulum_sim ** 2)
    print('Identified vs. true pendulum-dynamics relative error: %f' % sim_error)

    test_data = get_pendulum_data(10)
    results = evaluate(model, params, test_data, device=device)

    decoder_x_error = np.mean((test_data['x'] - results['x_decode']) ** 2) / np.mean(test_data['x'] ** 2)
    decoder_ddx_error = np.mean((test_data['ddx'] - results['ddx_decode']) ** 2) / np.mean(test_data['ddx'] ** 2)
    sindy_ddz_error = np.mean((results['ddz'] - results['ddz_predict']) ** 2) / np.mean(results['ddz'] ** 2)

    print('Decoder relative error: %f' % decoder_x_error)
    print('Decoder relative SINDy error: %f' % decoder_ddx_error)
    print('SINDy relative error, z: %f' % sindy_ddz_error)
