"""Modern training loop for the PyTorch SindyAutoencoder.

The original TF1 code fed one mini-batch at a time through a `feed_dict`,
which means a Python<->session round trip (and a host->device copy, if a GPU
were used) for every single step. For ~250 steps/epoch x thousands of epochs
that round-trip overhead dominates wall-clock time, not the actual compute -
which is why putting that code on a GPU barely helped.

Here the whole train/validation set is moved onto the target device once,
up front. Every batch is then just a slice of an already-resident tensor, so
there is no per-step host<->device traffic at all, and the GPU (if used) is
fed continuously instead of waiting on data movement between steps.
"""
import pickle

import numpy as np
import torch

from model import SindyAutoencoder


def _to_device(data, device):
    return {k: torch.as_tensor(v, dtype=torch.float32, device=device)
            for k, v in data.items() if k in ('x', 'dx', 'ddx')}


def train_network(training_data, val_data, params, device=None):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

    model = SindyAutoencoder(params).to(device)
    model.coefficient_mask.copy_(torch.as_tensor(params['coefficient_mask'], dtype=torch.float32))
    optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])

    train_t = _to_device(training_data, device)
    val_t = _to_device(val_data, device)

    n_train = train_t['x'].shape[0]
    batch_size = params['batch_size']
    steps_per_epoch = params['epoch_size'] // batch_size
    shuffle = params.get('shuffle', False)

    x_norm = (val_t['x'] ** 2).mean().item()
    deriv_key = 'dx' if params['model_order'] == 1 else 'ddx'
    sindy_predict_norm_x = (val_t[deriv_key] ** 2).mean().item()

    validation_losses = []
    sindy_model_terms = [int(np.sum(params['coefficient_mask']))]

    def run_epochs(num_epochs, refinement):
        for i in range(num_epochs):
            order = torch.randperm(n_train, device=device) if shuffle else None
            for j in range(steps_per_epoch):
                if order is None:
                    idx = slice(j * batch_size, (j + 1) * batch_size)
                    batch = {k: v[idx] for k, v in train_t.items()}
                else:
                    idx = order[j * batch_size:(j + 1) * batch_size]
                    batch = {k: v[idx] for k, v in train_t.items()}

                optimizer.zero_grad()
                outputs = model(batch)
                loss, losses, loss_refinement = model.compute_losses(outputs, params)
                (loss_refinement if refinement else loss).backward()
                optimizer.step()

            if params['print_progress'] and (i % params['print_frequency'] == 0):
                validation_losses.append(
                    print_progress(model, i, params, batch, val_t, x_norm, sindy_predict_norm_x))

            if (not refinement and params['sequential_thresholding']
                    and (i % params['threshold_frequency'] == 0) and i > 0):
                with torch.no_grad():
                    new_mask = (model.sindy_coefficients.abs() > params['coefficient_threshold']).float()
                model.coefficient_mask.copy_(new_mask)
                params['coefficient_mask'] = new_mask.cpu().numpy()
                n_active = int(new_mask.sum().item())
                print('THRESHOLDING: %d active coefficients' % n_active)
                sindy_model_terms.append(n_active)
        return i

    print('TRAINING')
    last_epoch = run_epochs(params['max_epochs'], refinement=False)

    print('REFINEMENT')
    run_epochs(params['refinement_epochs'], refinement=True)

    torch.save({'model_state_dict': model.state_dict(), 'params': params},
               params['data_path'] + params['save_name'] + '.pt')
    pickle.dump(params, open(params['data_path'] + params['save_name'] + '_params.pkl', 'wb'))

    model.eval()
    with torch.no_grad():
        outputs = model(val_t)
        _, final_losses, _ = model.compute_losses(outputs, params)
        deriv_pred_key = 'dz_predict' if params['model_order'] == 1 else 'ddz_predict'
        deriv_actual_key = 'dz' if params['model_order'] == 1 else 'ddz'
        sindy_predict_norm_z = (outputs[deriv_actual_key] ** 2).mean().item()
        sindy_coefficients = model.sindy_coefficients.detach().cpu().numpy()

    results_dict = {
        'num_epochs': last_epoch,
        'x_norm': x_norm,
        'sindy_predict_norm_x': sindy_predict_norm_x,
        'sindy_predict_norm_z': sindy_predict_norm_z,
        'sindy_coefficients': sindy_coefficients,
        'loss_decoder': final_losses['decoder'].item(),
        'loss_decoder_sindy': final_losses['sindy_x'].item(),
        'loss_sindy': final_losses['sindy_z'].item(),
        'loss_sindy_regularization': final_losses['sindy_regularization'].item(),
        'validation_losses': np.array(validation_losses),
        'sindy_model_terms': np.array(sindy_model_terms),
    }
    return results_dict


def print_progress(model, epoch, params, train_batch, val_data, x_norm, sindy_predict_norm):
    model.eval()
    with torch.no_grad():
        train_out = model(train_batch)
        train_loss, train_losses, _ = model.compute_losses(train_out, params)
        val_out = model(val_data)
        val_loss, val_losses, _ = model.compute_losses(val_out, params)
    model.train()

    train_vals = (train_loss.item(),) + tuple(v.item() for v in train_losses.values())
    val_vals = (val_loss.item(),) + tuple(v.item() for v in val_losses.values())
    print("Epoch %d" % epoch)
    print("   training loss {0}, {1}".format(train_vals[0], train_vals[1:]))
    print("   validation loss {0}, {1}".format(val_vals[0], val_vals[1:]))
    decoder_loss = val_losses['decoder'].item()
    sindy_x_loss = val_losses['sindy_x'].item()
    print("decoder loss ratio: %f, decoder SINDy loss  ratio: %f" %
          (decoder_loss / x_norm, sindy_x_loss / sindy_predict_norm))
    return val_vals
