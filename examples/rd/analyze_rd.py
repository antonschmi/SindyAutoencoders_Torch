"""NOTE: needs reaction_diffusion.mat in this directory (see train_rd.py) -
untested without that data file, but ported faithfully from the original
analyze_rd_model*.ipynb notebooks."""
import sys
sys.path.append('../../src')
import argparse

import numpy as np
import torch

from data import get_rd_data
from model import SindyAutoencoder
from sindy_utils import sindy_simulate


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

    _, _, test_data = get_rd_data()
    results = evaluate(model, params, test_data, device=device)

    z_sim = sindy_simulate(results['z'][0], test_data['t'][:, 0],
                            params['coefficient_mask'] * results['sindy_coefficients'],
                            params['poly_order'], params['include_sine'])

    decoder_x_error = np.mean((test_data['x'] - results['x_decode']) ** 2) / np.mean(test_data['x'] ** 2)
    decoder_dx_error = np.mean((test_data['dx'] - results['dx_decode']) ** 2) / np.mean(test_data['dx'] ** 2)
    sindy_dz_error = np.mean((results['dz'] - results['dz_predict']) ** 2) / np.mean(results['dz'] ** 2)

    print('Decoder relative error: %f' % decoder_x_error)
    print('Decoder relative SINDy error: %f' % decoder_dx_error)
    print('SINDy relative error, z: %f' % sindy_dz_error)
