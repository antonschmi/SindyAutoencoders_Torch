"""NOTE: this example needs reaction_diffusion.mat in this directory, generated
by the MATLAB solver in SindyAutoencoders/rd_solver/. That file is not part of
this repo, so this script cannot currently be run end-to-end - the code path
below is a faithful port, but is untested without that data file."""
import sys
sys.path.append('../../src')
import os
import datetime

import numpy as np
import pandas as pd
import torch

from data import get_rd_data
from sindy_utils import library_size
from training import train_network


def build_params(training_data):
    params = {}

    params['input_dim'] = training_data['y1'].size * training_data['y2'].size
    params['latent_dim'] = 2
    params['model_order'] = 1
    params['poly_order'] = 3
    params['include_sine'] = True
    params['library_dim'] = library_size(params['latent_dim'], params['poly_order'],
                                          params['include_sine'], True)

    params['sequential_thresholding'] = True
    params['coefficient_threshold'] = 0.1
    params['threshold_frequency'] = 500
    params['coefficient_mask'] = np.ones((params['library_dim'], params['latent_dim']))
    params['coefficient_initialization'] = 'constant'

    params['loss_weight_decoder'] = 1.0
    params['loss_weight_sindy_z'] = 0.01
    params['loss_weight_sindy_x'] = 0.5
    params['loss_weight_sindy_regularization'] = 0.1

    params['activation'] = 'sigmoid'
    params['widths'] = [128, 64, 32]

    params['epoch_size'] = training_data['x'].shape[0]
    params['batch_size'] = 1000
    params['learning_rate'] = 1e-4
    params['shuffle'] = False  # matches the original paper's fixed sequential batching

    params['data_path'] = os.getcwd() + '/'
    params['print_progress'] = True
    params['print_frequency'] = 100

    params['max_epochs'] = 5001
    params['refinement_epochs'] = 1001

    return params


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using device:', device)

    training_data, validation_data, test_data = get_rd_data()

    params = build_params(training_data)

    num_experiments = 10
    df = pd.DataFrame()
    for i in range(num_experiments):
        print('EXPERIMENT %d' % i)

        params['coefficient_mask'] = np.ones((params['library_dim'], params['latent_dim']))
        params['save_name'] = 'rd_' + datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")

        results_dict = train_network(training_data, validation_data, params, device=device)
        df = pd.concat([df, pd.DataFrame([{**results_dict, **params}])], ignore_index=True)

    df.to_pickle('experiment_results_' + datetime.datetime.now().strftime("%Y%m%d%H%M") + '.pkl')
