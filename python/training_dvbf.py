import numpy as np
import pandas as pd
import time
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
import joblib
import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import matplotlib
import os

import torch
import torch.nn as nn
from torch.autograd import Variable
from libs.orientation_utils import compute_rotation_matrix_from_euler, \
            compute_ortho6d_from_rotation_matrix, compute_rotation_matrix_from_ortho6d, \
            compute_euler_angles_from_rotation_matrices

import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers.csv_logs import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from libs.data_modules import SensorsDataModule
from libs.nn_models import LSTMRegressor
from libs.dvbf_models import DVBF


if __name__ == '__main__':
    body_part = "shoulder_left"
    file_names = ['./python/training_data/shoulder_left_train_ts_5_6_7_9_10.log']

    # body_part = "shoulder_right"
    # file_names = ['./python/training_data_old/shoulder_right_data_6.log']

    # Parse data
    # dataset = [pd.read_csv(f, delim_whitespace=True, header=0) for f in file_names]
    # dataset = [data[(np.abs(stats.zscore(data[["roll", "pitch", "yaw"]])) < 2.75).all(axis=1)] for data in dataset]
    orig_dataset = pd.concat([pd.read_csv(f, delim_whitespace=True, header=0) for f in file_names])
    orig_dataset = orig_dataset.values[1:len(orig_dataset)-1,1:]

    abnormal_threshold = 0.5
    dataset = []

    for name in file_names:
        df = pd.read_csv(name, delim_whitespace=True, header=0)
        
        # interpolate nan outputs (loss tracking)
        for c in df.columns[-3:]:
            df[c] = df[c].interpolate()
        
        # interpolate abnormal inputs
        for c in df.columns[1:-3]:
            bad_idx = df.index[df[c].pct_change().abs().ge(abnormal_threshold)]
            df.loc[bad_idx, c] = np.nan
            df[c] = df[c].interpolate()
        
        # Add action
        for c in df.columns[-3:]:
            df["dt"] = (df["timestamp"][1:].tolist() - df["timestamp"][:-1]) / 10**9
            df[c + "_u"] = (df[c][1:].tolist() - df[c][:-1]) / df["dt"]
        
        # Remove last row
        df = df[:-1]
        dataset.append(df)

    dataset_len = [len(data) for data in dataset]
    dataset = pd.concat(dataset) 

    print(f'{np.sum(dataset_len)} values')

    dataset = dataset.values[1:len(dataset)-1,1:]
    dataset = dataset[abs(dataset[:,12])!=0.0,:]
    dataset = dataset[abs(dataset[:,13])!=0.0,:]
    dataset = dataset[abs(dataset[:,14])!=0.0,:]

    # print(f'{len(dataset)} values after filtering outliers')

    euler_set = dataset[:, 12:15]
    action_set = dataset[:, 16:]
    sensors_set = dataset[:, :12]
    orig_sensors_set = orig_dataset[:, :12]
    print(f'max euler {str(np.amax(euler_set))}')
    print(f'min euler {str(np.amin(euler_set))}')

    # Preproccess

    euler_set_in = np.zeros_like(euler_set)
    euler_set_in[:, 0] = euler_set[:, 2] 
    euler_set_in[:, 1] = euler_set[:, 1] 
    euler_set_in[:, 2] = euler_set[:, 0] 

    euler_set = torch.Tensor(euler_set_in).cuda()
    rot_set = compute_rotation_matrix_from_euler(euler_set)
    rot_set = compute_ortho6d_from_rotation_matrix(rot_set).cpu().detach().numpy()

    sensors_scaler = MinMaxScaler(feature_range=(-1., 1.))
    action_scaler = MinMaxScaler(feature_range=(-1., 1.))

    # Split magnetic sensors into 4 independent distributions again
    orig_sensors_set = sensors_scaler.fit_transform(orig_sensors_set).reshape(-1, 4, 3)
    sensors_set = sensors_scaler.transform(sensors_set).reshape(-1, 4, 3)
    action_set = action_scaler.fit_transform(action_set)


    # Dataloader

    look_back = 10

    data_in = []
    data_u = []
    data_hat_in = []
    data_out = []

    start_idx = 0
    for l in dataset_len:
        # Ignore the last batch
        for i in range(start_idx, start_idx+l-look_back*2):
            data_in.append(orig_sensors_set[i:i+look_back])
            data_hat_in.append(sensors_set[i:i+look_back])
            data_u.append(action_set[i:i+look_back])
            data_out.append(rot_set[i+1:i+look_back+1])
        print(len(data_in))
        start_idx += l

    p = dict(
        seq_len = look_back,
        batch_size = 1000, 
        max_epochs = 1000,
        n_frames = 4,
        n_observations = 3,
        n_actions = 3,
        n_latents = 16,
        n_outputs = 6,
        n_initial_obs = 3,
        hidden_size = 100,
        learning_rate = 5e-4,
        alpha = 3.0,
        beta = 3.0,
        annealing = 0.3,
        temperature = 5e-2
    )

    model_path = f"./outputs/{body_part}_dvbf_with_action_rot6D"

    if not os.path.exists(model_path):
        os.makedirs(model_path)

    joblib.dump((sensors_scaler, action_scaler), f'{model_path}/scaler.pkl')
    with open(f'{model_path}/hyperparams.json', 'w') as fp:
        json.dump(p, fp)

    seed_everything(1)

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=model_path,
        filename="best-{epoch:03d}-{val_loss:.5f}",
        save_top_k=3,
        mode="min",
    )

    trainer = Trainer(
        max_epochs=p['max_epochs'],
        callbacks=[checkpoint_callback],
        gpus=1,
        log_every_n_steps=10,
        progress_bar_refresh_rate=2,
    )

    model = DVBF(
        n_frames = p['n_frames'],
        n_observations = p['n_observations'],
        n_actions = p['n_actions'],
        n_latents = p['n_latents'],
        n_outputs = p['n_outputs'],
        n_initial_obs = p['n_initial_obs'],
        hidden_size = p['hidden_size'],
        seq_len = p['seq_len'],
        batch_size = p['batch_size'],
        learning_rate = p['learning_rate'],
        alpha = p['alpha'],
        beta = p['beta'],
        annealing = p['annealing'],
        temperature = p['temperature'],
    )


    dm = SensorsDataModule(
    #     data_x = data_in,
        data_x = data_hat_in,
        data_u = data_u,
        data_y = data_out,
    #     data_x_hat = data_hat_in,
        seq_len = p['seq_len'],
        batch_size = p['batch_size']
    )

    trainer.fit(model, dm)
    trainer.test(model, datamodule=dm)

    device = "cuda"
    model.to(device)
    torch.set_grad_enabled(False)
    model.eval()

    in_set = []
    for i in range(0, 1000):
        in_set.append(sensors_set[i:i+look_back])

    in_set = torch.tensor(in_set, dtype=torch.float32).to(device)
    out_set = model.predict(in_set)[:, -1]