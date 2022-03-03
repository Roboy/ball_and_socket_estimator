import torch
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader


class SensorsDataset(Dataset):   
    '''
    Custom Dataset subclass. 
    Serves as input to DataLoader to transform X 
      into sequence data using rolling window. 
    DataLoader using this dataset will output batches 
      of `(batch_size, seq_len, n_features)` shape.
    Suitable as an input to RNNs. 
    '''
    def __init__(self, X: np.ndarray, u: np.ndarray, y: np.ndarray, X_hat: np.ndarray = None):
        self.X = torch.tensor(X).float()
        self.u = torch.tensor(u).float()
        self.y = torch.tensor(y).float()
        if X_hat is not None:
            self.X_hat = torch.tensor(X_hat).float()
        else:
            self.X_hat = None

    def __len__(self):
        return self.X.__len__()

    def __getitem__(self, index):
        # if self.X_hat is not None:
        #     return (self.X[index], self.u[index], self.y[index], self.X_hat[index])
        # else:
        #     return (self.X[index], self.u[index], self.y[index])
        if self.X_hat is not None:
            return (self.X[index], self.u[index], self.y[index], self.X_hat[index])
        else:
            return (self.X[index], self.u[index], self.y[index])


class SensorsDataModule(pl.LightningDataModule):
    '''
    PyTorch Lighting DataModule subclass:
    https://pytorch-lightning.readthedocs.io/en/latest/datamodules.html

    Serves the purpose of aggregating all data loading 
      and processing work in one place.
    '''
    
    def __init__(self, data_x, data_u, data_y, data_x_hat = None, seq_len = 1, batch_size = 128, num_workers=0):
        super().__init__()
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.X_train = None
        self.U_train = None
        self.X_hat_train = None
        self.y_train = None
        self.X_val = None
        self.U_val = None
        self.X_hat_val = None
        self.y_val = None
        self.X_test = None
        self.U_test = None
        self.X_hat_test = None
        self.y_test = None
        self.columns = None
        self.preprocessing = None
        self.data_x = data_x
        self.data_u = data_u
        self.data_y = data_y
        self.data_x_hat = data_x_hat

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        '''
        Data is resampled to hourly intervals.
        Both 'np.nan' and '?' are converted to 'np.nan'
        'Date' and 'Time' columns are merged into 'dt' index
        '''

        if stage == 'fit' and self.X_train is not None:
            return 
        if stage == 'test' and self.X_test is not None:
            return
        if stage is None and self.X_train is not None and self.X_test is not None:  
            return
        
        data_split = 0.7
        data_idx = np.arange(len(self.data_x))
        np.random.shuffle(data_idx)
        split_idx = int(len(self.data_x)*data_split)

        if stage == 'fit' or stage is None:
            self.X_train = np.array(self.data_x)[data_idx[:split_idx]]
            self.U_train = np.array(self.data_u)[data_idx[:split_idx]]
            self.y_train = np.array(self.data_y)[data_idx[:split_idx]]
            self.X_val = np.array(self.data_x)[data_idx[split_idx:]]
            self.U_val = np.array(self.data_u)[data_idx[split_idx:]]
            self.y_val = np.array(self.data_y)[data_idx[split_idx:]]

            if self.data_x_hat is not None:
                self.X_hat_train = np.array(self.data_x_hat)[data_idx[:split_idx]]
                self.X_hat_val = np.array(self.data_x_hat)[data_idx[split_idx:]]
            
        if stage == 'test' or stage is None:
            self.X_test = np.array(self.data_x)[data_idx[split_idx:]]
            self.U_test = np.array(self.data_u)[data_idx[split_idx:]]
            self.y_test = np.array(self.data_y)[data_idx[split_idx:]]

            if self.data_x_hat is not None:
                self.X_hat_test = np.array(self.data_x_hat)[data_idx[split_idx:]]
        

    def train_dataloader(self):
        train_dataset = SensorsDataset(self.X_train, 
                                       self.U_train,
                                       self.y_train,
                                       self.X_hat_train)
        train_loader = DataLoader(train_dataset, 
                                  batch_size = self.batch_size, 
                                  shuffle = False, 
                                  num_workers = self.num_workers)
        
        return train_loader

    def val_dataloader(self):
        val_dataset = SensorsDataset(self.X_val, 
                                     self.U_val,
                                     self.y_val,
                                     self.X_hat_val)
        val_loader = DataLoader(val_dataset, 
                                batch_size = self.batch_size, 
                                shuffle = False, 
                                num_workers = self.num_workers)

        return val_loader

    def test_dataloader(self):
        test_dataset = SensorsDataset(self.X_test, 
                                      self.U_test,
                                      self.y_test,
                                      self.X_hat_test)
        test_loader = DataLoader(test_dataset, 
                                 batch_size = self.batch_size, 
                                 shuffle = False, 
                                 num_workers = self.num_workers)

        return test_loader
