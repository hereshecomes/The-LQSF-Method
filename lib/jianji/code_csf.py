import numpy as np
import torch 
from torch.utils.data import Dataset
import pandas as pd
from d2l import torch as d2l

class csf(Dataset):
    def __init__(self, csv_path='mbessay1.pkl', mode='train'):
        self.df = pd.read_pickle(csv_path)
        self.num_exp = len(self.df)
        self.mode = mode
        self.num_feature = self.df.shape[1] - 1
        self.train_cols = list(range(self.num_feature))
        self.label_col = self.num_feature
        self.train_list = self.df[self.train_cols].values.tolist()
        self.train_list = self.train_list
        self.label_list = self.df[self.label_col].values.tolist()

    def data_size(self):
        return (self.num_exp, self.num_feature)
    
    def __len__(self):
        return self.num_exp
    
    def __getitem__(self, idx):

        feature = np.array(self.train_list[idx]).reshape(-1).astype(np.float32)
        feature = feature / 4
        feature = feature[None, None, :] #for Conv2d
        # feature = feature[None, :]# for Conv1d
        label = np.array(self.label_list[idx]).reshape(-1).astype(np.float32)
        return  feature, label

class csf1(Dataset):
    def __init__(self, csv_path='mbessay1.pkl', mode='train'):
        self.df = pd.read_pickle(csv_path)
        self.num_exp = len(self.df)
        self.mode = mode
        self.num_feature = self.df.shape[1] - 1
        self.train_cols = list(range(self.num_feature))
        self.label_col = self.num_feature
        self.train_list = self.df[self.train_cols].values.tolist()
        self.train_list = self.train_list
        self.label_list = self.df[self.label_col].values.tolist()

    def data_size(self):
        return (self.num_exp, self.num_feature)
    
    def __len__(self):
        return self.num_exp
    
    def __getitem__(self, idx):

        feature = np.array(self.train_list[idx]).reshape(-1).astype(np.float32)
        feature = feature / 4
        # feature = feature[None, None, :] #for Conv2d
        feature = feature[None, :]# for Conv1d
        label = np.array(self.label_list[idx]).reshape(-1).astype(np.float32)
        return  feature, label
