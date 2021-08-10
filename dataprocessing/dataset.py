import numpy as np
from torch.utils.data import Dataset

class MoADataset(Dataset):
    def __init__(self, df, target):
        self.df = df
        self.target = target
    
    def __getitem__(self, index):
        x = self.df.iloc[index].iloc[1:]
        y = self.target.iloc[index].iloc[1:]
        x = np.array(x, dtype = np.float32)
        y = np.array(y, dtype = np.float32)
        
        return x, y
    
    def __len__(self):
        return (len(self.df))

class MoADataset_test(Dataset):
    def __init__(self, df):
        self.df = df
    
    def __getitem__(self, index):
        x = self.df.iloc[index].iloc[1:]
        x = np.array(x, dtype = np.float32)
        
        return x
    
    def __len__(self):
        return (len(self.df))