import torch.nn as nn

from activation import Swish, Mish

class SwishNN(nn.Module):
    def __init__(self, n_features, hidden_size1, hidden_size2, n_output):
        super(SwishNN, self).__init__()
        
        self.nn = nn.Sequential(
            nn.Linear(in_features = n_features, out_features = hidden_size1),
            nn.BatchNorm1d(hidden_size1),
            Swish(),
            nn.Linear(in_features = hidden_size1, out_features = hidden_size2),
            nn.BatchNorm1d(hidden_size2),
            Swish(),
            nn.Linear(in_features = hidden_size2, out_features = n_output),
        )

    def forward(self, x):
        x = self.nn(x)
        return x

class MishNN(nn.Module):
    def __init__(self, n_features, hidden_size1, hidden_size2, n_output):
        super(MishNN, self).__init__()
        
        self.nn = nn.Sequential(
            nn.Linear(in_features = n_features, out_features = hidden_size1),
            nn.BatchNorm1d(hidden_size1),
            Mish(),
            nn.Linear(in_features = hidden_size1, out_features = hidden_size2),
            nn.BatchNorm1d(hidden_size2),
            Mish(),
            nn.Linear(in_features = hidden_size2, out_features = n_output),
        )
    
    def forward(self, x):
        x = self.nn(x)
        return x