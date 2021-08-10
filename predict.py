import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataprocessing.dataset import MoADataset_test

def predict(test_ds, model_path):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = torch.load(model_path)
    
    pred = torch.zeros((len(test_ds), 206), dtype = torch.float32, device = device)

    test_dl = DataLoader(test_ds, batch_size = 128, shuffle = False)
    
    model.eval()
    
    with torch.no_grad():
        for i, x_test in enumerate(test_dl):
            x_test = torch.tensor(x_test, device = device, dtype = torch.float32)
            
            z_test = model(x_test)
            z_test = torch.sigmoid(z_test)
            
            pred[i * test_dl.batch_size : (i + 1) * test_dl.batch_size] += z_test
            
        pred = pred.cpu().detach().numpy()
        pred = pd.DataFrame(pred)
            
    return pred