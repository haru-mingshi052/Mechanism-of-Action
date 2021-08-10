import numpy as np
import time
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

def train_model(model, tr_dl, val_dl, output_folder, epochs, es_patience):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    model.to(device)
    
    criterion = nn.BCEWithLogitsLoss()
    
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)
    
    scheduler = ReduceLROnPlateau(
        optimizer = optimizer, 
        mode = 'min', 
        patience = 5, 
        verbose = True, 
        factor = 0.2
    )
    
    model_path = output_folder + 'model_weight.pth'
    patience = es_patience
    best_loss = np.inf
    
    
    for epoch in range(epochs):
        start_time = time.time()
        
        train_loss = 0.0
        val_loss = 0.0
        
        model.train()
        
        for x, y in tr_dl:
            x = torch.tensor(x, device = device, dtype = torch.float32)
            y = torch.tensor(y, device = device, dtype = torch.float32)
            optimizer.zero_grad()
            z = model(x)
            loss = criterion(z, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        train_loss /= len(tr_dl)
        
        model.eval()
        with torch.no_grad():
            for x_val, y_val in val_dl:
                x_val = torch.tensor(x_val, device = device, dtype = torch.float32)
                y_val = torch.tensor(y_val, device = device, dtype = torch.float32)
                z_val = model(x_val)
                loss = criterion(z_val, y_val)
                val_loss += loss.item()
            
            val_loss /= len(val_dl)
            
            finish_time = time.time()
            
            print('Epochs：{:03} | Train Loss：{:.5f} | Val Loss：{:.5f} | Training Time：{:.7f}'
                  .format(epoch + 1, train_loss, val_loss, finish_time - start_time))
            
            scheduler.step(val_loss)
            
            
            if val_loss <= best_loss:
                best_loss = val_loss
                patience = es_patience
                
                torch.save(model, model_path)
                
            else:
                patience -= 1
                if patience == 0:
                    print('Early Stopping：{:.5f}'.format(best_loss))
                    break