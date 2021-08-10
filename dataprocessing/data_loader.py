from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from .dataset import MoADataset, MoADataset_test

def processing_dl(tr_x, tr_y, val_x, val_y):
    # dataset
    tr_ds = MoADataset(df = tr_x, target = tr_y)
    val_ds = MoADataset(df = val_x, target = val_y)

    # dataLoader
    tr_dl = DataLoader(dataset = tr_ds, batch_size = 128, shuffle = True)
    val_dl = DataLoader(dataset = val_ds, batch_size = 128, shuffle = False)

    return tr_dl, val_dl

def processing_test_dl(test):
    test_ds = MoADataset_test(df=test)
    test_dl = DataLoader(dataset=test_ds, batch_size = 128, shuffle = False)

    return test_dl