from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class sign_mnist(Dataset):
    def __init__(self, path, transform=None):
        self.data = pd.read_csv(path)
        self.transform = transform

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img = self.data.iloc[idx, 1:].values.astype(np.uint8).reshape((28, 28, 1))
        label = self.data.iloc[idx, 0]
        if self.transform is not None:
            img = self.transform(img)
        return img, label