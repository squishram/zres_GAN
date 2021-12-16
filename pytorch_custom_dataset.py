import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from skimage import io



class customDataset(Dataset):
    def __init__(self, csv, root_dir, transform=None):
        self.annotations = pd.read_csv(csv)
        self.root_dir = root_dir
        self.transform = transform
    def __len__(self):
        return len(self.annotation)
    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 0])
        image = io.imread(img_path)
        ylabel = torch.tensor(int(self.annotations.iloc[index, 0]))
        
        if self.transform:
            image = self.transform(image)
            
        return (image, ylabel)
    
