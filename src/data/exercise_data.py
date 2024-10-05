from torch.utils.data import Dataset
import torch
import numpy as np


class ExerciseDataset(Dataset):
    def __init__(self, X_data, y_data):
        self.X_data = X_data.astype(np.float32)
        self.y_data = y_data.astype(np.int64)

    def __len__(self):
        return len(self.X_data)

    def __getitem__(self, idx):
        sequence = torch.tensor(self.X_data[idx], dtype=torch.float32)
        label = torch.tensor(self.y_data[idx], dtype=torch.long)
        return sequence, label
