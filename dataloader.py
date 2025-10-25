
import pandas as pd
import torch
import yaml
from torch.utils.data import Dataset, DataLoader
import os

class LogDataset(Dataset):
    def __init__(self, csv_path, use_depth=False, use_well=False):

        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"文件未找到: {csv_path}")

        df = pd.read_csv(csv_path)

        feature_columns = ['SP', 'GR', 'AC']
        if use_depth:
            feature_columns.append('DEPTH')
        if use_well:
            feature_columns.append('WELL')

        self.ids = df['id'].values
        self.features = df[feature_columns].values
        
        self.labels = df['label'].values
        self.features = (self.features - self.features.mean(axis=0)) / (self.features.std(axis=0) + 1e-8)

        print("Label distribution in dataset:")
        print(df['label'].value_counts(normalize=True))

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        x = torch.tensor(self.features[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        id_ = torch.tensor(self.ids[idx], dtype=torch.long)
        return x, y, id_
    
    def get_labels(self):
        return self.labels


def get_dataloader(csv_path, batch_size=64, shuffle=True, num_workers=4, pin_memory=False, use_depth=False, use_well=False):
    dataset = LogDataset(csv_path, use_depth=use_depth, use_well=use_well)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )
    return dataloader, dataset.get_labels()