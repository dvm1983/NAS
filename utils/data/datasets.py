import torch
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    """
    вспомогательный класс, формирует датасет для объекта класса Dataloader
    """
    def __init__(self, data, labels=None):
        self.data = data
        self.labels = labels

    def __getitem__(self, idx):
        data_item = self.data[idx]
        if not self.labels is None:
            label = self.labels[idx]
            return data_item, label
        else:
            return data_item

    def __len__(self):
        return len(self.data)