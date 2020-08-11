import torch
import torchvision
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from utils.data.datasets import CustomDataset


def make_mnist_dataloaders(data_dir, random_state, val_size=0.15, test_size=0.15, batch_size=4, num_workers=0):
    mnist_mu = 0.1307
    mnist_sigma = 0.3081
    mnist = MNIST(root=data_dir, train=True, transform=None, target_transform=None, download=True)

    data = (mnist.data.unsqueeze(1).float()/255. - mnist_mu)/mnist_sigma
    labels = mnist.targets   
    
    indexes = list(range(len(data)))
    
    train_idx, test_idx, = train_test_split(indexes,test_size=test_size,random_state=random_state, stratify=labels)
    train_idx, val_idx = train_test_split(train_idx, test_size=val_size, random_state=random_state, stratify=labels[train_idx])    
    
    train_data = data[train_idx]
    val_data = data[val_idx]
    test_data = data[test_idx]
    
    train_labels = labels[train_idx]
    val_labels = labels[val_idx]
    test_labels = labels[test_idx]
    
    train_dataset = CustomDataset(train_data, train_labels)
    val_dataset = CustomDataset(val_data, val_labels)
    test_dataset = CustomDataset(test_data, test_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader, test_loader