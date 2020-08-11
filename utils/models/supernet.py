import torch
import torch.nn as nn
import copy
import random
import numpy as np

kernel_size1 = 3
kernel_size2 = 5
pooling_kernel = 2
in_channels = 1
out_channels = 5
img_size = 28


class SuperNet(nn.Module):
    def __init__(self, n_classes=10, strategy='random'):
        super(SuperNet, self).__init__()
        self.conv11 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,\
                           kernel_size=(kernel_size1, kernel_size1), padding=((kernel_size1-1)//2, (kernel_size1-1)//2))
    
        self.conv12 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,\
                           kernel_size=(kernel_size2, kernel_size2), padding=((kernel_size2-1)//2, (kernel_size2-1)//2)) 

        self.conv21 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels,\
                           kernel_size=(kernel_size1, kernel_size1), padding=((kernel_size1-1)//2, (kernel_size1-1)//2))
        
        self.conv22 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels,\
                           kernel_size=(kernel_size2, kernel_size2), padding=((kernel_size2-1)//2, (kernel_size2-1)//2)) 
        
        self.relu = nn.ReLU()
        self.max_pool = torch.nn.MaxPool2d(kernel_size=(pooling_kernel, pooling_kernel))
        
        self.linear1 = nn.Linear(out_channels*(img_size//(pooling_kernel*2))**2, img_size)
        self.linear2 = nn.Linear(img_size, n_classes)
        
        self.net_type = 0
        if 'dropout' in strategy:
            strategy, dropout_rate = strategy.split('_')
            dropout_rate = float(dropout_rate)
            self.dropout = nn.Dropout(np.sqrt(dropout_rate))
        self.strategy = strategy
                
    def sampler(self, net_type):
        model = copy.deepcopy(self)
        model.net_type = net_type
        model.strategy = 'sample'
        return model
    
    def random_init(self):
        nn.init.xavier_uniform_(self.conv11.weight)
        nn.init.xavier_uniform_(self.conv12.weight)
        nn.init.xavier_uniform_(self.conv21.weight)
        nn.init.xavier_uniform_(self.conv22.weight)
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)
        
    def forward(self, x):
        if self.strategy == 'random':
            self.net_type = random.randint(1, 4)
        
        if self.net_type == 0:
            out = self.conv11(x)
            out2 = self.conv12(x)    
            if self.strategy == 'mean':
                out += out2
                out /= 2
            elif self.strategy == 'dropout':
                out = self.dropout(out) + self.dropout(out2)
            elif self.strategy == 'sum':
                out += out2
                
        elif self.net_type == 1 or self.net_type == 2:
            out = self.conv11(x)
        elif self.net_type == 3 or self.net_type == 4:
            out = self.conv12(x)
        
        out = self.relu(out)
        out = self.max_pool(out)

        if self.net_type == 0:
            out2 = self.conv22(out)
            out = self.conv21(out)
            if self.strategy == 'mean':
                out += out2
                out /= 2
            elif self.strategy == 'dropout':
                out = self.dropout(out) + self.dropout(out2)
            elif self.strategy == 'sum':
                out += out2
                
        elif self.net_type == 1 or self.net_type == 3:
            out = self.conv21(out)
        elif self.net_type == 2 or self.net_type == 4:
            out = self.conv22(out)        

        out = self.relu(out)
        out = self.max_pool(out)
        out = torch.flatten(out, start_dim=1)
        out = self.linear1(out)
        out = self.relu(out)
        out = self.linear2(out)
        return out