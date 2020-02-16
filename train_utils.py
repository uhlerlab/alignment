import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class BaseLayer(nn.Module):
    def __init__(self, n, init_val):
        
        super(BaseLayer, self).__init__()
        self.n = n
        self.init_val = init_val
        self.init = self.init_val * torch.randn(n, n)
        self.weight = nn.Parameter(torch.tensor(self.init))
        
    def forward(self, input):
        return F.linear(input, self.weight)

    def weight_matrix(self):
    	return self.weight

class DeepNet(nn.Module):
    def __init__(self, n, depth):
        
        super(DeepNet, self).__init__()
        self.n = n
        self.depth = depth
        self.init_val = 1e-1
        self.layers = nn.ModuleList([BaseLayer(n, self.init_val) for i in range(depth)])

    def forward(self, input):
        x = input
        for i in range(self.depth):
            x = self.layers[i](x)
        return x
    
    def weight_matrix(self):
        net = torch.eye(self.n)
        for i in range(self.depth):
            net = torch.mm(self.layers[i].weight_matrix(), net)
        return net
