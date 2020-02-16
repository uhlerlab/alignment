# encode a 3x3 matrix
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib
import matplotlib.pyplot as plt
import train_utils as train

class ConvLayer(train.BaseLayer):
    def __init__(self, n, init_val, init_vec=None):
        # n has to equal 9 for now
        super(ConvLayer, self).__init__(n, init_val)
        if init_vec is None:
            self.init = self.init_val * torch.randn(9)
            self.weight = nn.Parameter(torch.tensor(self.init))
        else:
            self.weight = nn.Parameter(torch.tensor(init_vec, dtype=torch.float32))
        
    def forward(self, input):
        return F.linear(input, self.weight_matrix())
    
    def weight_matrix(self):
        rows = [
            torch.cat([self.weight.narrow(0, 4, 2), torch.zeros(1), self.weight.narrow(0, 7, 2), torch.zeros(4)]),
            torch.cat([self.weight.narrow(0, 3, 6), torch.zeros(3)]),
            torch.cat([torch.zeros(1), self.weight.narrow(0, 3, 2), torch.zeros(1), self.weight.narrow(0, 6, 2), torch.zeros(3)]),
            torch.cat([self.weight.narrow(0, 1, 2), torch.zeros(1), self.weight.narrow(0, 4, 2), torch.zeros(1), self.weight.narrow(0, 7, 2), torch.zeros(1)]),
            self.weight,
            torch.cat([torch.zeros(1), self.weight.narrow(0, 0, 2), torch.zeros(1), self.weight.narrow(0, 3, 2), torch.zeros(1), self.weight.narrow(0, 6, 2)]),
            torch.cat([torch.zeros(3), self.weight.narrow(0, 1, 2), torch.zeros(1), self.weight.narrow(0, 4, 2), torch.zeros(1)]),
            torch.cat([torch.zeros(3), self.weight.narrow(0, 0, 6)]),
            torch.cat([torch.zeros(4), self.weight.narrow(0, 0, 2), torch.zeros(1), self.weight.narrow(0, 3, 2)])
        ]
        return torch.stack(rows, dim = 1)
                            
class DeepConv(train.DeepNet):
    def __init__(self, n, depth, init_vec=None):
        
        super(DeepConv, self).__init__(n, depth)

        init_val = 0.1
        if init_vec is None:
            self.layers = nn.ModuleList([ConvLayer(n, init_val) for i in range(depth)])
        else:
            self.layers = nn.ModuleList([ConvLayer(n, 0.0, init_vec=init_vec) for i in range(depth)])

class Conv(nn.Module):
    def __init__(self, depth, init_vec):
        super(Conv, self).__init__()

        self.layers = nn.ModuleList([nn.Conv2d(1, 1, 3, 1, 1) for i in range(depth)])
        for layer in self.layers:
            layer.weight.data = init_vec.reshape(1, 1, 3, 3)
    def forward(self, input):
        input = input.reshape(-1, 1, 3, 3)
        x = input
        for layer in self.layers:
            x = layer(x)
        return x.reshape(-1, 9)

# A is an input matrix we're trying to factor
def factor(A, net):
    size = len(A)
    
    optimizer = optim.SGD(net.parameters(), lr = 1e-2)
    criterion = nn.MSELoss()
    
    loss_val = 1
    data = [np.array([1. if i == j else 0. for i in range(size)]) for j in range(size)]

    y = torch.Tensor([np.matmul(A, x) for x in data]).view(len(data), size)
    x = torch.Tensor(data).view(len(data), size)
    i = 0
    align_vals = []
    losses = []
    svals = []
    #print(np.linalg.svd(net.weight_matrix().detach().numpy())[1])
    
    #u_inits = []
    #vh_inits = []
    #for layer in net.layers:
    #    u, s, vh = np.linalg.svd(layer.weight_matrix().detach().numpy())
    #    u_inits.append(u)
    #    vh_inits.append(vh)
    device = 'cuda'
    print(device)
    y= y.to(device)
    x = x.to(device)
    net = net.to(device)
    while loss_val > 1e-4:
        optimizer.zero_grad()
        pred = net(x)
        #align = alignment_v2(net, u_inits, vh_inits)
        #align_vals.append(align)

        loss = criterion(y, pred)
        loss.backward()
        optimizer.step()
        loss_val = loss.data.item()
        losses.append(loss_val)
        #svals.append(np.linalg.svd(net.layers[0].weight_matrix().detach().numpy())[1])
        if i % 1000 == 0:
            print(loss_val)
        i += 1

            
if __name__ == "__main__":
    
    # 3 by 3 image, i.e. 9 x 9 matrix we are factorizing
    n = 9
    depth = 10
    # same initialization for all layers so that they begin aligned
    #init_vec = 0.1*np.random.normal(size=9)
    #init_vec[8] = init_vec[0]
    #init_vec[7] = init_vec[1]
    #init_vec[6] = init_vec[2]
    #init_vec[5] = init_vec[3]
    init_vec = np.zeros(9)
    init_vec[4] = 0.1
    #conv = DeepConv(n, depth, init_vec = init_vec)
    conv = Conv(depth, torch.Tensor(init_vec))
    #init = conv.layers[1].weight_matrix().detach().numpy()

    #Q = np.linalg.eig(init)[1]

    # A must be symmetric, psd, with eigenvalues aligned to the initialization
    #A = np.matmul(np.matmul(Q, np.diag(0.5*np.ones(n) + np.random.rand(n))), Q.transpose())
    #print(A)
    A = np.random.rand(n, n)
    factor(A, conv)
            
