import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import matplotlib
import matplotlib.pyplot as plt
import pickle

class Conv(nn.Module):
    def __init__(self, depth, init=None):
        super(Conv, self).__init__()

        self.layers = nn.ModuleList([nn.Conv2d(1, 1, 3, 1, 1) for i in range(depth)])
        if init is not None:
            for layer in self.layers:
                layer.weight.data = init
            
    def forward(self, input):
        
        x = input
        for layer in self.layers:
            x = layer(x)
        return x

def layer_to_matrix(weight):
    weight = weight.reshape(-1).to('cpu')
    rows = []
    n = 28
    for row in range(n):
        for col in range(n):
            current = []
            if row > 1:
                current.append(torch.zeros(n*(row-1)))
                    
            if row > 0:
                if col > 1:
                    current.append(torch.zeros(col-1))
                if col > 0:
                    current.append(weight.narrow(0, 0, 1))
                current.append(weight.narrow(0, 1, 1))
                if col < n-1:
                    current.append(weight.narrow(0, 2, 1))
                if col < n-2:
                    current.append(torch.zeros(n - 2 - col))

            if col > 1:
                current.append(torch.zeros(col-1))
            if col > 0:
                current.append(weight.narrow(0, 3, 1))
            current.append(weight.narrow(0, 4, 1))
            if col < n-1:
                current.append(weight.narrow(0, 5, 1))
            if col < n-2:
                current.append(torch.zeros(n - 2 - col))

            if row < n-1:
                if col > 1:
                    current.append(torch.zeros(col-1))
                if col > 0:
                    current.append(weight.narrow(0, 6, 1))
                current.append(weight.narrow(0, 7, 1))
                if col < n-1:
                    current.append(weight.narrow(0, 8, 1))
                if col < n-2:
                    current.append(torch.zeros(n - 2 - col))

            if row < n-2:
                current.append(torch.zeros(n*(n - 2 - row)))

            current_row = torch.cat(current)
            if len(current_row) != n*n:
                print(len(current_row))
                print("BAD")
            
            rows.append(current_row)
   
    return torch.stack(rows, dim = 1).to('cuda')

# this is actually invariance
# normal alignment, not strong alignment
def invariance(net, u_inits, v_inits, debug=False):

    u_vals = []
    v_vals = []

    for layer in net.layers:
        u, s, v = torch.svd(layer_to_matrix(layer.weight))
        u_vals.append(u)
        v_vals.append(v)

    total_alignment = 0

    for i in range(len(net.layers)):

        if i < len(net.layers) - 1:
            res = torch.mm(torch.transpose(u_inits[i], 0, 1), u_vals[i])
            total_alignment += torch.sum(torch.max(torch.abs(res), 1)[0])

        if i > 0:
            res = torch.mm(torch.transpose(v_inits[i], 0, 1), v_vals[i])
            total_alignment += torch.sum(torch.max(torch.abs(res), 1)[0])

    if debug:
        print(total_alignment)

    return total_alignment/float(784*2*2)

def alignment(net):
    u_vals = []
    v_vals = []
    
    for layer in net.layers:
        u, s, v = torch.svd(layer_to_matrix(layer.weight))
        u_vals.append(u)
        v_vals.append(v)
        
    total_alignment = 0
    
    if len(net.layers) > 1:
        for i in range(len(net.layers) - 1):

            res = torch.mm(torch.transpose(v_vals[i+1], 0, 1), u_vals[i])
            total_alignment += torch.sum(torch.max(torch.abs(res), 1)[0])
    
    return total_alignment/float((len(net.layers) - 1)*784)


def plot(losses, aligns, filename):
    fig, ax1 = plt.subplots()

    plt.title("Autoencoding 1 training example")

    color = 'tab:red'
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Alignment', color=color)
    ax1.plot([100*i for i in range(len(aligns))], aligns, color=color)
    #ax1.set_ylim(ymin=0.7, ymax=1.05)                                                                                                                                               
    ax1.tick_params(axis='y', labelcolor=color)

    plt.plot([0, 100*len(aligns)], [1.0, 1.0], '--', color='tab:red')

    ax2 = ax1.twinx()

    color = 'tab:blue'
    ax2.set_ylabel('MSE Loss', color=color)
    ax2.plot(losses, color=color)
    ax2.set_yscale('log')
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
    plt.savefig(filename)

if __name__ == '__main__':
    start = time.time()

    print(torch.cuda.is_available())
    print(torch.version.cuda)

    batch_size = 1

    kwargs = {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=batch_size, shuffle=False, **kwargs)

    device = "cuda"

    init = np.random.randn(9)
    for i in range(4):
        init[i] = init[8 - i]
    print(init)
    init = torch.Tensor(init)
    init = init.reshape(1, 1, 3, 3)
    
    net = Conv(3)
    net = net.to(device)

    optimizer = optim.SGD(net.parameters(), lr=1e-2)

    criterion = nn.MSELoss()                                                                                                                                                        
    
    print("device = %s" % device)

    u_inits = []
    v_inits = []
    for layer in net.layers:
        u, s, v = torch.svd(layer_to_matrix(layer.weight))
        u_inits.append(u)
        v_inits.append(v)

    aligns = []
    losses = []
    invs = []

    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device)

        start = time.time()
        print(data.size())
        loss_val = 1.0
        epoch = 0

        calc_aligns = 100

        while loss_val > 1e-4 and epoch < 1e5:
            if epoch % calc_aligns == 0:
                inv = invariance(net, u_inits, v_inits)
                invs.append(inv.item())
                align = alignment(net)
                aligns.append(align.item())

            optimizer.zero_grad()
            output = net(data)

            loss = criterion(output, data) # autoencoding                                                                                                                
            loss.backward()
            optimizer.step()
            loss_val = loss.item()
            losses.append(loss_val)
            if (epoch+1)  % 100 == 0:
                print(loss_val)

            epoch += 1
        break

    for layer in net.layers:
        print(layer_to_matrix(layer.weight))
    
    with open('Autoencode_1example_randinit' + str(time.time()) + '.pkl', 'wb') as f:
        pickle.dump([losses, aligns, invs, batch_size], f)

    plot(losses, aligns, 'alignment.png')

    print("total time = %f" % (time.time() - start))
