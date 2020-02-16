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

class Net(nn.Module):
    def __init__(self, hidden_size, init1, init2, init3):
        super(Net, self).__init__()

        self.hidden1, self.hidden2 = hidden_size
        self.layer1 = nn.Linear(784, self.hidden1, bias=False)
        self.layer1.weight.data = init1

        self.layer2 = nn.Linear(self.hidden1, self.hidden2, bias=False)
        self.layer2.weight.data = init2

        self.layer3 = nn.Linear(self.hidden2, 10, bias=False)
        self.layer3.weight.data = init3
        
        self.layers = [self.layer1, self.layer2, self.layer3]
        self.align_total = min(784, self.hidden1, self.hidden2) + min(self.hidden1, self.hidden2, 10)
        self.inv_total = min(784, self.hidden1) + 2*min(self.hidden1, self.hidden2) + min(self.hidden2, 10)
        
    def forward(self, x):
        x = x.reshape(-1, 784)
        return self.layer3(self.layer2(self.layer1(x)))


def invariance(net, u_inits, v_inits, debug=False):

    u_vals = []
    v_vals = []
    
    for layer in net.layers:
        u, s, v = torch.svd(layer.weight)
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

    return total_alignment/(net.inv_total)

def alignment(net):
    u_vals = []
    v_vals = []

    for layer in net.layers:
        u, s, v = torch.svd(layer.weight)
        u_vals.append(u)
        v_vals.append(v)
        #print(layer.weight.size())
        #print(u.size())
        #print(v.size())
    total_alignment = 0
    
    if len(net.layers) > 1:
        for i in range(len(net.layers) - 1):

            res = torch.mm(torch.transpose(v_vals[i+1], 0, 1), u_vals[i])
            total_alignment += torch.sum(torch.max(torch.abs(res), 1)[0])

    return total_alignment/net.align_total

def plot(losses, aligns, invs, filename):
    fig, ax1 = plt.subplots()

    plt.title("MNIST Multiclass classification, depth = 2")
    
    color = 'tab:green'
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Alignment', color=color)
    ax1.plot([100*i for i in range(len(aligns))], aligns, color=color)
    #ax1.set_ylim(ymin=0.7, ymax=1.05)
    ax1.tick_params(axis='y', labelcolor=color)

    ax1.plot([100*i for i in range(len(invs))], invs, color='tab:red')
    
    plt.plot([0, 100*len(aligns)], [1.0, 1.0], '--', color='tab:red')

    ax2 = ax1.twinx()

    color = 'tab:blue'
    ax2.set_ylabel('CrossEntropy Loss', color=color)
    ax2.plot(losses, color=color)
    ax2.set_yscale('log')
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
    plt.savefig(filename)

if __name__ == '__main__':
    start = time.time()

    print(torch.cuda.is_available())
    print(torch.version.cuda)

    batch_size = 256
    hidden_size = [1024, 64]
    
    kwargs = {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=batch_size, shuffle=False, **kwargs)
    
    for batch_idx, (data, target) in enumerate(train_loader):
        X = data.reshape(784, batch_size)
        target_onehot = torch.FloatTensor(batch_size, 10)
        target_onehot.zero_()
        target_onehot.scatter_(1, target.reshape(batch_size, 1), 1)
        Y = target_onehot.reshape(10, batch_size)

        # using numpy here to get a full orthonormal basis
        U_x, _, Vh_x = np.linalg.svd(X.detach().numpy())
        U_y, _, Vh_y = np.linalg.svd(Y.detach().numpy())
        print(U_x.shape)
        diag1 = np.zeros((hidden_size[0], 784))
        np.fill_diagonal(diag1, np.random.rand(min(hidden_size[0], 784)))

        init1 = torch.Tensor(np.matmul(diag1, U_x.transpose()))

        diag2 = np.zeros((hidden_size[1], hidden_size[0]))
        np.fill_diagonal(diag2, np.random.rand(min(hidden_size)))

        init2 = torch.Tensor(diag2)
                         
        diag3 = np.zeros((10, hidden_size[1]))
        np.fill_diagonal(diag3, np.random.rand(min(hidden_size[1], 10)))

        init3 = torch.Tensor(np.matmul(U_y, diag3))
        break


    device = "cuda"
    
    net = Net(hidden_size, init1, init2, init3)
    net = net.to(device)
    
    optimizer = optim.SGD(net.parameters(), lr=1e-2)
    #criterion = nn.MSELoss()
    criterion = nn.CrossEntropyLoss()
    print("device = %s" % device)
    u_inits = []
    v_inits = []
    for layer in net.layers:
        u, s, v = torch.svd(layer.weight)
        u_inits.append(u)
        v_inits.append(v)

    aligns = []
    losses = []
    invs = []
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device)
        
        target_onehot = torch.FloatTensor(batch_size, 10)
        target_onehot.zero_()
        target_onehot.scatter_(1, target.reshape(batch_size, 1), 1)
        
        target_onehot = target_onehot.to(device)
        target = target.to(device)
        
        start = time.time()
        print(data.size())
        loss_val = 1.0
        epoch = 0

        calc_aligns = 100
        
        while loss_val > 1e-4:
            if epoch % calc_aligns == 0:
                
                inv = invariance(net, u_inits, v_inits)
                align = alignment(net)
                aligns.append(align.item())
                invs.append(inv.item())
            
            optimizer.zero_grad()
            output = net(data)

            #loss = criterion(output, target_onehot)
            loss = criterion(output, target) # for cross entropy loss
            loss.backward()
            optimizer.step()
            loss_val = loss.item()
            losses.append(loss_val)
            if (epoch)  % 100 == 0:
                print(loss_val)
                #_, preds = output.max(1)
                #accuracy = torch.sum(preds == target)
                #print("accuracy = %f" % accuracy)
            epoch += 1
        break
    
    with open('CrossEntropy_multiclass' + str(time.time()) + '.pkl', 'wb') as f:
        pickle.dump([losses, aligns, invs,  batch_size, hidden_size], f)
        
    plot(losses, aligns, invs, 'alignment.png')
    
    print("total time = %f" % (time.time() - start))
