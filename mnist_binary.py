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
    def __init__(self, hidden_size, init1, init2):
        super(Net, self).__init__()

        self.hidden = hidden_size
        self.layer1 = nn.Linear(784, self.hidden, bias=False)
        self.layer1.weight.data = init1

        self.layer2 = nn.Linear(self.hidden, 1, bias=False)
        self.layer2.weight.data = init2
        self.layers = [self.layer1, self.layer2]
        self.align_total = float(2*min(784, self.hidden) + 2*min(self.hidden, 1))
        
    def forward(self, x):
        x = x.reshape(-1, 784)
        return self.layer2(self.layer1(x))


def alignment(net, u_inits, v_inits, debug=False):
    print("alignment:")
    u_vals = []
    v_vals = []
    
    for layer in net.layers:
        u, s, v = torch.svd(layer.weight)
        u_vals.append(u)
        v_vals.append(v)

            
    total_alignment = 0

    for i in range(len(net.layers)):

        #res = torch.mm(torch.transpose(u_inits[i], 0, 1), u_vals[i])
        #total_alignment += torch.sum(torch.max(torch.abs(res), 1)[0])
        total_alignment += torch.dot(torch.transpose(u_inits[i], 0, 1)[0], torch.transpose(u_vals[i], 0, 1)[0])
        print(total_alignment)
        
        #res = torch.mm(torch.transpose(v_inits[i], 0, 1), v_vals[i])
        #total_alignment += torch.sum(torch.max(torch.abs(res), 1)[0])
        total_alignment += torch.dot(torch.transpose(v_inits[i], 0, 1)[0], torch.transpose(v_vals[i], 0, 1)[0]) 
        print(total_alignment)
                
    #return total_alignment/net.align_total
    return total_alignment/float(2*len(net.layers))

def plot(losses, aligns, filename):
    fig, ax1 = plt.subplots()

    plt.title("MNIST Multiclass classification, depth = 2")

    color = 'tab:red'
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Alignment', color=color)
    ax1.plot([100*i for i in range(len(aligns))], aligns, color=color)
    #ax1.set_ylim(ymin=0.7, ymax=1.05)
    ax1.tick_params(axis='y', labelcolor=color)

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

    batch_size = 8
    hidden_size = 4
    
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
        Y = (target < 5).float().reshape(1, batch_size)
        print(Y)
         # using numpy here to get a full orthonormal basis
        U, _, Vh = np.linalg.svd(torch.mm(Y, torch.transpose(X, 0, 1)).detach().numpy())

        diag1 = np.zeros((hidden_size, 784))
        vals = np.zeros(min(hidden_size, 784))
        vals[0] = np.random.rand()
        np.fill_diagonal(diag1, vals)
        
        init1 = torch.Tensor(np.matmul(diag1, Vh))
        diag2 = np.zeros((1, hidden_size))
        vals = np.zeros(min(hidden_size, 1))
        vals[0] = np.random.rand()
        np.fill_diagonal(diag2, vals)

        init2 = torch.Tensor(diag2)
        break


    device = "cuda"
    
    net = Net(hidden_size, init1, init2)
    net = net.to(device)
    
    optimizer = optim.SGD(net.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    #criterion = nn.CrossEntropyLoss()
    print("device = %s" % device)
    u_inits = []
    v_inits = []
    for layer in net.layers:
        u, s, v = torch.svd(layer.weight)
        u_inits.append(u)
        v_inits.append(v)
        print(layer.weight.size())
        print(u.size())
        print(v.size())

    aligns = []
    losses = []
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device)
        target = (target < 5).float().reshape(-1, 1)
        target = target.to(device)
        print(target)
        start = time.time()
        print(data.size())
        loss_val = 1.0
        epoch = 0

        calc_aligns = 100
        
        while loss_val > 1e-4:
            if epoch % calc_aligns == 0:
                
                align = alignment(net, u_inits, v_inits)
                aligns.append(align.item())
            
            optimizer.zero_grad()
            output = net(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            loss_val = loss.item()
            losses.append(loss_val)
            if (epoch+1)  % 100 == 0:
                print(loss_val)
                            
            epoch += 1
        break
    
    with open('MSE_binary.pkl', 'wb') as f:
        pickle.dump([losses, aligns, batch_size, hidden_size], f)
        
    plot(losses, aligns, 'alignment.png')
    
    print("total time = %f" % (time.time() - start))
