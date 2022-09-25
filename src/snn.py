import torch, torch.nn as nn
from torch.utils.data import DataLoader
import snntorch as snn
from snntorch import surrogate
import numpy as np
from matplotlib import pyplot as plt 
import os
print(f'Using GPU: {torch.cuda.get_device_name(0)}')
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def lif_single_test(w):
    lif = snn.Leaky(beta=0.8)
    inp = torch.ones(100) * w
    V = torch.zeros(1)
    spikes, Vs = [], []
    for step in range(inp.shape[0]):
        spk, V = lif(inp[step], V)
        spikes.append(float(spk))
        Vs.append(float(V))
        
    plt.figure()
    plt.plot(Vs)
    plt.show()
    
    plt.figure()
    plt.plot(spikes)
    plt.show()
    
def lif_feedforward():
    net = nn.Sequential(
        nn.Linear(28*28, 100),
        snn.Leaky(beta=0.99, init_hidden=True),
        nn.Linear(100, 10),
        snn.Leaky(beta=0.99, init_hidden=True, output=True)
    )
    inp = torch.rand(100, 28*28)
    spikes, Vs = [], []
    for step in range(inp.shape[0]):
        spk, V = net(inp[step])
        spikes.append(spk.detach().numpy())
        Vs.append(spk.detach().numpy())   
        
    plt.plot(np.array(Vs))
    plt.show()
        
    plt.imshow(np.array(Vs).T, aspect='auto')
    plt.colorbar()
    plt.show()
    
def load_spiketrain_mnist():
    from spike_train_mnist import SpikeTrainMNIST
    from torchvision import datasets, transforms
    transform=transforms.Compose([
            transforms.ToTensor()
        ])
    train_mnist = datasets.MNIST(
        '../data/mnist_torch/',
        train=True, download=True, transform=transform,
    )
    test_mnist = datasets.MNIST(
        '../data/mnist_torch/',
        train=False, download=True, transform=transform,
    )
    train_dataset = SpikeTrainMNIST(train_mnist, 'train')
    val_dataset = SpikeTrainMNIST(test_mnist, 'validation')
    return train_dataset, val_dataset
    
def lif_mnsit():
    from config import CFG
    CFG.sim_t = 100
    spike_grad = surrogate.fast_sigmoid()
    net = nn.Sequential(
        nn.Linear(28*28, 100),
        snn.Leaky(beta=0.99, init_hidden=True, spike_grad=spike_grad),
        nn.Linear(100, 10),
        snn.Leaky(beta=0.99, init_hidden=True, spike_grad=spike_grad)
    )
    train_dset, val_dset = load_spiketrain_mnist()
    batch_size = 10
    train_loader = torch.utils.data.DataLoader(train_dset, batch_size=batch_size, shuffle=True)
    loss_fun = torch.nn.MSELoss()
    optim = torch.optim.Adam(net.parameters(), lr=0.001)
    out = torch.zeros((CFG.sim_t, batch_size, 10))
    for epoch in range(10):
        for batch, target in train_loader:
            optim.zero_grad()
            for t in range(CFG.sim_t):
                out[t, :, :] = net(batch[:, t, :])
            mean = torch.mean(out, (0, 1))
            loss = loss_fun(mean, target)
            loss.backward()
            optim.step()
    
def lif_tutorial_training(use_autodiff):
    from torchvision import datasets, transforms
    from config import CFG
    # Define Network
    # Network Architecture
    num_inputs = 28*28
    num_hidden = 1000
    num_outputs = 10
    
    # Temporal Dynamics
    num_steps = 25
    beta = 0.95
    class Net(nn.Module):
        def __init__(self, S, input_dim = 28*28, output_dim = 10, hidden_dims = [100]):
            super().__init__()
            
            self.dims = [input_dim] + hidden_dims + [output_dim]
            print('Network dimensions: ' + str(self.dims))
            self.Ws, self.Ws_noisy, self.noises, self.layers = [], [], [], []
            self.S = S
            for d1, d2 in zip(self.dims[:-1], self.dims[1:]):
                # Optimization: store transpose matrices instead of original.
                W_template = nn.Linear(d2, d1, bias=False).weight
                W_template = W_template.to('cuda')
                
                self.Ws.append(torch.nn.Parameter(W_template.detach()))
                self.Ws_noisy.append(None)
                self.noises.append(None)
                self.layers.append(snn.Leaky(beta=beta))
                    
            self.Ws = torch.nn.ParameterList(self.Ws) # Makes pytorch notice these.
            self.ticker = 0 # For picking parameter to optimize.
    
        def forward(self, batch, stddev, S_split=-1, direction=None):
            batch = batch.unsqueeze(1).repeat(1, CFG.sim_t, 1)
            
            self.ticker = (self.ticker + 1) % len(self.Ws)
            for i in range(len(self.Ws)):
                # Copies of weight for noisy sampling over batches and S samples.
                noisy_W = self.Ws[i]
                noisy_W = noisy_W.unsqueeze(0) # Batch size dimension. Need for batch multiplication.
                noisy_W = noisy_W.unsqueeze(0).repeat(self.S, 1, 1, 1) # Samples.
                self.Ws_noisy[i] = noisy_W
                
            # Add noise.        
            if stddev > 0:
                for i in range(len(self.noises)):
                    if i == self.ticker: # Only apply noise to one layer at a time.
                        self.noises[i] = torch.normal(mean=0.0, std=stddev * torch.ones_like(self.Ws_noisy[i]))
                        if direction is not None:
                            ts = torch.linspace(0.0, 1.0, self.S)
                            for s in range(self.S):
                                self.noises[i][s, 0, :, :] = ts[s] * direction.cuda()
                    else:
                        self.noises[i] = torch.zeros_like(self.Ws_noisy[i])
                    
                    self.noises[i][0, :, :, :] = 0.0 # Get loss at origin. Necessary for least squares fit.
                    self.Ws_noisy[i] += self.noises[i]
                
            if S_split < 0:
                S_split = self.S # Use all samples
                
            n_iters = (self.S + S_split - 1) // S_split # Round up
            res = torch.zeros((self.S, batch.shape[0], batch.shape[1], self.dims[-1])).cuda()
            for n in range(n_iters):
                T = batch.unsqueeze(0).repeat(S_split, 1, 1, 1)

                start, end = S_split*n, S_split*(n+1)
                for idx, (W, layer) in enumerate(zip(self.Ws_noisy, self.layers)):                    
                    # Note that W is already transposed.
                    W_split = W[start:end, :, :, :]
                    z = torch.matmul(T, W_split)
                    
                    mem = layer.init_leaky()
                    mem_rec = torch.zeros_like(z)
                    for t in range(CFG.sim_t):
                        z[:, :, t, :], mem = layer(z[:, :, t, :], mem)
                        mem_rec[:, :, t, :] = mem
                    T = z
                res[start:end, :, :, :] = mem_rec
            return res
    
    # dataloader arguments
    batch_size = 128
    data_path='../data/mnist_torch/'
    CFG.sim_t = 25
    
    dtype = torch.float
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        
    # Define a transform
    transform = transforms.Compose([
                transforms.Resize((28, 28)),
                transforms.Grayscale(),
                transforms.ToTensor(),
                transforms.Normalize((0,), (1,))])
    
    mnist_train = datasets.MNIST(data_path, train=True, download=True, transform=transform)
    mnist_test = datasets.MNIST(data_path, train=False, download=True, transform=transform)
    
    # Create DataLoaders
    train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=True, drop_last=True)
    
    # Load the network onto CUDA if available
    S = 1 if use_autodiff else 1000
    net = Net(S).to(device)
    num_epochs = 1
    loss_hist = []
    test_loss_hist = []
    counter = 0
    loss = nn.CrossEntropyLoss()
    loss_no_reduce = nn.CrossEntropyLoss(reduction='none')
    optimizer = torch.optim.Adam(net.parameters(), lr=5e-4, betas=(0.9, 0.999))
    
    def print_batch_accuracy(data, targets, train=False):
        pass
        # output, _ = net(data.view(batch_size, -1))
        # _, idx = output.sum(dim=0).max(1)
        # acc = np.mean((targets == idx).detach().cpu().numpy())
    
        # if train:
        #     print(f"Train set accuracy for a single minibatch: {acc*100:.2f}%")
        # else:
        #     print(f"Test set accuracy for a single minibatch: {acc*100:.2f}%")
    
    def train_printer():
        print(f"Epoch {epoch}, Iteration {iter_counter}")
        print(f"Train Set Loss: {loss_hist[counter]:.2f}")
        print(f"Test Set Loss: {test_loss_hist[counter]:.2f}")
        print_batch_accuracy(data, targets, train=True)
        print_batch_accuracy(test_data, test_targets, train=False)
        print("\n")
        
    # Outer training loop
    for epoch in range(num_epochs):
        iter_counter = 0
        train_batch = iter(train_loader)
    
        # Minibatch training loop
        for data, targets in train_batch:
            data = data.to(device)
            targets = targets.to(device)
                    
            optimizer.zero_grad()
            if use_autodiff:
                # forward pass
                net.train()
                spk_rec = net(data.view(batch_size, -1), stddev=0.0)
        
                # initialize the loss & sum over time
                loss_val = torch.zeros((1), dtype=dtype, device=device)
                for step in range(num_steps):
                    loss_val += loss(spk_rec[0, :, step, :], targets)
        
                # Gradient calculation
                loss_val.backward()
            else:
                print('hio')
                spk_rec = net(data.view(batch_size, -1), stddev=0.15, S_split=50)
        
                # initialize the loss & sum over time
                losses = torch.zeros(net.S, dtype=dtype, device=device)
                targets = targets.unsqueeze(0).repeat(net.S, 1)
                for step in range(num_steps):
                    predicted = torch.transpose(spk_rec[:, :, step, :],1,2)
                    l = loss_no_reduce(predicted, targets)
                    l = torch.mean(l, 1)
                    losses += l
                
                losses = losses.cpu().detach().numpy()
                l0 = losses[0] # Loss at origin. Note that Noisy_BNN always samples first point at origin!
                loss_val = l0
                for i in range(len(net.noises)):
                    noise_offset, W = net.noises[i], net.Ws[i]
                    S = noise_offset.shape[0]
                    flat_noise = noise_offset.reshape((S, -1))
                    flat_noise = flat_noise.cpu().detach().numpy()
                    M, _, _, _ = np.linalg.lstsq(flat_noise, losses - l0)
                    M = M.reshape(W.shape)
                    
                    # Assign gradient manually for optimization
                    net.Ws[i].grad = torch.from_numpy(M).to('cuda')
            optimizer.step()
    
            # Store loss history for future plotting
            loss_hist.append(loss_val.item())
    
            # Test set
            with torch.no_grad():
                net.eval() 
                test_data, test_targets = next(iter(test_loader))
                test_data = test_data.to(device)
                test_targets = test_targets.to(device)
    
                # Test set forward pass
                test_spk = net(test_data.view(batch_size, -1), 0.0)
    
                # Test set loss
                test_loss = torch.zeros((1), dtype=dtype, device=device)
                for step in range(num_steps):
                    test_loss += loss(test_spk[0, :, step, :], test_targets)   
                test_loss_hist.append(test_loss.item())
    
                # Print train/test loss/accuracy
                if counter % 50 == 0:
                    train_printer()
                counter += 1
                iter_counter +=1
                
    # Plot Loss
    plt.figure(facecolor="w", figsize=(10, 5))
    plt.plot(loss_hist)
    plt.plot(test_loss_hist)
    plt.title("Loss Curves")
    plt.legend(["Train Loss", "Test Loss"])
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.show()
                
lif_tutorial_training(False)
exit()
    
num_steps = 50 # number of time steps
batch_size = 1
beta = 0.5  # neuron decay rate
spike_grad = surrogate.fast_sigmoid()

net = nn.Sequential(
      nn.Conv2d(1, 8, 5),
      nn.MaxPool2d(2),
      snn.Leaky(beta=beta, init_hidden=True, spike_grad=spike_grad),
      nn.Conv2d(8, 16, 5),
      nn.MaxPool2d(2),
      snn.Leaky(beta=beta, init_hidden=True, spike_grad=spike_grad),
      nn.Flatten(),
      nn.Linear(16 * 4 * 4, 10),
      snn.Leaky(beta=beta, init_hidden=True, spike_grad=spike_grad, output=True)
      )

net = snn.Leaky(beta=beta, init_hidden=True, spike_grad=spike_grad, output=True)

# random input data
data_in = torch.rand(num_steps, batch_size, 1, 10)

spike_recording, state_recording = [], []

for step in range(num_steps):
    spike, state = net(data_in[step])
    spike_recording.append(spike)
    state_recording.append(state)

for record in [spike_recording, state_recording]:
    plt.figure(dpi=500)
    grid = np.array([out.detach().numpy().flatten() for out in record])
    plt.imshow(grid.T, aspect='auto', interpolation = 'none', cmap = 'hot')
    plt.xlabel('Timestep')
    plt.ylabel('Neuron')
    plt.colorbar()
    plt.show()
    
plt.figure(dpi=500)
plt.plot([r.flatten()[0] for r in state_recording])
plt.show()