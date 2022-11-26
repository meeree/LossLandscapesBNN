# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 18:44:35 2022

@author: jhazelde
"""

import torch
import argparse
import gradient_methods as gm
from torchvision import datasets, transforms
from matplotlib import pyplot as plt 
from tqdm import tqdm
import os
print(f'Using GPU: {torch.cuda.get_device_name(0)}')
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

parser = argparse.ArgumentParser(description='Train MNIST BNN')
parser.add_argument('-b', '--batch-size', default=10, type=int)
parser.add_argument('-T', '--timesteps', default=1000, type=int,
                    help='Simulation timesteps')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    help='Initial learning rate')
parser.add_argument('--dt', default=0.01, type=float,
                    help='Timestep for numerical integration')
parser.add_argument('-H', '--hidden-dim', default=100, type=int,
                    help='Dimension of hidden neuron layer')
parser.add_argument('-S', '--grad-samples', default=1, type=int,
                    help='Number of samples to use for weak gradient')
parser.add_argument('--stddev', '--standard-deviation', default=0.1, type=float,
                    help='Standard deviation to use for sampling for weak gradient')
parser.add_argument('--grad-method', default='SmoothGrad', type=str,
                    help='Gradient method to use (right now one of AutoGrad, SmoothGrad, RegGrad)')
parser.add_argument('--epochs', default=-1, type=int,
                    help='Number of training epochs. Defaults to infinite.')
parser.add_argument('--epoch-size', default=2000, type=int,
                    help='Number of training samples per epoch.')

class Trainer:
    # If grad_computer is None, use autodiff.
    def __init__(self, model, optim = None, stddev = 0.0, T_sim = -1, grad_computer = None):
        self.model = model
        self.optim = optim
        self.grad_computer = grad_computer
        self.stddev = stddev
        self.T_sim = T_sim
        
    def eval_on_batch(self, batch):
        if self.optim is not None:
            self.optim.zero_grad()
        with torch.set_grad_enabled(self.grad_computer is None):
            return self.model(batch.cuda(), self.stddev, T_sim=self.T_sim)
        
    def eval_losses(self, loss_fun, out, expected):
        mean_out = torch.mean(out, dim=2)
        target = expected.to(mean_out.device).unsqueeze(0).repeat((out.shape[0], 1, 1))
        unreduced_loss = loss_fun(mean_out, target)
        losses = torch.mean(unreduced_loss, dim=(1,2))
        return losses
    
    def eval_accuracy(self, out, expected):
        mean_out = torch.mean(out, dim=2)
        target = expected.to(mean_out.device).unsqueeze(0).repeat((out.shape[0], 1, 1))
        correct = (torch.argmax(target, -1) == torch.argmax(mean_out, -1)).float()
        return torch.mean(correct, 1) 
    
    def optim_step(self, losses):
        # Compute gradients an step optimizer with computed gradient results.
        if self.grad_computer is None: 
            losses[0].backward() # Use autodiff on first sampled loss.
        else:
            grads = self.grad_computer.compute_grad(losses, self.model.noises)
            for i in range(len(self.model.Ws)):
                self.model.Ws[i].grad = grads[i] # Manually copy gradient.
        if self.optim is not None:
            self.optim.step()
        
def load_mnist(train_batch_sz, test_batch_sz):
    from spike_train_mnist import SpikeTrainMNIST
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
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=train_batch_sz, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=test_batch_sz, shuffle=False)
    return train_loader, val_loader

def train_mnist(args):
    from bnn import Noisy_Weights_BNN
    from config import CFG
    ident = f'N-{args.epoch_size}-lr-{args.lr}-S-{args.grad_samples}-stddev-{args.stddev}-T-{args.timesteps}-dt-{args.dt}-b-{args.batch_size}-method-{args.grad_method}'
    print(f'Identifier: {ident}')
    
    CFG.n_samples_train = args.epoch_size
    CFG.n_samples_val = 500
    CFG.n_samples_test = 500
    CFG.test_batch_sz = 250
    CFG.plot = False
    CFG.plot_all = False
    CFG.dt = args.dt
    CFG.sim_t = args.timesteps
    CFG.hidden_layers = [args.hidden_dim]
    loss_fun = torch.nn.MSELoss(reduction='none').to('cuda')
    
    # Load MNIST.
    train_loader, val_loader = load_mnist(args.batch_size, 250)
    
    # Setup model.
    model = Noisy_Weights_BNN(args.grad_samples).cuda()
    
    # Setup trainer.
    optim = torch.optim.Adam(model.parameters(), lr = args.lr)
    grad_computer = None
    if args.grad_method == 'SmoothGrad':
        grad_computer = gm.SmoothGrad(model.Ws, args.stddev)
    elif args.grad_method == 'RegGrad':
        grad_computer = gm.RegGrad(model.Ws)        
    trainer = Trainer(model, optim, stddev = args.stddev, grad_computer = grad_computer)
    # trainer.model.load_state_dict(torch.load('../data/model_best_110_N-4000-lr-0.01-S-100-stddev-0.15-T-1000-dt-0.01-b-10-method-RegGrad.pt'))
    
    epochs = args.epochs
    if epochs < 0:
        epochs = int(1e10)
        
    max_accuracy = 0.0
    for e in range(epochs):
        # Evaluate.
        trainer.model.S = 1 # Only need one sample to get accuracy.
        accuracy = 0.0 
        for batch, expected in tqdm(val_loader):
            out = trainer.eval_on_batch(batch)
            accuracy += trainer.eval_accuracy(out, expected)[0]
            plt.figure(dpi=400)
            plt.plot(trainer.model.layers[1].V.detach().cpu().numpy()[0, :, :])
            plt.title(f'Sample voltage, epoch = {e}')
            plt.show()
        trainer.model.S = args.grad_samples # Reset for training.
        print('validation accuracy: ', accuracy.item() / len(val_loader))
 
        if max_accuracy < accuracy:
            max_accuracy = accuracy
            torch.save(trainer.model.state_dict(), f'../data/model_best_{e}_{ident}.pt')
            with open(f'../data/accuracy_{e}_{ident}.txt', 'w') as fl:
                print(accuracy.item() / len(val_loader), file=fl)       
        
        # Train.
        loss_record, smooth_loss_record = [], []
        for batch, expected in tqdm(train_loader):
            out = trainer.eval_on_batch(batch)
            losses = trainer.eval_losses(loss_fun, out, expected)
            trainer.optim_step(losses)
            loss_record.append(losses[0]. item())
            smooth_loss_record.append(gm.compute_smoothed_loss(losses).item())
            
        plt.figure(dpi=300)
        plt.plot(loss_record)
        plt.show()
        
def loss_landscape_mnist(args):
    from bnn import Noisy_Weights_BNN
    from config import CFG
    ident = f'N-{args.epoch_size}-lr-{args.lr}-S-{args.grad_samples}-stddev-{args.stddev}-T-{args.timesteps}-dt-{args.dt}-b-{args.batch_size}-method-{args.grad_method}'
    print(f'Identifier: {ident}')
    
    CFG.n_samples_train = 1
    CFG.n_samples_val = 1
    CFG.n_samples_test = 500
    CFG.test_batch_sz = 250
    CFG.plot = False
    CFG.plot_all = False
    CFG.dt = args.dt
    CFG.sim_t = args.timesteps
    CFG.hidden_layers = [args.hidden_dim]
    loss_fun = torch.nn.MSELoss(reduction='none').to('cuda')
    
    # Load MNIST.
    train_loader, val_loader = load_mnist(args.batch_size, CFG.n_samples_val)
    for batch, expected in val_loader:
        break
    
    # Setup model.
    model = Noisy_Weights_BNN(1).cuda()
    
    # Setup trainer.
    trainer = Trainer(model, None, stddev = args.stddev)
    
    s0 = dict(trainer.model.state_dict())
    s1 = dict(torch.load('../data//model_best_582_N-2000-lr-0.01-S-100-stddev-0.15-T-1000-dt-0.1-b-10-method-RegGrad.pt'))
    cur_state = dict(s0)
    loss_interp = []
    interpolant = torch.linspace(0.0, 1.0, 100)
    for interp in tqdm(interpolant):
        for key in cur_state:
            cur_state[key] = s0[key] + interp * (s1[key] - s0[key])
        trainer.model.load_state_dict(cur_state)
        plt.figure(dpi=400)
        plt.plot(trainer.model.layers[1].V.detach().cpu().numpy()[0, :, :])
        plt.show()
            
        out = trainer.eval_on_batch(batch)
        losses = trainer.eval_losses(loss_fun, out, expected)
        loss_interp.append(losses[0].cpu().item())
        
    plt.plot(interpolant, loss_interp)
    plt.show()
    
    
class Reservoir_Trainer:
    # If grad_computer is None, use autodiff.
    def __init__(self, reservoir, optim = None, S = 1, stddev = 0.0, grad_computer = None):
        self.model = reservoir
        self.model.W.set_params(S, stddev)
        self.optim = optim
        self.grad_computer = grad_computer
        
    def eval_on_batch(self, batch):
        if self.optim is not None:
            self.optim.zero_grad()
            
        self.model.W.noisify()
        with torch.set_grad_enabled(self.grad_computer is None):
            inp = batch.repeat(self.model.W.S, 1, 1).cuda()
            out = self.model(inp)
            return out.reshape((self.model.W.S, -1, out.shape[1], out.shape[2]))
        
    def eval_losses(self, loss_fun, out, expected):
        mean_out = torch.mean(out, dim=2)
        target = expected.to(mean_out.device).unsqueeze(0).repeat((out.shape[0], 1, 1))
        unreduced_loss = loss_fun(mean_out, target)
        losses = torch.mean(unreduced_loss, dim=(1,2))
        return losses
    
    def eval_accuracy(self, out, expected):
        mean_out = torch.mean(out, dim=2)
        target = expected.to(mean_out.device).unsqueeze(0).repeat((out.shape[0], 1, 1))
        correct = (torch.argmax(target, -1) == torch.argmax(mean_out, -1)).float()
        return torch.mean(correct, 1) 
    
    # Compute gradients and step optimizer with computed gradient results.
    def optim_step(self, losses):
        if self.grad_computer is None: 
            losses[0].backward() # Use autodiff on first sampled loss.
        else:
            grads = self.grad_computer.compute_grad(losses, [self.model.W.noise])
            self.model.W.W.grad = grads[0] # Manually copy gradient.
            
        if self.optim is not None:
            self.optim.step()
    
def reservoir_example(args):
    from bnn import HH_Complete
    from config import CFG
    import numpy as np
    CFG.dt = args.dt
    CFG.sim_t = args.timesteps
    
    res = HH_Complete(10).cuda()
    z = torch.ones((1, CFG.sim_t, 10)).cuda()
    
    for i in range(1):
        res(z)
        
    T = res.T.detach().cpu().numpy()
    
    plt.figure(dpi=500)
    plt.plot(T[0, :, :])
    plt.show()   

    W = 1000    
    rates = torch.mean(res.T[:, -W:, :], 1)
    rates = rates.detach().cpu().numpy()
    plt.figure(dpi=500)
    plt.plot(rates[0])
    plt.show()
    
    rates = np.zeros((T.shape[0], T.shape[1] - W, T.shape[2]))
    for i in range(T.shape[1] - W):
        rates[:, i, :] = np.mean(T[:, i:i+W, :])
    
    plt.figure(dpi=500)
    plt.plot(rates[0, :, :])
    plt.show()    
    
def loss_landscape_reservoir_turn_off(args):
    from bnn import HH_Complete
    from config import CFG
    CFG.dt = args.dt
    CFG.sim_t = args.timesteps

    model = HH_Complete(10).cuda()
    z = torch.ones((100, CFG.sim_t, 10)).cuda()

    # If we want to turn off the network, it is clear that an optimal solution is 
    # achieved if all weights are zero. We can analyze loss landscape by interpolation
    # to this optimum from an initial set of typical weights.
    s0 = model.W.weight.data
    s1 = torch.zeros_like(s0)
    cur_state = s0
    loss_interp = []
    interpolant = torch.linspace(0.0, 1.0, 100)
    for interp in tqdm(interpolant):
        cur_state = s0 + interp * (s1 - s0)
        model.W.weight.data = cur_state
        out = model(z)
        losses = torch.mean(out, (1,2))
        for loss in losses.detach().cpu().numpy():
            loss_interp.append(loss)
        break
        
    plt.plot(interpolant, loss_interp)
    plt.show()  
    
    return interpolant, loss_interp

def train_reservoir_turn_off(args):
    from bnn import HH_Complete
    from config import CFG
    CFG.dt = args.dt
    CFG.sim_t = args.timesteps
    
    model = HH_Complete(10).cuda()
    optim = torch.optim.Adam([model.W.W], lr = args.lr)
    grad_computer = None
    if args.grad_method == 'SmoothGrad':
        grad_computer = gm.SmoothGrad([model.W.W], args.stddev)
    elif args.grad_method == 'RegGrad':
        grad_computer = gm.RegGrad([model.W.W])        

    z = torch.ones((1, CFG.sim_t, 10)).cuda()
    expected = torch.zeros((1, 10)).cuda()
    trainer = Reservoir_Trainer(model, optim, args.grad_samples, args.stddev, grad_computer)
    loss_fun = torch.nn.MSELoss(reduction='none').to('cuda')
    
    loss_record = []
    for i in tqdm(range(1000)):
        out = trainer.eval_on_batch(z)
        losses = trainer.eval_losses(loss_fun, out, expected)
        trainer.optim_step(losses)
        loss_record.append(losses[0].item())
        
        if i % 25 == 0:
            plt.plot(loss_record)
            plt.show()
            
if __name__ == '__main__':
    args = parser.parse_args()
    train_reservoir_turn_off(args)
    #loss_landscape_mnist(args)
    #train_mnist(args)
