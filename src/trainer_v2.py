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
parser.add_argument('--use-snn', default=False, type=bool,
                    help='Toggles SNN/BNN use.')

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
    
    # CFG.lif_beta = 0.99
    CFG.neuron_model = 'LIF'
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
    interpolant = torch.linspace(0.0, 0.4, 100)
    for interp in tqdm(interpolant):
        for key in cur_state:
            cur_state[key] = s0[key] + interp * (s1[key] - s0[key])
        trainer.model.load_state_dict(cur_state)
            
        out = trainer.eval_on_batch(batch)
        losses = trainer.eval_losses(loss_fun, out, expected)
        loss_interp.append(losses[0].cpu().item())
        
        # plt.figure(dpi=400)
        # plt.plot(trainer.model.layers[1].V.detach().cpu().numpy()[0, :, :])
        # plt.show()
        
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
           
def fi_curve(args, model_str = '', comp_fn = lambda x: x):
    from bnn import LIF_Complete, HH_Complete
    from config import CFG
    CFG.dt = args.dt
    CFG.sim_t = args.timesteps
    CFG.Iapp = 0.0
    
    B = 100
    L = 50
    if args.use_snn:
        model = LIF_Complete(L).cuda()
    else:
        model = HH_Complete(L).cuda()
    if len(model_str) > 0:
        model.load_state_dict(torch.load(model_str))
    model.W.noisify()
    W = 2000
    print(W)
    Iapp = torch.linspace(0.0, 2, B)

    inp = torch.zeros((B, CFG.sim_t, L)).cuda()
    for b in range(B):
        inp[b, :, :] = Iapp[b]
        inp[b, :, :-1] = 0.0

    out = model(inp)
    out = out[:, :, 0:1]
    
    for i in [B // 5, B // 2]:
        plt.plot(model.V[i, :, 0:1].detach().cpu(), linewidth = 1.0)
        plt.plot(out[i, :, 0:1].detach().cpu(), linewidth = 0.5)
        plt.title(f'Iapp = {Iapp[i].item():.1f}')
        plt.show()
    
    rate = out[:, -W:, :]
    rate = torch.logical_and(rate[:, :-1, :] < 0.5, rate[:, 1:, :] >= 0.5).float()
    rate = rate * (1000 / CFG.dt)
    rate = torch.mean(rate, 1)
    print(rate.shape)
   
    FI = rate.detach().cpu()
         
    start = 0
    end = -1
    
    print(FI.shape)
    plt.plot(Iapp[start:end], FI[start:end, :])
    plt.xlabel('Applied Current')
    plt.ylabel('Firing Rate (Hz)')
    loss = torch.mean(torch.abs(FI - comp_fn(Iapp).reshape(-1, 1)))
    plt.title(f'Weights scale: {torch.mean(torch.abs(model.W.W.data))}; loss {loss}')
    # plt.plot(Iapp, torch.mean(FI, 1), color='black', linestyle = 'dashed', linewidth=5, alpha=0.5)
    # plt.plot(Iapp[start:end], comp_fn(Iapp)[start:end])
    plt.show()

def train_reservoir_match(args):
    from bnn import HH_Complete, LIF_Complete
    from config import CFG
    torch.manual_seed(0)
    CFG.dt = args.dt
    CFG.sim_t = args.timesteps
    CFG.Iapp = 0.1
    
    L = 50
    model = LIF_Complete(L).cuda()
    # model.W.noisify()
    # optim = torch.optim.Adam([model.W.W], lr = args.lr)
    # z = torch.ones((1, CFG.sim_t, 10)).cuda()
    # out = model(z)
    # print(out.shape)
    # loss = torch.mean(out)
    # loss.backward()
    # exit()
    
    optim = torch.optim.Adam([model.W.W], lr = args.lr)
    grad_computer = None
    if args.grad_method == 'SmoothGrad':
        grad_computer = gm.SmoothGrad([model.W.W], args.stddev)
    elif args.grad_method == 'RegGrad':
        grad_computer = gm.RegGrad([model.W.W])        

    trainer = Reservoir_Trainer(model, optim, args.grad_samples, args.stddev, grad_computer)
    loss_fun = torch.nn.MSELoss(reduction='none').to('cuda')
    
    B = args.batch_size
    W = 400
    match_fn = lambda x: (1 - torch.cos(2 * torch.pi * x)) / 2.0
    
    loss_record = []
    for i in tqdm(range(1000)):
        s = torch.rand(B).cuda()
        z = torch.ones((B, CFG.sim_t, L)).cuda() * s.reshape((B,1,1))
        z[:, :, :-1] = 0.0 # Only input to final neuron
        match = match_fn(s)
        expected = torch.ones((B, 1)).cuda() * match.reshape((B, 1))
        
        out = trainer.eval_on_batch(z)
        rate = out[:, :, -W:, 0:1]
        
        # Discrete rate (more precise/interpretable).
        # rate = torch.logical_and(rate[:, :, :-1, :] < 0.5, rate[:, :, 1:, :] >= 0.5).float()
        # rate = rate * (1000 / CFG.dt)
        # expected = expected * 1000
        
        target = expected.reshape((1, -1, 1)).repeat(rate.shape[0], 1, 1)
        losses = (torch.mean(rate, 2) - target)**2
        losses = torch.mean(losses, (1, 2))
     #   losses = trainer.eval_losses(loss_fun, rate, expected)
        trainer.optim_step(losses)
        loss_record.append(losses[0].item())
        print(torch.mean(expected).item(), torch.mean(rate).item())
        
        if i % 50 == 0:
            model_str = f'../data/reservoir{L}_match_{i}_cos_inout2.pt'
            torch.save(model.state_dict(), model_str)
            plt.plot([l**0.5 for l in loss_record])
            plt.show()
            
            plt.plot(out[0, 0, :, 0].cpu().detach())
            plt.show()
            
            fi_curve(args, model_str, match_fn)
 
   
def train_reservoir_mean_integrate(args):
    from bnn import HH_Complete, LIF_Complete
    from config import CFG
    # torch.manual_seed(0)
    CFG.dt = args.dt
    CFG.sim_t = args.timesteps
    CFG.Iapp = 0.0
    
    L = 50
    model = LIF_Complete(L).cuda()
    # model.load_state_dict(torch.load('../data/reservoir50_mean_200_0.pt'))
    # model.W.noisify()
    # optim = torch.optim.Adam([model.W.W], lr = args.lr)
    # z = torch.ones((1, CFG.sim_t, 10)).cuda()
    # out = model(z)
    # print(out.shape)
    # loss = torch.mean(out)
    # loss.backward()
    # exit()
    
    optim = torch.optim.Adam([model.W.W], lr = args.lr)
    grad_computer = None
    if args.grad_method == 'SmoothGrad':
        grad_computer = gm.SmoothGrad([model.W.W], args.stddev)
    elif args.grad_method == 'RegGrad':
        grad_computer = gm.RegGrad([model.W.W])        

    trainer = Reservoir_Trainer(model, optim, args.grad_samples, args.stddev, grad_computer)
    loss_fun = torch.nn.MSELoss(reduction='none').to('cuda')
    
    B = args.batch_size
    W = 30 # Very small window because we want to have some notion of memory
    delay = 0

    loss_record = []
    for i in tqdm(range(1000)):
        s = torch.rand(B).cuda()
        mean = torch.ones((B, CFG.sim_t, L)).cuda() * s.reshape((B,1,1))
        z = torch.normal(mean, 0.0)
        
        if False: # Plot integral
            integral = torch.zeros(CFG.sim_t)
            integral[0] = z[0,0,0]
            for p in range(1, CFG.sim_t):
                integral[p] = integral[p-1] + z[0, p, 0]
                
            integral /= CFG.sim_t
            plt.subplot(2,1,1)
            plt.plot(z[0, :, 0])
            plt.xticks([])
            plt.axhline(s[0], 0, CFG.sim_t)
            plt.title(s[0].item())
            plt.subplot(2,1,2)
            plt.plot(integral)
            plt.title(integral[-1].item())
            plt.show()        
        
        z[:, :, :-1] = 0.0 # Only input to final neuron
        if delay > 0:
            z[:, -delay:, :] = 0.0 # Turn off for final interval
        expected = torch.ones((B, 1)).cuda() * s.reshape((B, 1))
        
        out = trainer.eval_on_batch(z)
        rate = out[:, :, -W:, 0:1]
        
        # Discrete rate (more precise/interpretable).
        # rate = torch.logical_and(rate[:, :, :-1, :] < 0.5, rate[:, :, 1:, :] >= 0.5).float()
        # rate = rate * (1000 / CFG.dt)
        # expected = expected * 1000
        
        target = expected.reshape((1, -1, 1)).repeat(rate.shape[0], 1, 1)
        losses = (torch.mean(rate, 2) - target)**2
        losses = torch.mean(losses, (1, 2))
     #   losses = trainer.eval_losses(loss_fun, rate, expected)
        loss_record.append(losses[0].item())
        print(torch.mean(torch.abs(torch.mean(rate, 2) - target)), losses[0].item())
        
        if i % 50 == 0:
            model_str = f'../data/reservoir{L}_mean_{i}_{delay}.pt'
            torch.save(model.state_dict(), model_str)
            plt.plot([l**0.5 for l in loss_record])
            plt.show()
                   
            plt.subplot(2,1,1)
            plt.plot(z.cpu().detach()[:, :, -1].transpose(0,1))
            plt.subplot(2,1,2)
            integral = torch.zeros(CFG.sim_t)
            integral[0] = out[0, 0, 0, 0]
            for i in range(1, CFG.sim_t):
                integral[i] = out[0, 0, i, 0] + integral[i-1]
                mn = 0 if i <= W else i - W
                integral[i] = torch.mean(out[0, 0, mn:i, 0])
            # integral /= CFG.sim_t
            plt.plot(integral.cpu().detach())
            # plt.plot(z[0, :, -1].cpu().detach())
            plt.axhline(s[0].cpu().item(), 0, CFG.sim_t)
            # plt.plot(out[0, :, :, 0].cpu().detach().transpose(0,1), linewidth = 0.9)
            plt.show()
            
            fi_curve(args, model_str)  
        
        trainer.optim_step(losses)
        
def plot_weight_change(model, A, B, L):
    import glob
    fls = glob.glob('../data/wang_RUN2_50_*.pt')
    fls = [fls[i] for i in [0, 1, 2] + list(range(3, len(fls), 3))]
    Ws = []
    means = []
    for fl in fls:
        model.load_state_dict(torch.load(fl))
        W = model.W.W.data.cpu().detach()
        Ws.append(W)
        mean_grid = torch.zeros((3,3))
        inds = [0, 15, 30, L]
        for i in range(3):
            for j in range(3):
                mean_grid[i, j] = torch.mean(W[inds[i]:inds[i+1], inds[j]:inds[j+1]])
        means.append(mean_grid)
        
    vmin = min([torch.min(mean) for mean in means])
    vmax = max([torch.max(mean) for mean in means])
    abs_max = max(abs(vmin), abs(vmax))
    vmin = -abs_max; vmax = abs_max
    
    plt.figure(figsize = (3,3))
    for idx in range(len(fls)):  
        plt.subplot(3, 3, 1 + idx)
        W = means[idx]
        im = plt.imshow(W, cmap='PiYG', vmin = vmin, vmax = vmax)
        plt.box(False); plt.xticks([]); plt.yticks([])
        if idx == 0:
            plt.xticks([0, 1, 2], ['A', 'B', 'ext'])
            plt.gca().xaxis.tick_top()
            plt.yticks([0, 1, 2], ['A', 'B', 'ext'])
        # plt.axhline(14.5, 0, L, color = 'black')
        # plt.axhline(29.5, 0, L, color = 'black')
        # plt.axvline(14.5, 0, L, color = 'black')
        # plt.axvline(29.5, 0, L, color = 'black')      
        
    fig = plt.gcf()
    fig.subplots_adjust(right=0.8)  
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    plt.colorbar(im, cax=cbar_ax)
    plt.show()
        
def wang_task_train(args, train = True, linear_mu = False):
    from bnn import HH_Complete, LIF_Complete
    from config import CFG
    from torchviz import make_dot
    # torch.manual_seed(0)
    CFG.dt = args.dt
    CFG.sim_t = args.timesteps
    # CFG.Iapp = 0.5
    # CFG.lif_beta = 0.99
    
    L = 50
    A = range(0, 15)
    B = range(15, 30)
    
    if args.use_snn:
        model = LIF_Complete(L).cuda()
    else:
        model = HH_Complete(L).cuda()
    # model.load_state_dict(torch.load('../data/wang_RUN2_50_950.pt'))
    
    if not train:
        plot_weight_change(model, A, B, L)
    
    optim = torch.optim.Adam([model.W.W], lr = args.lr)
    grad_computer = None
    if args.grad_method == 'SmoothGrad':
        grad_computer = gm.SmoothGrad([model.W.W], args.stddev)
    elif args.grad_method == 'RegGrad':
        grad_computer = gm.RegGrad([model.W.W])        

    trainer = Reservoir_Trainer(model, optim, args.grad_samples, args.stddev, grad_computer)    
    bsize = args.batch_size
    std = 0.5
    firing_window = CFG.sim_t // 10
    select_rate, deny_rate = 0.8, 0.2 # We want rate to be equal to select_rate if we select this category and equal to deny_rate if not.
    
    loss_fun = torch.nn.MSELoss(reduction='none').cuda()
    loss_record = []
    for i in tqdm(range(1000)):
        mu_a, mu_b = torch.rand(bsize).cuda(), torch.rand(bsize).cuda()
        if linear_mu:
            mu_a, mu_b = torch.linspace(0, 1.0, bsize).cuda(), torch.zeros(bsize).cuda()
        z = torch.normal(torch.zeros(bsize, CFG.sim_t, L), std).cuda()
        z[:, :, A] += mu_a.reshape((bsize, 1, 1)) 
        z[:, :, B] += mu_b.reshape((bsize, 1, 1))
        
        out = trainer.eval_on_batch(z).reshape((-1, CFG.sim_t, L))
        window = out[:, -firing_window:, :]
        
        rA = torch.mean(window[:, :, A], (1, 2))
        rB = torch.mean(window[:, :, B], (1, 2))
        
        targetA = (mu_a > mu_b).float() * (select_rate - deny_rate) + deny_rate
        targetB = (mu_a < mu_b).float() * (select_rate - deny_rate) + deny_rate
        targetA = targetA.repeat(args.grad_samples)
        targetB = targetB.repeat(args.grad_samples)
        
        losses = 0.5 * (loss_fun(rA, targetA) + loss_fun(rB, targetB))
        losses = torch.mean(losses.reshape((-1, bsize)), 1) # Mean over batches, not grad samples.
        loss_record.append(losses[0].item())
        
        accuracy = ((rA[:bsize] > rB[:bsize]) == (mu_a > mu_b)).float()            
        
        if train:
            if i % 5 == 0:
                print(f'accuracy: {accuracy.mean().item()}')
                model_str = f'../data/wang_BNN_{L}_{i}.pt'
                torch.save(model.state_dict(), model_str)
                plt.plot(loss_record)
                plt.show()
                      
                fi_curve(args, model_str)  
    
            trainer.optim_step(losses)
        else:
            print(f'accuracy: {accuracy.mean().item()}')
            if linear_mu:
                # Compute accuracy versus mean.
                accuracy = accuracy.detach().cpu()
                bins = torch.linspace(0.0, 1.0, 30)
                hist = torch.zeros_like(bins)
                stride = bsize // len(bins)
                
                for k in range(len(bins)):
                    hist[k] = torch.mean(accuracy[stride*k:stride*(k+1)])
            
                plt.plot(bins[:-1], hist[:-1], '-o')
                plt.xlabel('$|\mu_A - \mu_B|$') 
                plt.ylabel('Accuracy')
                plt.show()
                
                # Compute reaction time versus mean.
                thresh = 0.7
                timesteps = torch.argmax((torch.mean(out[:, :, A], 2) > thresh).float(), 1)
                reaction_times = timesteps * CFG.dt 
                plt.plot(mu_a.cpu().detach(), reaction_times.cpu().detach(), '.')
                plt.ylabel('Reaction time (ms)')
                plt.xlabel('$|\mu_A - \mu_B|$')
                
                import numpy as np
                fit = np.polyfit(mu_a.cpu().detach().numpy(), reaction_times.cpu().detach().numpy(), 4)
                p = np.poly1d(fit)
                plt.plot(mu_a.cpu().detach(), p(mu_a.cpu().detach().numpy()))
                plt.show()
            
            # Analyze neuron responses.
            for wind in [1, firing_window]:
                rate_all_A = torch.zeros((bsize, CFG.sim_t))
                rate_all_B = torch.zeros((bsize, CFG.sim_t))
                for i in range(CFG.sim_t):
                    mx = min(CFG.sim_t, wind + i)
                    rate_all_A[:, i] = torch.mean(out[:, i:mx, A], (1,2))
                    rate_all_B[:, i] = torch.mean(out[:, i:mx, B], (1,2))
                        
                outc = out.cpu().detach()
                for b in range(1):
                    pop_matrix = outc[b, :, :31].transpose(0,1)
                    pop_matrix[A, :] = 1 - pop_matrix[A, :]
                    pop_matrix[B, :] += 1 # This is just a trick to make imshow color two pops differently.
                    plt.imshow(pop_matrix, cmap='bwr', aspect='auto', interpolation = 'none')
                    plt.show()
                    
                    plt.title(f'$\mu_A =$ {mu_a[b].item():.2f}, $\mu_B =$ {mu_b[b].item():.2f}')
                    plt.plot(rate_all_A.detach().cpu()[b, :], color = 'blue')
                    plt.plot(rate_all_B.detach().cpu()[b, :], color = 'red')
                    plt.show()
            return
        
            
if __name__ == '__main__':
    args = parser.parse_args()
    wang_task_train(args)
    exit()
    # fi_curve(args, '../data/reservoir50_mean_250.pt', lambda x : x**2)
    wang_task_train(args)

    exit()
    #train_mnist(args)
