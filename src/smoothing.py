#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 24 15:29:15 2022

@author: jhazelde
"""

from train_bbp import Trainer
from config import CFG
import config
import numpy as np
import matplotlib.pyplot as plt
import json
from os.path import exists
import torch
from bnn import *
from time import time
import scipy.stats
import os
from torchviz import make_dot
from sklearn.linear_model import Ridge
from config import print_cuda_mem
from progressbar import ProgressBar
from gradient_methods import *
print(f'Using GPU: {torch.cuda.get_device_name(0)}')
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def test_optimized_sampling_ann(random_w):
    w = torch.ones(1)
    S = 10
    eta = 0.1
    torch.manual_seed(0)
    losses = []
    for i in range(1000):
        inp = torch.rand(1)
        
        if random_w:
            ws = torch.normal(w * torch.ones(S), 0.5)
            ws[0] = w
            out = torch.nn.functional.sigmoid(inp * ws)
            loss = out
            grad = compute_grad_regression(loss, [ws], [w])[0]
        else:
            zs = torch.normal(w * inp * torch.ones(S), 0.5)
            zs[0] = w * inp
            out = torch.nn.functional.sigmoid(zs)
            loss = out
            dl_dz = compute_grad_regression(loss, [zs], [zs[0]])[0]
            grad = dl_dz * inp

        losses.append(loss[0])
        w = w - eta * grad
    
    plt.plot(losses)
    plt.show()
        
# test_optimized_sampling_ann(True)
# test_optimized_sampling_ann(False)
# exit()
   
def simulate_simple_snn():
    L = 10
    CFG.sim_t = 25
    snn = LIF(L)
    inp = 0.4 * torch.ones((1, CFG.sim_t, L)).to('cuda')
    out = snn(inp)
    plt.plot(out.cpu().detach()[0, :, :])

CFG.n_samples_val = 500
CFG.test_batch_sz = 250
CFG.train_batch_sz = 5
CFG.plot = False
CFG.plot_all = False
CFG.dt = 0.1
CFG.sim_t = 1000
#CFG.use_DNN = True

def single_hh_loss():
    CFG.Iapp = 0.0
    # delta_t = torch.tensor(1e-3)
    # tau = torch.tensor(2e-3)
    # beta = torch.exp(-delta_t/tau)
    beta = 0.3
    CFG.lif_beta = beta
    print(beta)                     
                
    S = 5000
    hh1 = LIF(1)
    hh = LIF(1)
    ws = torch.linspace(0.3, 1.5, S)
    ws.requires_grad = True
    inp = torch.ones((ws.shape[0], CFG.sim_t, 1)).cuda()
    for i in range(ws.shape[0]):
        inp[i, :, :] = ws[i] 
                
        
    T = inp
    for i in range(10):
        T = hh1(T + torch.normal(torch.zeros_like(inp), 0.01))
    out = hh(T)

    plt.figure(figsize = (15, 3))    
    for p, idx in enumerate(range(0, S, 1000)):
        plt.subplot(1, 5, p + 1) 
        plt.plot(hh.V[idx, :200, 0].detach().cpu().numpy().T, linewidth = 0.8)
        plt.title(f'w = {ws[idx]: .3f}')
    plt.show()
    
    deriv = 1.0 / ((1 + 25 * hh.V)**2)
    val = 26 *  hh.V / (1 + 25 * hh.V)
    idx = 0
    plt.figure(dpi=500)
    ax = plt.gca()
    plt.plot(hh.V[S//2, 200:250, 0].detach().cpu(), linewidth = 0.8, label='voltage')
    plt.ylabel('Voltage value')
    
    ax2 = ax.twinx()
    ax2.plot(val[S//2, 200:250, 0].detach().cpu(), linewidth = 0.8, color='red', label='surrogate')
    plt.ylabel('Surrogate value')
    
    plt.show()
    
    out = out[:, 200:, :]
    loss = torch.mean(out, (1, 2))
    loss[S//2].backward()
   # loss = torch.mean(val, (1, 2))
  #   loss[S//2] = ws[S//2]
  #   make_dot(loss[S//2], params={'w': ws[S//2]}, show_attrs=True, show_saved=True).render("out", format="pdf")
  #   exit()
    
    plt.figure(500)
    plt.plot(ws.detach(), loss.detach().cpu())
    plt.title(f'$\\beta$ = {CFG.lif_beta: .3f}, surrogate gradient at center = {ws.grad[S//2]: .3f}')
    # plt.xticks([0, len(ws) - 1], [ws[0].detach().item(), ws[len(ws)-1].detach().item()])
    plt.xlabel('w')
    plt.ylabel('Loss')
    plt.show()
    
# single_hh_loss()
# exit()

def toy_3_dim(S):
    def f(x):
        return x[0] + 2 * x[1] - 0.01 * x[2]
    
    smpls = np.random.normal(0.0, 1.0, size=(S, 3))
    out = np.apply_along_axis(f, 1, smpls)
    m, _, _, _ = np.linalg.lstsq(smpls, out)
    plt.figure(dpi=500)
    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.scatter(smpls[:, i], out, s=1)
        X = np.linspace(np.min(smpls), np.max(smpls), 100)
        plt.plot(X, m[i] * X)
        plt.title(f'{m[i]:.2f}')
    plt.show()

def plot_weights(trainer):
    if trainer.model.Ws[0].shape != 28 * 28:
        return
    
    for W in trainer.model.Ws:
        plt.figure(dpi=500)
        plt.imshow(W.cpu().detach().numpy(), aspect='auto', cmap='gist_rainbow')
        plt.colorbar()
        plt.show()
        
    print(trainer.model.Ws[0].cpu().detach().numpy().shape)
    W1 = trainer.model.Ws[0].cpu().detach().numpy().reshape((28, 28, -1))
    plt.figure(dpi=600)
    for i in range(10):
        for j in range(10):
            plt.subplot(10, 10, i*10 + j + 1)
            W = W1[:, :, i*10 + j]
            plt.imshow(W, aspect='auto', cmap='seismic')
            plt.box(False)
            plt.xticks([])
            plt.yticks([])
    plt.show()
    
def load_noisy_to_trainer(model_str):
     # Copy parameters from noisy BNN to normal BNN for evaluation using Trainer method.
    data = torch.load(model_str)
    trainer = Trainer(False)
    print(data['Ws.0'].shape)
    trainer.model = Noisy_BNN(1, data['Ws.0'].shape[0])
    trainer.model.load_state_dict(data)
    return trainer

def noisy_to_not(model_str):
    # Copy parameters from noisy BNN to normal BNN for evaluation using Trainer method.
    data = torch.load(model_str)
    trainer = Trainer(False)
    for i in range(len(trainer.model.Ws)):
        # Have to transpose because NoisyBNN stores transposed raw weights, unlike nn.Linear.
        trainer.model.Ws[i].weight.data = torch.transpose(data[f'Ws.{i}'], 0, 1)
    return trainer
      
def evaluate(model_str):
    trainer = noisy_to_not(model_str)
    return trainer.validate()
      
def train(use_autodiff, S = 500, plot = False, plot_debug = False, n_samples = 1, log_timing = False, STD=1e-2, T_sim=-1, epochs = 1, turn_off = False):
    if use_autodiff:
        STD = 0.0
        S = 1
    CFG.lr = 0.1
    CFG.n_samples_train = n_samples
    CFG.identifier = str(n_samples) + f'_MNIST_LR{CFG.lr}_{use_autodiff}_{S}_{STD}'
    
    torch.manual_seed(0)
    trainer = Trainer()
    accuracy_out = open(f'../data/accuracy_{trainer.identifier}.txt', 'w')

    if turn_off:
        CFG.Iapp = 0.0
        CFG.hidden_layers = [1 for i in range(0)]
        trainer.model = Noisy_Weights_BNN(S, 1, 1).to('cuda')
        for i in range(len(trainer.model.Ws)):
            trainer.model.Ws[i] = torch.ones_like(trainer.model.Ws[i])
    else:
        # CFG.Iapp = 0.3
        CFG.hidden_layers = [100]
        trainer.model = Noisy_Weights_BNN(S).to('cuda')
        
        # trainer.model.load_state_dict(torch.load('../data/model_19_59_19___10_21_2022_2000_MNIST_LR0.002_False_1000_0.05_0_191.pt'))
        # W2 = trainer.model.Ws[1].clone()
        # trainer.model.load_state_dict(torch.load('../data/model_19_59_19___10_21_2022_2000_MNIST_LR0.002_False_1000_0.05_0_1.pt'))
        # direction = W2 - trainer.model.Ws[1]
        
    trainer.optimizer = torch.optim.Adam(trainer.model.parameters(), lr = CFG.lr)
    loss_log, grad_log, weight_log = [], [[] for w in trainer.model.Ws], [[] for w in trainer.model.Ws]
    smooth_loss_log = []
    pbar = ProgressBar()
    if plot:
        plot_weights(trainer)
        
    for e in range(epochs):
        print(f"ON EPOCH: {e}")
        dataloader = torch.utils.data.DataLoader(trainer.train_dataset, batch_size=50, shuffle=True)
        for idx, (batch, expected) in enumerate(dataloader):
            trainer.optimizer.zero_grad()
            with torch.set_grad_enabled(use_autodiff):
                t0 = time()
                loss_fun = torch.nn.MSELoss(reduction='none').to('cuda')
                if turn_off:
                    batch = torch.ones((batch.shape[0], batch.shape[1], trainer.model.Ws[0].shape[0])).cuda()
                    trainer.model.ticker = len(trainer.model.Ws) - 2
                out = trainer.model(batch.to('cuda'), STD, T_sim=T_sim)
                mean_out = torch.mean(out, dim=2)
                target = expected.to('cuda').unsqueeze(0).repeat((out.shape[0], 1, 1))
                unreduced_loss = mean_out # Turn neurons off benchmark
                # if not turn_off:
                    # unreduced_loss = loss_fun(mean_out, target) # MNIST benchmark
                # losses = torch.mean(unreduced_loss, dim=(1,2))
                
                losses = (torch.argmax(target, 2) == torch.argmax(mean_out, 2)).float()
                losses = 1 - torch.mean(losses, 1)
                if log_timing:
                    print(f'Evaluation time : {time() - t0}')
    
            # Approximate gradient descent using regression.
            if not use_autodiff:
                t0 = time()
                loss_log.append(losses[0].cpu().item())
                smooth_loss_log.append(compute_smoothed_loss(losses).cpu().item())
                grads = compute_grad_log_trick(losses, trainer.model.noises, trainer.model.Ws, STD)
                for i in range(len(trainer.model.Ws)):
                    trainer.model.Ws[i].grad = grads[i]
                    
                if log_timing:
                    print(f'Gradient compute time : {time() - t0}')
                
                if plot_debug and idx % 1 == 0:
                    def isolated_scatter(inds):
                        l0 = losses[0].cpu().item()
                        X = trainer.model.noises[1][:,0,inds[0],inds[1]].reshape(-1).detach().cpu().numpy()
                        plt.scatter(X, losses, s=1.3, zorder=1, color='black')
                        plt.scatter(X[0], l0, s=1.5, zorder=2, color='red')
                        X = np.linspace(np.min(X), np.max(X), 100)
                        m = trainer.model.Ws[1].grad[inds[0],inds[1]].cpu()
                        plt.plot(X, m * X + l0, zorder=0)
                        plt.title(f"{m.item():.3f}")
                        plt.xlabel('$w^2_{' + f'{inds[0],inds[1]}' + '}$')
                        plt.ylabel('Loss') 
                
                    plt.figure(dpi=500)
                    plt.subplot(1,3,1)
                    isolated_scatter([0,0])
                    plt.subplot(1,3,2)
                    isolated_scatter([0,1])            
                    plt.subplot(1,3,3)
                    isolated_scatter([0,5])
                   
                    plt.show()
            else:
                t0 = time()
                loss = losses[0]
                loss_log.append(loss.cpu().item())
                smooth_loss_log.append(loss_log[-1])
                loss.backward()
                if log_timing:
                    print(f'Autodiff time : {time() - t0}')
                        
            # Optimization step.
            t0 = time()
            trainer.optimizer.step()
            if log_timing:
                print(f'Optimization time : {time() - t0}')
                
            if (idx-1) % 50 == 0:
                model_str = f'../data/model_{trainer.identifier}_{e}_{idx}.pt'
                torch.save(trainer.model.state_dict(), model_str)
                print(f'{idx}: ', end='')
                
                if plot:
                    plt.plot(loss_log)
                    plt.title('Singular Loss')
                    plt.show()
                    plt.plot(smooth_loss_log)
                    plt.title('Smooth Loss')
                    plt.show()
                                  
                    V = trainer.model.layers[0].V.detach().cpu().numpy()
                    plt.plot(V[0, :, :])
                    plt.show()
                    
            if idx == len(dataloader) - 1:
                accuracy = evaluate(model_str)
                print(f'{e}, {idx}, {accuracy}, {loss_log[-1]}', file=accuracy_out)
                print(f'{e}, {idx}, {accuracy}, {loss_log[-1]}')
                accuracy_out.flush()
    
            for i in range(len(trainer.model.Ws)):
                grad_log[i].append(trainer.model.Ws[i].grad.cpu().numpy())
                weight_log[i].append(trainer.model.Ws[i].detach().cpu().numpy())
      
    accuracy_out.close()
    if plot:
        plot_weights(trainer)
    return loss_log, grad_log, weight_log

def plot_accuracy(fl_name, title=''):
    results = np.genfromtxt(fl_name, delimiter=',')
    plt.figure(dpi=500)
    plt.plot(results[:,2])
    ticks, inds = np.unique(results[:, 0], return_index=True)
    ticks = ticks.astype(int)
    plt.xticks(inds, ticks)
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.show()

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! BEST RESULTS SO FAR !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# plot_accuracy('../data/accuracy_17_31_09___11_04_2022_2000_MNIST_LR0.002_False_100_0.05.txt', 'SNN?')
# plot_accuracy('../data/accuracy_22_56_06___08_17_2022_2000_BASELINE_LR0.1_False_1000_0.15_dt0.05.txt', '50 ms timeframe')
# plot_accuracy('../data/accuracy_18_32_21___08_16_2022_2000_BASELINE_LR0.1_False_1000_0.15.txt', '10 ms timeframe')
# plot_accuracy('../data/accuracy_15_21_46___08_17_2022_2000_BASELINE_LR0.1_False_1000_0.15_dt0.1.txt', '100 ms timeframe')
# exit()
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

def evaluate_losses_along_direction(dts, S = 500, T_sim=-1):
    torch.manual_seed(0)
    trainer = Trainer()
    trainer.model = Noisy_BNN(S).to('cuda')
    direction = torch.ones_like(trainer.model.Ws[1])
    dataloader = torch.utils.data.DataLoader(trainer.train_dataset, batch_size=10, shuffle=True)
    loss_fun = torch.nn.MSELoss(reduction='none').to('cuda')
    for batch, expected in dataloader:
        target = expected.to('cuda').unsqueeze(0).repeat((S, 1, 1))
 
    losses = np.zeros((len(dts), S))
    for dt_idx, dt in enumerate(dts):
        CFG.dt = dt
        trainer.model.ticker = 0 # Force changes along Ws[1]. Ticker is incremented in forward, so set it to 0 here.
        out = trainer.model(batch.to('cuda'), 0.0, T_sim=T_sim, direction=direction)
        mean_out = torch.mean(out, dim=2)
        unreduced_loss = loss_fun(mean_out, target) # MNIST benchmark
        loss_line = torch.mean(unreduced_loss, dim=(1,2))
        losses[dt_idx, :] = loss_line.detach().cpu().numpy()
    return losses
    
# losses = evaluate_losses_along_direction([0.01], S = 1000, T_sim = 50)
# plt.plot(losses[0, :])
# plt.show()
# exit()
    
def train_and_compare():
    CFG.test_batch_size = 500
    # true_loss_log, true_grad_log, true_ws = train(True, n_samples = 2000, epochs = 1000, plot=True)
    # print(true_loss_log)
    loss_log, grad_log, ws = train(False, 100, n_samples = 2000, STD=0.15, epochs = 100, plot=True, T_sim = -1, log_timing = False)
    print(loss_log)
    
    for index in range(len(grad_log)):  
        grad_log = [m.flatten() for m in grad_log[index]]
        true_grad_log = [m.flatten() for m in true_grad_log[index]]
               
        ws = [m.flatten() for m in ws[index]]
        true_ws = [m.flatten() for m in true_ws[index]]
        
        for v1, v2, name in zip([loss_log, grad_log, ws], [true_loss_log, true_grad_log, true_ws], ['Loss', 'Gradients', 'w']): 
            plt.figure(dpi=500)
            plt.plot(v1)
            plt.plot(v2, '--')
            plt.legend(['Regression', 'Autodiff'])
            plt.title('Layer ' + str(index) + ' ' + name)
            plt.xlabel('Batch Index')
            # plt.ylabel('Loss (MSE)')
            plt.show()
    
CFG.dt = 0.1
CFG.sim_t = 500
# CFG.neuron_model = 'LIF'
# CFG.lif_beta = 0.3
train_and_compare()
exit()
# evaluate('../data/model_2000_RUN1_LR0.001_True_1_0.0_191.pt')
# exit()
S_vals, errs = list(range(1000, 1001, 10)), []
S_vals = [10**(-x) for x in np.linspace(1, 1, 1)]
true_loss, true_grad_log = train(True)
for S in S_vals:
    print("S VALUE: ", S)
    loss_log, grad_log = train(False, S=300, STD=0.15)
    
    true_grad = true_grad_log[1][0]
    grad = grad_log[1][0]
    
    diff = true_grad[:, :] - grad[:, :]
    errs.append(np.linalg.norm(diff))
    
print("VALS", true_grad[0,0], grad[0,0])
print(errs)
    
plt.figure(dpi=500)
plt.plot(S_vals, errs)
plt.xlabel('Std')
plt.ylabel('L2 Error')
plt.title('First 10 rows')
plt.show()

plt.figure(dpi=500)
plt.plot(loss_log)
plt.plot(true_loss)
plt.fill_between(range(len(loss_log)), loss_log, true_loss, alpha=0.5, color='grey')
plt.legend(['Regression', 'Autodiff'], prop={'size': 6})
plt.show()

vmin = min(np.min(true_grad), np.min(grad))
vmax = max(np.max(true_grad), np.max(grad))
plt.figure(dpi=500)
plt.subplot(2,2,1)
plt.imshow(true_grad[:, :], aspect='auto', cmap='hot', vmin=vmin, vmax=vmax)
plt.title('Autodiff Gradient')
plt.xticks([])
plt.yticks([])
plt.colorbar()
plt.subplot(2,2,2)
plt.imshow(grad[:, :], aspect='auto', cmap='hot', vmin=vmin, vmax=vmax)
plt.title('Regression Gradient')
plt.xticks([])
plt.yticks([])
plt.colorbar()

plt.subplot(2,2,(3,4))
mn, true_mean = np.mean(grad[:, :], 0), np.mean(true_grad[:, :], 0)
plt.plot(mn)
plt.plot(true_mean)
plt.legend(['Regression', 'Autodiff'], prop={'size': 6})
plt.title('Mean Along Columns')
plt.fill_between(range(len(mn)), mn, true_mean, alpha=0.5, color='grey')
plt.show()

plt.figure()
plt.plot(grad[:, 5])
plt.plot(true_grad[:, 5])
plt.show()
                             
# if idx == len(dataloader)-1:
#     plt.figure()
#     plt.plot(vals)
#     plt.show()
#     plt.plot(grads)
#     plt.show()
#     plt.figure()
#     plt.subplot(2,2,(3,4))
#     plt.plot(loss_log)
#     plt.subplot(2,2,1)
#     plt.plot(grad_log[0])
#     plt.fill_between(range(len(grad_log[0])), grad_max_min_log[0][1], grad_max_min_log[0][0], alpha=0.3)
#     plt.subplot(2,2,2)
#     plt.plot(grad_std_log[1])
#     # plt.plot(grad_log[1])
#     # plt.fill_between(range(len(grad_log[1])), grad_max_min_log[1][1], grad_max_min_log[1][0], alpha=0.3)
#     plt.suptitle(STD)
#     plt.show()
#     print(loss_log)

    
# CFG.hidden_layers = []
# trainer = load_noisy_to_trainer('../data/model_21_29_33___08_26_2022_NLB_LR0.1_False_100_0.15_dt0.1_16_161.pt')
# #plot_weights(trainer)
# plot_accuracy('../data/accuracy_21_29_33___08_26_2022_NLB_LR0.1_False_100_0.15_dt0.1.txt', 'NLB First Try')
# plot_accuracy('../data/accuracy_18_02_33___08_28_2022_NLB3layers_LR0.1_False_100_0.15_dt0.1.txt', 'NLB Three Layer')
