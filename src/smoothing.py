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
from bnn import BNN, Noisy_BNN
from time import time
import scipy.stats
import os
from sklearn.linear_model import Ridge
print(f'Using GPU: {torch.cuda.get_device_name(0)}')
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

CFG.n_samples_val = 500
CFG.test_batch_sz = 250
CFG.train_batch_sz = 5
#CFG.plot = False
#CFG.plot_all = False
#CFG.dt = 0.1
CFG.sim_t = 1

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

def print_cuda_mem(timestep):
    total = torch.cuda.get_device_properties(0).total_memory
    reserved = torch.cuda.memory_reserved(0)
    allocated = torch.cuda.memory_allocated(0)
    free = reserved - allocated
    print(f'CUDA Memory Usage {timestep}: Total {total/1e9:.3f}GB, free {free/1e9:.3f}GB, reserved {reserved/1e9:.3f}GB, allocated {allocated/1e9:.3f}GB')

def train(use_autodiff, S = 500, plot = False, n_samples = 1, log_timing = False):
    STD = 0.000001
    if use_autodiff:
        STD = 0.0
        S = 1
    CFG.lr = 0.01
    CFG.n_samples_train = n_samples
    CFG.identifier = str(n_samples) + '_ NON_SMOOTHED'
    
  #  torch.manual_seed(0)
    trainer = Trainer()
    trainer.model = Noisy_BNN(S).to('cuda')
    trainer.optimizer = torch.optim.Adam(trainer.model.Ws, lr = CFG.lr)
    dataloader = torch.utils.data.DataLoader(trainer.train_dataset, batch_size=1, shuffle=True)
    loss_log, grad_log = [], [[], []]
    for idx, (batch, expected) in enumerate(dataloader):
        print_cuda_mem(f'{idx}, 1')
        with torch.set_grad_enabled(use_autodiff):
            t0 = time()
            trainer.optimizer.zero_grad()
            loss_fun = torch.nn.MSELoss(reduction='none').to('cuda')
            out = trainer.model(batch.to('cuda'), STD)
            print_cuda_mem(f'{idx}, 2')
            mean_out = torch.mean(out, dim=2)
            target = expected.to('cuda').unsqueeze(0).repeat((out.shape[0], 1, 1))
            unreduced_loss = mean_out # Turn neurons off benchmark
            #unreduced_loss = loss_fun(mean_out, target) # MNIST benchmark
            losses = torch.mean(unreduced_loss, dim=(1,2))
            if log_timing:
                print(f'Evaluation time : {time() - t0}')
        
        # Approximate gradient descent using regression.
        if not use_autodiff:
            t0 = time()
            losses = losses.cpu().detach().numpy()
            l0 = losses[0] # Loss at origin. Note that Noisy_BNN always samples first point at origin!
            loss_log.append(np.mean(losses))
            for i in range(len(trainer.model.noises)):
                noise_offset, Wsmpls = trainer.model.noises[i], trainer.model.Ws[i]
                S = noise_offset.shape[0]
                flat_noise = noise_offset.reshape((S, -1))
                # if i == 1:
                #     flat_noise = torch.mean(noise_offset, -1).reshape((S, -1)) 

                flat_noise = flat_noise.cpu().detach().numpy()
                M, _, _, _ = np.linalg.lstsq(flat_noise, losses - l0)
                # if i == 1:
                #     M = M.reshape((-1, 1))
                #     M = np.tile(M, (1, Wsmpls.shape[3])) / Wsmpls.shape[3]
               # clf = Ridge(alpha = 0.0)
              #  clf.fit(flat_noise, losses - l0)
              #  M = clf.coef_
                M = M.reshape((Wsmpls.shape[2], Wsmpls.shape[3]))
                # Assign gradient manually for optimization
                M = torch.from_numpy(M).to('cuda')
                trainer.optimizer.param_groups[0]['params'][i] = None
                trainer.model.Ws[i].grad = \
                    M.reshape((1, 1, Wsmpls.shape[2], Wsmpls.shape[3])).repeat(S, 1, 1, 1)
                trainer.optimizer.param_groups[0]['params'][i] = trainer.model.Ws[i] # Copy weight to optimizer.
            if log_timing:
                print(f'Regression time : {time() - t0}')
            
            if plot and idx % 10 == 0:
                def isolated_scatter(inds):
                    X = trainer.model.noises[1][:,0,inds[0],inds[1]].reshape(-1).detach().cpu().numpy()
                    plt.scatter(X, losses, s=1.3, zorder=1, color='black')
                    plt.scatter(X[0], l0, s=1.5, zorder=2, color='red')
                    X = np.linspace(np.min(X), np.max(X), 100)
                    m = trainer.model.Ws[1].grad[0,0,inds[0],inds[1]].cpu()
                    plt.plot(X, m * X + l0, zorder=0)
                    plt.title(m)
                    plt.xlabel('$w^2_{' + f'{inds[0],inds[1]}' + '}$')
                    plt.ylabel('Loss') 
            
                plt.figure(dpi=500)
                plt.subplot(1,3,1)
                isolated_scatter([0,0])
                plt.subplot(1,3,2)
                isolated_scatter([0,1])            
                plt.subplot(1,3,3)
                isolated_scatter([1,0])
               
                plt.show()
        else:
            t0 = time()
            loss = losses[0]
            loss_log.append(loss.cpu().item())
            loss.backward()
            if log_timing:
                print(f'Autodiff time : {time() - t0}')
        print_cuda_mem(f'{idx}, 3')
   
        # Optimization step.
        t0 = time()
        trainer.optimizer.step()
        if log_timing:
            print(f'Optimization time : {time() - t0}')
        print_cuda_mem(f'{idx}, 4')

        for i in range(len(trainer.model.Ws)):
            grad_log[i].append(trainer.model.Ws[i].grad.cpu().numpy())
    return loss_log, grad_log

loss_log, _ = train(False, 100, n_samples = 100)
true_loss_log, _ = train(True, n_samples = 100)
plt.plot(loss_log)
plt.plot(true_loss_log)
plt.show()

true_loss, true_grad_log = train(True)
S_vals, errs = list(range(1, 75)), []
for S in S_vals:
    print(S)
    loss_log, grad_log = train(False, S)
    
    true_grad = true_grad_log[1][0][0, 0, :, :]
    grad = grad_log[1][0][0, 0, :, :]
    
    diff = true_grad[:10, :] - grad[:10, :]
    errs.append(np.linalg.norm(diff))
    
plt.figure(dpi=500)
plt.plot(S_vals, errs)
plt.xlabel('S')
plt.ylabel('L2 Error')
plt.title('First 10 rows')
plt.show()

plt.figure(dpi=500)
plt.plot(loss_log)
plt.plot(true_loss)
plt.fill_between(range(len(loss_log)), loss_log, true_loss, alpha=0.5, color='grey')
plt.legend(['Regression', 'Autodiff'], prop={'size': 6})
plt.show()

plt.figure(dpi=500)
plt.subplot(2,2,1)
plt.imshow(true_grad[:10, :], aspect='auto', cmap='gist_rainbow')
plt.title('Autodiff Gradient')
plt.xticks([])
plt.yticks([])
plt.colorbar()
plt.subplot(2,2,2)
plt.imshow(grad[:10, :], aspect='auto', cmap='gist_rainbow')
plt.title('Regression Gradient')
plt.xticks([])
plt.yticks([])
plt.colorbar()

plt.subplot(2,2,(3,4))
mn, true_mean = np.mean(grad[:10, :], 1), np.mean(true_grad[:10, :], 1)
plt.plot(mn)
plt.plot(true_mean)
plt.legend(['Regression', 'Autodiff'], prop={'size': 6})
plt.title('Mean Along Rows')
plt.fill_between(range(len(mn)), mn, true_mean, alpha=0.5, color='grey')
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