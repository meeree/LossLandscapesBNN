# -*- coding: utf-8 -*-
"""
Created on Fri Jan 14 16:56:30 2022

@author: jhazelde
"""

from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.autograd import Function
from torchviz import make_dot
from torch.distributions.exponential import Exponential
import torchvision
import time
from torch.utils.data import Dataset
from os.path import exists

torch.set_default_dtype(torch.float32)

def to_spiketrain (output, sample, total_timesteps, max_firings, n_timesteps_spike):
    for pix_id, s in enumerate(sample.flatten()):
        if s < 0.01:
            continue # No spikes, pixel is black.
            
        rate = (max_firings * s) / total_timesteps
        exp = Exponential(rate)
        i = 0
        while i < total_timesteps:
            period = exp.sample()
            i += int(period)
            end_pt = min(total_timesteps, i+n_timesteps_spike)
            output[i:end_pt, pix_id] = 1.0
            i = end_pt

class SpikeTrainMNIST(Dataset):
    def __init__(self, mnist_dset, total_timesteps, max_firings, n_timesteps_spike, test):
        self.spiketrains = torch.zeros((2000, total_timesteps, 28*28))
        self.labels = torch.nn.functional.one_hot(mnist_dset.targets, num_classes=10) * 1.0
        fname = f'spiketrains_test_{total_timesteps}.pt' if test else f'spiketrains_train_{total_timesteps}.pt'
        if exists(fname):
            self.spiketrains = torch.load(fname)    
        else:
            for i, img in enumerate(mnist_dset):
                if i % 500 == 0:
                    print("%f%%, %d, %d" % (100 * i / len(mnist_dset), i, len(mnist_dset)))
                    if i == 2000: 
                        break
                to_spiketrain(self.spiketrains[i, :, :], img[0][0, :, :], total_timesteps, max_firings, n_timesteps_spike)
            torch.save(self.spiketrains, fname)

    def __len__(self):
        return self.spiketrains.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.spiketrains[idx, :, :], self.labels[idx, :]
    
class LIFLayer(Function):
    @staticmethod
    def forward(ctx, V_z):
        ctx.save_for_backward(V_z) # save input for backward pass
        return 0.9 * V_z[0,:] + 0.1 * V_z[1,:]

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = None # set output to None

        V_z, = ctx.saved_tensors # restore input from context

        # check that input requires grad
        # if not requires grad we will return None to speed up computation
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.clone()
            grad_input = torch.cat((grad_input, grad_input), 0)
            
            # Derivative is (dV_current/dV_prev, dV/dz).
            grad_input[0,:] *= 0.9
            grad_input[1,:] *= 0.1
        return grad_input
  
SIM_T = 2000
BATCH_SIZE = 10

class LIFNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.W1 = nn.Linear(28*28, 100, bias=False)
        self.W2 = nn.Linear(100, 10, bias=False)     
   
    def forward(self, batch : torch.Tensor) -> torch.Tensor:
        V1 = torch.zeros((BATCH_SIZE, SIM_T, 100)).to('cuda')
        z1 = torch.zeros((BATCH_SIZE, SIM_T, 100)).to('cuda')
        V2 = torch.zeros((BATCH_SIZE, SIM_T, 10)).to('cuda')
        z2 = torch.zeros((BATCH_SIZE, SIM_T, 10)).to('cuda')
        
        z1 = self.W1(batch)
        for k in range(1, SIM_T):
            V1[:, k, :] = 0.7 * V1[:, k-1, :] + 0.3 * z1[:, k-1, :]
      #      V1[:, k, :] = torch.where(V1[:, k, :] > 0.08, torch.zeros(1).cuda(), V1[:, k, :])
            
        z2 = self.W2(V1[:, :, :])
        for k in range(1, SIM_T):
            V2[:, k, :] = 0.7 * V2[:, k-1, :] + 0.3 * z2[:, k-1, :]
       #     V2[:, k, :] = torch.where(V2[:, k, :] > 0.08, torch.zeros(1).cuda(), V2[:, k, :])

        return V2 
    
DT = 0.01

class FHNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.W1 = nn.Linear(28*28, 100, bias=False)
        self.W2 = nn.Linear(100, 10, bias=False)
   
    def forward(self, batch : torch.Tensor) -> torch.Tensor:
        V1 = torch.zeros((BATCH_SIZE, SIM_T, 100)).to('cuda')
        pow1 = torch.zeros((BATCH_SIZE, 100)).to('cuda')
        S1 = torch.zeros((BATCH_SIZE, SIM_T, 100)).to('cuda')
        z1 = torch.zeros((BATCH_SIZE, SIM_T, 100)).to('cuda')
        V2 = torch.zeros((BATCH_SIZE, SIM_T, 10)).to('cuda')
        pow2 = torch.zeros((BATCH_SIZE, 10)).to('cuda')
        S2 = torch.zeros((BATCH_SIZE, SIM_T, 10)).to('cuda')
        z2 = torch.zeros((BATCH_SIZE, SIM_T, 10)).to('cuda')

        z1 = self.W1(batch)
        for k in range(1, SIM_T):
            pow1 = V1[:, k-1, :].clone() ** 3
            V1[:, k, :] = (1 + DT) * V1[:, k-1, :] - DT/3 * pow1 - DT * S1[:, k-1, :] + DT * z1[:, k-1, :]
            S1[:, k, :] = S1[:, k-1, :] + DT * 0.08 * (V1[:, k-1, :] + 0.7 - 0.8 * S1[:, k-1, :])
            
        z2 = self.W2(V1 / 2)
        for k in range(1, SIM_T):
            pow2 = V2[:, k-1, :].clone() ** 3
            V2[:, k, :] = (1 + DT) * V2[:, k-1, :] - DT/3 * pow2 - DT * S2[:, k-1, :] + DT * z2[:, k-1, :]
            S2[:, k, :] = S2[:, k-1, :] + DT * 0.08 * (V2[:, k-1, :] + 0.7 - 0.8 * S2[:, k-1, :])
            
        return V2 
  
#======================NEURON=PARAMETERS=================================================
gna = 40.0;
gk = 35.0;
gl = 0.3;

Ena = 55.0;
Ek = -77.0;
El = -65.0;

gs = 0.04;
Vs = 0.0;
Iapp = 1.5;

Vt = 20;
Kp = 3;
a_d = 1;
a_r = 0.1;
#=======================================================================================
class HHNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.W1 = nn.Linear(28*28, 100, bias=False)
        self.W2 = nn.Linear(100, 10, bias=False)  
        
    def forward(self, batch : torch.Tensor) -> torch.Tensor:
        V1 = torch.full((BATCH_SIZE, SIM_T, 100), -70.0).to('cuda')
        m1 = torch.zeros((BATCH_SIZE, SIM_T, 100)).to('cuda')
        n1 = torch.zeros((BATCH_SIZE, SIM_T, 100)).to('cuda')
        h1 = torch.ones((BATCH_SIZE, SIM_T, 100)).to('cuda')
        y1 = torch.zeros((BATCH_SIZE, SIM_T, 100)).to('cuda')
        z1 = torch.zeros((BATCH_SIZE, SIM_T, 100)).to('cuda')
        G1 = torch.zeros((BATCH_SIZE, 100, 2)).to('cuda')
        pow11 = torch.zeros((BATCH_SIZE, 100)).to('cuda')
        pow12 = torch.zeros((BATCH_SIZE, 100)).to('cuda')
  
        V2 = torch.full((BATCH_SIZE, SIM_T, 10), -70.0).to('cuda')
        m2 = torch.zeros((BATCH_SIZE, SIM_T, 10)).to('cuda')
        n2 = torch.zeros((BATCH_SIZE, SIM_T, 10)).to('cuda')
        h2 = torch.ones((BATCH_SIZE, SIM_T, 10)).to('cuda')
        y2 = torch.ones((BATCH_SIZE, SIM_T, 10)).to('cuda')
        z2 = torch.zeros((BATCH_SIZE, SIM_T, 10)).to('cuda')
        G2 = torch.zeros((BATCH_SIZE, 10, 2)).to('cuda')
        pow21 = torch.zeros((BATCH_SIZE, 10)).to('cuda')
        pow22 = torch.zeros((BATCH_SIZE, 10)).to('cuda')
            
        z1 = self.W1(batch)
        for k in range(1, SIM_T):
            pow11 = gna * (m1[:, k-1, :].clone() ** 3) * h1[:, k-1, :].clone()
            pow12 = gk * n1[:, k-1, :].clone() ** 4
            
            G_scaled = (DT / 2) * (pow11 + pow12 + gl + gs * y1[:, k-1, :].clone())
            E = pow11 * Ena + pow12 * Ek + gl * El + gs * Vs * y1[:, k-1, :].clone()
            
            V1[:, k, :] = (V1[:, k-1, :].clone() * (1 - G_scaled) + DT * (E + Iapp)) / (1 + G_scaled)
            
            aN = 0.02 * (V1[:, k, :] - 25) / (1 - torch.exp((-V1[:, k, :] + 25) / 9.0))
            aM = 0.182 * (V1[:, k, :] + 35) / (1 - torch.exp((-V1[:, k, :] - 35) / 9.0))
            aH = 0.25 * torch.exp((-V1[:, k, :] - 90) / 12.0)
                
            bN = -0.002 * (V1[:, k, :] - 25) / (1 - torch.exp((V1[:, k, :] - 25) / 9.0))
            bM = -0.124 * (V1[:, k, :] + 35) / (1 - torch.exp((V1[:, k, :] + 35) / 9.0))
            bH = 0.25 * torch.exp((V1[:, k, :] + 34) / 12.0)
            
            if torch.any(V1[:, k, :] == 25) or torch.any(V1[:, k, :] == -35):
                aN[torch.where(V1[:, k, :] == 25)] = 0.18
                bN[torch.where(V1[:, k, :] == 25)] = 0.018
                aM[torch.where(V1[:, k, :] == -35)] = 1.638
                bM[torch.where(V1[:, k, :] == -35)] = 1.116

            m1[:, k, :] = (aM * DT + (1 - DT / 2 * (aM + bM)) * m1[:, k-1, :].clone()) / (DT / 2 * (aM + bM) + 1)
            n1[:, k, :] = (aN * DT + (1 - DT / 2 * (aN + bN)) * n1[:, k-1, :].clone()) / (DT / 2 * (aN + bN) + 1)
            h1[:, k, :] = (aH * DT + (1 - DT / 2 * (aH + bH)) * h1[:, k-1, :].clone()) / (DT / 2 * (aH + bH) + 1)    
            y1[:, k, :] = (a_d * z1[:, k-1, :] * DT + (1 - DT / 2 * (a_d * z1[:, k-1, :] + a_r)) * y1[:, k-1, :].clone()) / (DT / 2 * (a_d * z1[:, k-1, :] + a_r) + 1)


        T1 = torch.sigmoid((V1 - Vt) / Kp)
        z2 = self.W2(T1)
        for k in range(1, SIM_T):
            pow21 = gna * (m2[:, k-1, :].clone() ** 3) * h2[:, k-1, :].clone()
            pow22 = gk * n2[:, k-1, :].clone() ** 4
            
            G_scaled = (DT / 2) * (pow21 + pow22 + gl + gs * y2[:, k-1, :].clone())
            E = pow21 * Ena + pow22 * Ek + gl * El + gs * Vs * y2[:, k-1, :].clone()
            
            V2[:, k, :] = (V2[:, k-1, :].clone() * (1 - G_scaled) + DT * (E + Iapp)) / (1 + G_scaled)
     
            aN = 0.02 * (V2[:, k, :] - 25) / (1 - torch.exp((-V2[:, k, :] + 25) / 9.0))
            aM = 0.182 * (V2[:, k, :] + 35) / (1 - torch.exp((-V2[:, k, :] - 35) / 9.0))
            aH = 0.25 * torch.exp((-V2[:, k, :] - 90) / 12.0)

            bN = -0.002 * (V2[:, k, :] - 25) / (1 - torch.exp((V2[:, k, :] - 25) / 9.0))
            bM = -0.124 * (V2[:, k, :] + 35) / (1 - torch.exp((V2[:, k, :] + 35) / 9.0))
            bH = 0.25 * torch.exp((V2[:, k, :] + 34) / 12.0)

            if torch.any(V2[:, k, :] == 25) or torch.any(V2[:, k, :] == -35):
                aN[torch.where(V2[:, k, :] == 25)] = 0.18
                bN[torch.where(V2[:, k, :] == 25)] = 0.018
                aM[torch.where(V2[:, k, :] == -35)] = 1.638
                bM[torch.where(V2[:, k, :] == -35)] = 1.116
                
            m2[:, k, :] = (aM * DT + (1 - DT / 2 * (aM + bM)) * m2[:, k-1, :].clone()) / (DT / 2 * (aM + bM) + 1)
            n2[:, k, :] = (aN * DT + (1 - DT / 2 * (aN + bN)) * n2[:, k-1, :].clone()) / (DT / 2 * (aN + bN) + 1)
            h2[:, k, :] = (aH * DT + (1 - DT / 2 * (aH + bH)) * h2[:, k-1, :].clone()) / (DT / 2 * (aH + bH) + 1)            
            y2[:, k, :] = (a_d * z2[:, k-1, :] * DT + (1 - DT / 2 * (a_d * z2[:, k-1, :] + a_r)) * y2[:, k-1, :].clone()) / (DT / 2 * (a_d * z2[:, k-1, :] + a_r) + 1)
         
        T2 = torch.sigmoid((V2 - Vt) / Kp)
        return T2

class HHNet_No_Gating(nn.Module):
    def __init__(self):
        super().__init__()
        self.W1 = nn.Linear(28*28, 100, bias=False)
        self.W2 = nn.Linear(100, 10, bias=False)  
        
    def forward(self, batch : torch.Tensor) -> torch.Tensor:
        V1 = torch.full((BATCH_SIZE, SIM_T, 100), -70.0).to('cuda')
        m1 = torch.zeros((BATCH_SIZE, SIM_T, 100)).to('cuda')
        n1 = torch.zeros((BATCH_SIZE, SIM_T, 100)).to('cuda')
        h1 = torch.ones((BATCH_SIZE, SIM_T, 100)).to('cuda')
        z1 = torch.zeros((BATCH_SIZE, SIM_T, 100)).to('cuda')
        G1 = torch.zeros((BATCH_SIZE, 100, 2)).to('cuda')
        pow11 = torch.zeros((BATCH_SIZE, 100)).to('cuda')
        pow12 = torch.zeros((BATCH_SIZE, 100)).to('cuda')
  
        V2 = torch.full((BATCH_SIZE, SIM_T, 10), -70.0).to('cuda')
        m2 = torch.zeros((BATCH_SIZE, SIM_T, 10)).to('cuda')
        n2 = torch.zeros((BATCH_SIZE, SIM_T, 10)).to('cuda')
        h2 = torch.ones((BATCH_SIZE, SIM_T, 10)).to('cuda')
        z2 = torch.zeros((BATCH_SIZE, SIM_T, 10)).to('cuda')
        G2 = torch.zeros((BATCH_SIZE, 10, 2)).to('cuda')
        pow21 = torch.zeros((BATCH_SIZE, 10)).to('cuda')
        pow22 = torch.zeros((BATCH_SIZE, 10)).to('cuda')
            
        z1 = self.W1(batch)
        for k in range(1, SIM_T):
            pow11 = gna * (m1[:, k-1, :].clone() ** 3) * h1[:, k-1, :].clone()
            pow12 = gk * n1[:, k-1, :].clone() ** 4
            
            G_scaled = (DT / 2) * (pow11 + pow12 + gl)
            E = pow11 * Ena + pow12 * Ek + gl * El
            
            V1[:, k, :] = (V1[:, k-1, :].clone() * (1 - G_scaled) + DT * (E + Iapp + z1[:, k-1, :])) / (1 + G_scaled)
            
            aN = 0.02 * (V1[:, k, :] - 25) / (1 - torch.exp((-V1[:, k, :] + 25) / 9.0))
            aM = 0.182 * (V1[:, k, :] + 35) / (1 - torch.exp((-V1[:, k, :] - 35) / 9.0))
            aH = 0.25 * torch.exp((-V1[:, k, :] - 90) / 12.0)
                
            bN = -0.002 * (V1[:, k, :] - 25) / (1 - torch.exp((V1[:, k, :] - 25) / 9.0))
            bM = -0.124 * (V1[:, k, :] + 35) / (1 - torch.exp((V1[:, k, :] + 35) / 9.0))
            bH = 0.25 * torch.exp((V1[:, k, :] + 34) / 12.0)
            
            if torch.any(V1[:, k, :] == 25) or torch.any(V1[:, k, :] == -35):
                aN[torch.where(V1[:, k, :] == 25)] = 0.18
                bN[torch.where(V1[:, k, :] == 25)] = 0.018
                aM[torch.where(V1[:, k, :] == -35)] = 1.638
                bM[torch.where(V1[:, k, :] == -35)] = 1.116

            m1[:, k, :] = (aM * DT + (1 - DT / 2 * (aM + bM)) * m1[:, k-1, :].clone()) / (DT / 2 * (aM + bM) + 1)
            n1[:, k, :] = (aN * DT + (1 - DT / 2 * (aN + bN)) * n1[:, k-1, :].clone()) / (DT / 2 * (aN + bN) + 1)
            h1[:, k, :] = (aH * DT + (1 - DT / 2 * (aH + bH)) * h1[:, k-1, :].clone()) / (DT / 2 * (aH + bH) + 1)    

        # plt.plot(V1[0, :, :].detach().cpu().numpy())
        # plt.show()

        T1 = torch.sigmoid((V1 - Vt) / Kp)
        z2 = self.W2(T1)
        for k in range(1, SIM_T):
            pow21 = gna * (m2[:, k-1, :].clone() ** 3) * h2[:, k-1, :].clone()
            pow22 = gk * n2[:, k-1, :].clone() ** 4
            
            G_scaled = (DT / 2) * (pow21 + pow22 + gl)
            E = pow21 * Ena + pow22 * Ek + gl * El
            
            V2[:, k, :] = (V2[:, k-1, :].clone() * (1 - G_scaled) + DT * (E + Iapp + z2[:, k-1, :])) / (1 + G_scaled)
     
            aN = 0.02 * (V2[:, k, :] - 25) / (1 - torch.exp((-V2[:, k, :] + 25) / 9.0))
            aM = 0.182 * (V2[:, k, :] + 35) / (1 - torch.exp((-V2[:, k, :] - 35) / 9.0))
            aH = 0.25 * torch.exp((-V2[:, k, :] - 90) / 12.0)

            bN = -0.002 * (V2[:, k, :] - 25) / (1 - torch.exp((V2[:, k, :] - 25) / 9.0))
            bM = -0.124 * (V2[:, k, :] + 35) / (1 - torch.exp((V2[:, k, :] + 35) / 9.0))
            bH = 0.25 * torch.exp((V2[:, k, :] + 34) / 12.0)

            if torch.any(V2[:, k, :] == 25) or torch.any(V2[:, k, :] == -35):
                aN[torch.where(V2[:, k, :] == 25)] = 0.18
                bN[torch.where(V2[:, k, :] == 25)] = 0.018
                aM[torch.where(V2[:, k, :] == -35)] = 1.638
                bM[torch.where(V2[:, k, :] == -35)] = 1.116
                
            m2[:, k, :] = (aM * DT + (1 - DT / 2 * (aM + bM)) * m2[:, k-1, :].clone()) / (DT / 2 * (aM + bM) + 1)
            n2[:, k, :] = (aN * DT + (1 - DT / 2 * (aN + bN)) * n2[:, k-1, :].clone()) / (DT / 2 * (aN + bN) + 1)
            h2[:, k, :] = (aH * DT + (1 - DT / 2 * (aH + bH)) * h2[:, k-1, :].clone()) / (DT / 2 * (aH + bH) + 1)            
        
        plt.plot(V2[-1, :, :].detach().cpu().numpy())
        plt.show()
         
        T2 = torch.sigmoid((V2 - Vt) / Kp)
        return T2

model = HHNet_No_Gating()
model.load_state_dict(torch.load('HH_2000_model_200_I0_1.5_0.100000.pt'))

plt.imshow(model.W2.weight.data.numpy(), cmap='seismic', aspect='auto')
plt.show()

plt.figure(figsize=(30,30))
W1 = model.W1.weight.data.numpy()
vmin, vmax = W1.min(), W1.max()
print(vmin, vmax)
W1 = W1.reshape(10, 10, 28, 28)

for i in range(10):
    for j in range(10):     
        plt.subplot(10, 10, i + j * 10 + 1)
        plt.imshow(W1[i, j, :, :], cmap='seismic', vmin = -1.5, vmax = 1.5, interpolation='none')
        plt.box(False)
        plt.xticks([])
        plt.yticks([])

plt.show()

plt.imshow(model.W2.weight.data.numpy(), aspect = 'auto', cmap='seismic', interpolation='none')
plt.colorbar()
plt.show()

transform=transforms.Compose([
    transforms.ToTensor()
])
train_mnist = datasets.MNIST(
    '../../data/mnist_torch/',
    train=True, download=True, transform=transform,
)
test_mnist = datasets.MNIST(
    '../../data/mnist_torch/',
    train=False, download=True, transform=transform,
)

train_dataset = SpikeTrainMNIST(train_mnist, SIM_T, 10, 100, test=False)
test_dataset = SpikeTrainMNIST(test_mnist, SIM_T, 10, 100, test=True)

# Print first sample of first sample.

# Baseline DNN
# model = nn.Sequential(
#     nn.Linear(28*28, 100, bias=False),
#     nn.Sigmoid(), 
#     nn.Linear(100, 10, bias=False),
#     nn.Sigmoid(),
# )

# print("Model Parameters:")
# for name,   param in model.named_parameters():
#     if param.requires_grad:
#         print (name)

    
model = HHNet_No_Gating()
model = model.to('cuda')
lr=0.01
optimizer = torch.optim.Adam(model.parameters(), lr = lr)
loss_fun = nn.MSELoss().to('cuda')

BATCH_SIZE = 100
n_batches = 20
        
for epch in range(8):
    accuracy_out = open(f'accuracy_2000_0.01_EPOCH_{epch}.txt', 'w')

    loss_record = []
    start_time = time.time()
    
    BATCH_SIZE = 100
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=500, shuffle=False)
    batch_idx = 0
   
    for batch, expected in train_loader:
        if batch_idx == n_batches:
            break
        
        optimizer.zero_grad()   
        V2_out = model(batch.to('cuda'))
        out_avg = torch.mean(V2_out, dim=1)
        loss = loss_fun(out_avg, expected.to('cuda'))
        loss_record.append(loss.detach())
                   
        loss.backward()
        
        # Set NaN values to zero! This is a hack way to fix this issue, but I think
        # NaNs only occur very rarely due to 0/0 in gating vars, so it is not a big issue.
        model.W1.weight.grad[torch.isnan(model.W1.weight.grad)] = 0.0
        model.W2.weight.grad[torch.isnan(model.W2.weight.grad)] = 0.0
        optimizer.step()
        
        # Save weights every 50 batches.
        if (batch_idx + 1) % 200 == 0 or batch_idx == n_batches - 1:
            torch.save(model.state_dict(), 'TEST_HH_2000_model_%d_I0_%f_%f_EPOCH_%d.pt' % (n_batches, Iapp, lr, epch))
        
        if batch_idx % 5 == 0:
            print(batch_idx, float(loss.detach()), time.time() - start_time)
            start_time = time.time()
            
            v = V2_out[0, :, :].detach().cpu().numpy()
            plt.plot(v)
            plt.title("%d" % batch_idx)
            plt.show()
        
            if batch_idx % 5 == 0:
                plt.imshow(model.W1.weight.grad.cpu().numpy(), aspect='auto', cmap='seismic')
                plt.colorbar()
                plt.show()
                
                plt.imshow(model.W2.weight.grad.cpu().numpy(), aspect='auto', cmap='seismic')
                plt.colorbar()
                plt.show()
                
                plt.plot(loss_record)
                plt.title('Loss %d' % batch_idx)
                plt.xlabel('Batch index')
                plt.ylabel('Loss')
                plt.show()  
        
        batch_idx += 1
        
    plt.figure(dpi=600)
    plt.plot(loss_record)
    plt.title('Loss - Simple MNIST Example')
    plt.xlabel('Batch index')
    plt.ylabel('Loss')
    plt.show()
    
    n_hit = 0
    n_total = 0
    start = time.time()
    BATCH_SIZE = 500
    for batch, expected in test_loader:
        if n_total == 500:
            break
        
        if (n_total + 1) % 51 == 0:
            print(time.time() - start, n_total, n_hit / n_total * 100.0)
            start = time.time()
   
        with torch.no_grad():
            out_avg = torch.mean(model(batch.to('cuda')), dim=1)
            guess = torch.argmax(out_avg, dim=1).cpu()
            labels = torch.argmax(expected, dim=1)
            n_hit += torch.sum(guess == labels)
        print(time.time() - start, batch.shape)
        start = time.time()
        n_total += batch.shape[0]
        
    print("%f : %f" % (lr, n_hit / n_total * 100.0))
    print("%f : %f" % (lr, n_hit / n_total * 100.0), file=accuracy_out)
    accuracy_out.close()
    