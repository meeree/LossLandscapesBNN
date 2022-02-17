from config import CFG
from spike_train_mnist import SpikeTrainMNIST
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
import time

torch.set_default_dtype(torch.float32)

class FHNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.W1 = nn.Linear(28*28, 100, bias=False)
        self.W2 = nn.Linear(100, 10, bias=False)
   
    def forward(self, batch : torch.Tensor) -> torch.Tensor:
        B, T = batch.shape[:2] # Batch size and number of timesteps
        dt = CFG.dt
        
        V1 = torch.zeros((B, T, 100)).to('cuda')
        pow1 = torch.zeros((B, 100)).to('cuda')
        S1 = torch.zeros((B, T, 100)).to('cuda')
        z1 = torch.zeros((B, T, 100)).to('cuda')
        V2 = torch.zeros((B, T, 10)).to('cuda')
        pow2 = torch.zeros((B, 10)).to('cuda')
        S2 = torch.zeros((B, T, 10)).to('cuda')
        z2 = torch.zeros((B, T, 10)).to('cuda')

        z1 = self.W1(batch)
        for k in range(1, T):
            pow1 = V1[:, k-1, :].clone() ** 3
            V1[:, k, :] = (1 + dt) * V1[:, k-1, :] - dt/3 * pow1 - dt * S1[:, k-1, :] + dt * z1[:, k-1, :]
            S1[:, k, :] = S1[:, k-1, :] + dt * 0.08 * (V1[:, k-1, :] + 0.7 - 0.8 * S1[:, k-1, :])
            
        z2 = self.W2(V1 / 2)
        for k in range(1, T):
            pow2 = V2[:, k-1, :].clone() ** 3
            V2[:, k, :] = (1 + dt) * V2[:, k-1, :] - dt/3 * pow2 - dt * S2[:, k-1, :] + dt * z2[:, k-1, :]
            S2[:, k, :] = S2[:, k-1, :] + dt * 0.08 * (V2[:, k-1, :] + 0.7 - 0.8 * S2[:, k-1, :])
            
        return V2 
  
class HHNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.W1 = nn.Linear(28*28, 100, bias=False)
        self.W2 = nn.Linear(100, 10, bias=False)  
        
    def forward(self, batch : torch.Tensor) -> torch.Tensor:
        B, T = batch.shape[:2] # Batch size and number of timesteps
        gna = CFG.gna; gk = CFG.gk; gl = CFG.gl;
        Ena = CFG.Ena; Ek = CFG.Ek; El = CFG.El;
        gs = CFG.gs; Vs = CFG.Vs; Iapp = CFG.Iapp;
        Vt = CFG.Vt; Kp = CFG.Kp; 
        a_d = CFG.a_d; a_r = CFG.a_r; dt = CFG.dt;
        
        V1 = torch.full((B, T, 100), -70.0).to('cuda')
        m1 = torch.zeros((B, T, 100)).to('cuda')
        n1 = torch.zeros((B, T, 100)).to('cuda')
        h1 = torch.ones((B, T, 100)).to('cuda')
        y1 = torch.zeros((B, T, 100)).to('cuda')
        z1 = torch.zeros((B, T, 100)).to('cuda')
        pow11 = torch.zeros((B, 100)).to('cuda')
        pow12 = torch.zeros((B, 100)).to('cuda')
  
        V2 = torch.full((B, T, 10), -70.0).to('cuda')
        m2 = torch.zeros((B, T, 10)).to('cuda')
        n2 = torch.zeros((B, T, 10)).to('cuda')
        h2 = torch.ones((B, T, 10)).to('cuda')
        y2 = torch.ones((B, T, 10)).to('cuda')
        z2 = torch.zeros((B, T, 10)).to('cuda')
        pow21 = torch.zeros((B, 10)).to('cuda')
        pow22 = torch.zeros((B, 10)).to('cuda')
        
        z1 = self.W1(batch)
        for k in range(1, T):
            pow11 = gna * (m1[:, k-1, :].clone() ** 3) * h1[:, k-1, :].clone()
            pow12 = gk * n1[:, k-1, :].clone() ** 4
            
            G_scaled = (dt / 2) * (pow11 + pow12 + gl + gs * y1[:, k-1, :].clone())
            E = pow11 * Ena + pow12 * Ek + gl * El + gs * Vs * y1[:, k-1, :].clone()
            
            V1[:, k, :] = (V1[:, k-1, :].clone() * (1 - G_scaled) + dt * (E + Iapp)) / (1 + G_scaled)
            
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

            m1[:, k, :] = (aM * dt + (1 - dt / 2 * (aM + bM)) * m1[:, k-1, :].clone()) / (dt / 2 * (aM + bM) + 1)
            n1[:, k, :] = (aN * dt + (1 - dt / 2 * (aN + bN)) * n1[:, k-1, :].clone()) / (dt / 2 * (aN + bN) + 1)
            h1[:, k, :] = (aH * dt + (1 - dt / 2 * (aH + bH)) * h1[:, k-1, :].clone()) / (dt / 2 * (aH + bH) + 1)    
            y1[:, k, :] = (a_d * z1[:, k-1, :] * dt + (1 - dt / 2 * (a_d * z1[:, k-1, :] + a_r)) * y1[:, k-1, :].clone()) / (dt / 2 * (a_d * z1[:, k-1, :] + a_r) + 1)


        T1 = torch.sigmoid((V1 - Vt) / Kp)
        z2 = self.W2(T1)
        for k in range(1, T):
            pow21 = gna * (m2[:, k-1, :].clone() ** 3) * h2[:, k-1, :].clone()
            pow22 = gk * n2[:, k-1, :].clone() ** 4
            
            G_scaled = (dt / 2) * (pow21 + pow22 + gl + gs * y2[:, k-1, :].clone())
            E = pow21 * Ena + pow22 * Ek + gl * El + gs * Vs * y2[:, k-1, :].clone()
            
            V2[:, k, :] = (V2[:, k-1, :].clone() * (1 - G_scaled) + dt * (E + Iapp)) / (1 + G_scaled)
     
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
                
            m2[:, k, :] = (aM * dt + (1 - dt / 2 * (aM + bM)) * m2[:, k-1, :].clone()) / (dt / 2 * (aM + bM) + 1)
            n2[:, k, :] = (aN * dt + (1 - dt / 2 * (aN + bN)) * n2[:, k-1, :].clone()) / (dt / 2 * (aN + bN) + 1)
            h2[:, k, :] = (aH * dt + (1 - dt / 2 * (aH + bH)) * h2[:, k-1, :].clone()) / (dt / 2 * (aH + bH) + 1)            
            y2[:, k, :] = (a_d * z2[:, k-1, :] * dt + (1 - dt / 2 * (a_d * z2[:, k-1, :] + a_r)) * y2[:, k-1, :].clone()) / (dt / 2 * (a_d * z2[:, k-1, :] + a_r) + 1)
         
        T2 = torch.sigmoid((V2 - Vt) / Kp)
        return T2

class HHNet_No_Gating(nn.Module):
    def __init__(self):
        super().__init__()
        self.W1 = nn.Linear(28*28, 100, bias=False)
        self.W2 = nn.Linear(100, 10, bias=False)  
        
    def forward(self, batch : torch.Tensor) -> torch.Tensor:
        B, T = batch.shape[:2] # Batch size and number of timesteps
        gna = CFG.gna; gk = CFG.gk; gl = CFG.gl;
        Ena = CFG.Ena; Ek = CFG.Ek; El = CFG.El;
        Iapp = CFG.Iapp; Vt = CFG.Vt; Kp = CFG.Kp; 
        dt = CFG.dt;
        
        V1 = torch.full((B, T, 100), -70.0).to('cuda')
        m1 = torch.zeros((B, T, 100)).to('cuda')
        n1 = torch.zeros((B, T, 100)).to('cuda')
        h1 = torch.ones((B, T, 100)).to('cuda')
        z1 = torch.zeros((B, T, 100)).to('cuda')
        pow11 = torch.zeros((B, 100)).to('cuda')
        pow12 = torch.zeros((B, 100)).to('cuda')
  
        V2 = torch.full((B, T, 10), -70.0).to('cuda')
        m2 = torch.zeros((B, T, 10)).to('cuda')
        n2 = torch.zeros((B, T, 10)).to('cuda')
        h2 = torch.ones((B, T, 10)).to('cuda')
        z2 = torch.zeros((B, T, 10)).to('cuda')
        pow21 = torch.zeros((B, 10)).to('cuda')
        pow22 = torch.zeros((B, 10)).to('cuda')
            
        z1 = self.W1(batch)
        for k in range(1, T):
            pow11 = gna * (m1[:, k-1, :].clone() ** 3) * h1[:, k-1, :].clone()
            pow12 = gk * n1[:, k-1, :].clone() ** 4
            
            G_scaled = (dt / 2) * (pow11 + pow12 + gl)
            E = pow11 * Ena + pow12 * Ek + gl * El
            
            V1[:, k, :] = (V1[:, k-1, :].clone() * (1 - G_scaled) + dt * (E + Iapp + z1[:, k-1, :])) / (1 + G_scaled)
            
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

            m1[:, k, :] = (aM * dt + (1 - dt / 2 * (aM + bM)) * m1[:, k-1, :].clone()) / (dt / 2 * (aM + bM) + 1)
            n1[:, k, :] = (aN * dt + (1 - dt / 2 * (aN + bN)) * n1[:, k-1, :].clone()) / (dt / 2 * (aN + bN) + 1)
            h1[:, k, :] = (aH * dt + (1 - dt / 2 * (aH + bH)) * h1[:, k-1, :].clone()) / (dt / 2 * (aH + bH) + 1)    

        # plt.plot(V1[0, :, :].detach().cpu().numpy())
        # plt.show()

        T1 = torch.sigmoid((V1 - Vt) / Kp)
        z2 = self.W2(T1)
        for k in range(1, T):
            pow21 = gna * (m2[:, k-1, :].clone() ** 3) * h2[:, k-1, :].clone()
            pow22 = gk * n2[:, k-1, :].clone() ** 4
            
            G_scaled = (dt / 2) * (pow21 + pow22 + gl)
            E = pow21 * Ena + pow22 * Ek + gl * El
            
            V2[:, k, :] = (V2[:, k-1, :].clone() * (1 - G_scaled) + dt * (E + Iapp + z2[:, k-1, :])) / (1 + G_scaled)
     
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
                
            m2[:, k, :] = (aM * dt + (1 - dt / 2 * (aM + bM)) * m2[:, k-1, :].clone()) / (dt / 2 * (aM + bM) + 1)
            n2[:, k, :] = (aN * dt + (1 - dt / 2 * (aN + bN)) * n2[:, k-1, :].clone()) / (dt / 2 * (aN + bN) + 1)
            h2[:, k, :] = (aH * dt + (1 - dt / 2 * (aH + bH)) * h2[:, k-1, :].clone()) / (dt / 2 * (aH + bH) + 1)            
        
        # plt.plot(V2[-1, :, :].detach().cpu().numpy())
        # plt.show()
         
        T2 = torch.sigmoid((V2 - Vt) / Kp)
        return T2

model = HHNet_No_Gating()
#model.load_state_dict(torch.load('HH_2000_model_200_I0_1.5_0.100000.pt'))

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
        plt.imshow(W1[i, j, :, :], cmap='seismic', interpolation='bilinear')
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
    '../data/mnist_torch/',
    train=True, download=True, transform=transform,
)
test_mnist = datasets.MNIST(
    '../data/mnist_torch/',
    train=False, download=True, transform=transform,
)

train_dataset = SpikeTrainMNIST(train_mnist, 'train')
val_dataset = SpikeTrainMNIST(test_mnist, 'validation')

model = HHNet_No_Gating()
model = model.to('cuda')
optimizer = torch.optim.Adam(model.parameters(), lr = CFG.lr)
loss_fun = nn.MSELoss().to('cuda')
for epch in range(10):
#for lr in [0.0001, 0.001, 0.01, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 1.0, 2.0, 10.0]:
    accuracy_out = open(f'../data/EPOCH_accuracy_2000_{CFG.lr}_EPOCH_{epch}.txt', 'w')

    loss_record = []
    start_time = time.time()
       
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=CFG.train_batch_sz, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=CFG.test_batch_sz, shuffle=False)
    batch_idx = 0
   
    for batch, expected in train_loader:
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
        
        if batch_idx % 20 == 0:
            print(batch_idx, float(loss.detach()), time.time() - start_time)
            start_time = time.time()
            
            v = V2_out[0, :, :].detach().cpu().numpy()
            plt.plot(v)
            plt.title("%d" % batch_idx)
            plt.show()
        
            if batch_idx % 20 == 0:
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
    torch.save(model.state_dict(), '../data/EPOCH_HH_2000_model_%d_I0_%f_%f_EPOCH_%d.pt' % (CFG.n_samples_train, CFG.Iapp, CFG.lr, epch))

        
    plt.figure(dpi=600)
    plt.plot(loss_record)
    plt.title('Loss - Simple MNIST Example')
    plt.xlabel('Batch index')
    plt.ylabel('Loss')
    plt.show()
    
    n_hit = 0
    n_total = 0
    start = time.time()
    for batch, expected in val_loader:
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
        
    print("%f : %f" % (CFG.lr, n_hit / n_total * 100.0))
    print("%f : %f" % (CFG.lr, n_hit / n_total * 100.0), file=accuracy_out)
    accuracy_out.close()
    