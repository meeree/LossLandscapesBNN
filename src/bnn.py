# Biological neural network models.
from config import CFG
import matplotlib.pyplot as plt
import torch
from torch import nn

torch.set_default_dtype(torch.float32)

class BNN(nn.Module):
    def __init__(self, input_dim=28*28, output_dim=10):
        super().__init__()
        NeuronModel = eval(CFG.neuron_model)
        
        self.dims = [input_dim] + CFG.hidden_layers + [output_dim]
        print('Network dimensions: ' + str(self.dims))
        self.Ws = []
        self.layers = []
        for d1, d2 in zip(self.dims[:-1], self.dims[1:]):
            self.Ws.append(nn.Linear(d1, d2, bias=False))
            if CFG.use_DNN:
                self.layers.append(torch.nn.Sigmoid())
            else:
                self.layers.append(NeuronModel(d2))

        # Need to add weights as parameters of model.
        for i, W in enumerate(self.Ws):
            self.__setattr__(f'W{i+1}', W)
        
    def forward(self, batch : torch.Tensor) -> torch.Tensor:
        T = batch
        for W, layer in zip(self.Ws, self.layers):
            z = W(T)
            T = layer(z)
            if CFG.plot and CFG.plot_all:
                if CFG.use_DNN:
                    plt.figure(dpi=500)
                    plt.subplot(2,1,1)
                    plt.plot(T[0, :, :].detach().cpu().numpy(), linewidth=1.0)
                    plt.subplot(2,1,2)
                    plt.plot(z[0, :, :].detach().cpu().numpy(), linewidth=1.0)
                    plt.show()
                
        return T

class FH(nn.Module):
    def __init__(self, L):
        super().__init__()
        self.L = L
   
    def forward(self, z):
        B, T = z.shape[:2] # Batch size and number of timesteps
        dt = CFG.dt
        
        V = torch.zeros((B, T, self.L)).to('cuda')
        pow1 = torch.zeros((B, self.L)).to('cuda')
        S = torch.zeros((B, T, self.L)).to('cuda')
        for k in range(1, T):
            pow1 = V[:, k-1, :].clone() ** 3
            V[:, k, :] = (1 + dt) * V[:, k-1, :] - dt/3 * pow1 - dt * S[:, k-1, :] + dt * z[:, k-1, :]
            S[:, k, :] = S[:, k-1, :] + dt * 0.08 * (V[:, k-1, :] + 0.7 - 0.8 * S[:, k-1, :])
        return V / 2
    
class HH_Synaptic(nn.Module):
    def __init__(self, L):
        super().__init__()
        self.L = L
        
    def forward(self, z):
        B, N = z.shape[:2] # Batch size and number of timesteps
        gna = CFG.gna; gk = CFG.gk; gl = CFG.gl;
        Ena = CFG.Ena; Ek = CFG.Ek; El = CFG.El;
        gs = CFG.gs; Vs = CFG.Vs; Iapp = CFG.Iapp;
        Vt = CFG.Vt; Kp = CFG.Kp;
        a_d = CFG.a_d; a_r = CFG.a_r; dt = CFG.dt;
        
        V = torch.full((B, N, self.L), -70.0).to('cuda')
        m = torch.zeros((B, N, self.L)).to('cuda')
        n = torch.zeros((B, N, self.L)).to('cuda')
        h = torch.ones((B, N, self.L)).to('cuda')
        y = torch.zeros((B, N, self.L)).to('cuda')
        pow1 = torch.zeros((B, self.L)).to('cuda')
        pow2 = torch.zeros((B, self.L)).to('cuda')  
        
        for k in range(1, N):
            pow1 = gna * (m[:, k-1, :].clone() ** 3) * h[:, k-1, :].clone()
            pow2 = gk * n[:, k-1, :].clone() ** 4
            
            G_scaled = (dt / 2) * (pow1 + pow2 + gl + gs * y[:, k-1, :].clone())
            E = pow1 * Ena + pow2 * Ek + gl * El + gs * Vs * y[:, k-1, :].clone()
            
            V[:, k, :] = (V[:, k-1, :].clone() * (1 - G_scaled) + dt * (E + Iapp)) / (1 + G_scaled)
            
            aN = 0.02 * (V[:, k, :] - 25) / (1 - torch.exp((-V[:, k, :] + 25) / 9.0))
            aM = 0.182 * (V[:, k, :] + 35) / (1 - torch.exp((-V[:, k, :] - 35) / 9.0))
            aH = 0.25 * torch.exp((-V[:, k, :] - 90) / 12.0)
                
            bN = -0.002 * (V[:, k, :] - 25) / (1 - torch.exp((V[:, k, :] - 25) / 9.0))
            bM = -0.124 * (V[:, k, :] + 35) / (1 - torch.exp((V[:, k, :] + 35) / 9.0))
            bH = 0.25 * torch.exp((V[:, k, :] + 34) / 12.0)
            
            if torch.any(V[:, k, :] == 25) or torch.any(V[:, k, :] == -35):
                aN[torch.where(V[:, k, :] == 25)] = 0.18
                bN[torch.where(V[:, k, :] == 25)] = 0.08
                aM[torch.where(V[:, k, :] == -35)] = 1.638
                bM[torch.where(V[:, k, :] == -35)] = 1.16

            m[:, k, :] = (aM * dt + (1 - dt / 2 * (aM + bM)) * m[:, k-1, :].clone()) / (dt / 2 * (aM + bM) + 1)
            n[:, k, :] = (aN * dt + (1 - dt / 2 * (aN + bN)) * n[:, k-1, :].clone()) / (dt / 2 * (aN + bN) + 1)
            h[:, k, :] = (aH * dt + (1 - dt / 2 * (aH + bH)) * h[:, k-1, :].clone()) / (dt / 2 * (aH + bH) + 1)    
            y[:, k, :] = (a_d * z[:, k-1, :] * dt + (1 - dt / 2 * (a_d * z[:, k-1, :] + a_r)) * y[:, k-1, :].clone()) / (dt / 2 * (a_d * z[:, k-1, :] + a_r) + 1)

        T = torch.sigmoid((V - Vt) / Kp)
        return T

class HH_Gap(nn.Module):
    def __init__(self, L):
        super().__init__()
        self.L = L
        self.V = None
       
    def forward(self, z):
        B, N = z.shape[:2] # Batch size and number of timesteps
        gna = CFG.gna; gk = CFG.gk; gl = CFG.gl;
        Ena = CFG.Ena; Ek = CFG.Ek; El = CFG.El;
        Iapp = CFG.Iapp; Vt = CFG.Vt; Kp = CFG.Kp; 
        dt = CFG.dt;
        
        m = torch.zeros((B, N, self.L)).to('cuda')
        n = torch.zeros((B, N, self.L)).to('cuda')
        h = torch.ones((B, N, self.L)).to('cuda')
        pow1 = torch.zeros((B, self.L)).to('cuda')
        pow2 = torch.zeros((B, self.L)).to('cuda')  
        self.V = torch.ones((B, N, self.L)).to('cuda') * -70.0
        
        for k in range(1, N):
           pow1 = gna * (m[:, k-1, :].clone() ** 3) * h[:, k-1, :].clone()
           pow2 = gk * n[:, k-1, :].clone() ** 4
           
           G_scaled = (dt / 2) * (pow1 + pow2 + gl)
           E = pow1 * Ena + pow2 * Ek + gl * El
           
           self.V[:, k, :] = (self.V[:, k-1, :].clone() * (1 - G_scaled) + dt * (E + Iapp + z[:, k-1, :])) / (1 + G_scaled)
           
           aN = 0.02 * (self.V[:, k, :] - 25) / (1 - torch.exp((-self.V[:, k, :] + 25) / 9.0))
           aM = 0.182 * (self.V[:, k, :] + 35) / (1 - torch.exp((-self.V[:, k, :] - 35) / 9.0))
           aH = 0.25 * torch.exp((-self.V[:, k, :] - 90) / 12.0)
               
           bN = -0.002 * (self.V[:, k, :] - 25) / (1 - torch.exp((self.V[:, k, :] - 25) / 9.0))
           if CFG.beta_n_modified:
               bN = 0.125 * torch.exp((-self.V[:, k, :] + 70) / 19.7)
           bM = -0.124 * (self.V[:, k, :] + 35) / (1 - torch.exp((self.V[:, k, :] + 35) / 9.0))
           bH = 0.25 * torch.exp((self.V[:, k, :] + 34) / 12.0)
           
           if torch.any(self.V[:, k, :] == 25) or torch.any(self.V[:, k, :] == -35):
               aN[torch.where(self.V[:, k, :] == 25)] = 0.18
               bN[torch.where(self.V[:, k, :] == 25)] = 0.08
               aM[torch.where(self.V[:, k, :] == -35)] = 1.638
               bM[torch.where(self.V[:, k, :] == -35)] = 1.16

           m[:, k, :] = (aM * dt + (1 - dt / 2 * (aM + bM)) * m[:, k-1, :].clone()) / (dt / 2 * (aM + bM) + 1)
           n[:, k, :] = (aN * dt + (1 - dt / 2 * (aN + bN)) * n[:, k-1, :].clone()) / (dt / 2 * (aN + bN) + 1)
           h[:, k, :] = (aH * dt + (1 - dt / 2 * (aH + bH)) * h[:, k-1, :].clone()) / (dt / 2 * (aH + bH) + 1)    

           # Lateral inhibition.
           if CFG.lat_inhibition:
               with torch.no_grad():
                    V_sub = self.V[:, k, :] + 70
                    below = V_sub.le(60)
                    self.V[:, k, :] = torch.where(below, V_sub * (below.sum(axis=1).reshape(-1,1) / self.L) - 70,  V_sub - 70)

        if CFG.plot:
            plt.figure(figsize=(15,5))
            plt.subplot(1,2,1)
            plt.plot(z[0, :, :].detach().cpu().numpy(), linewidth=1.0)
            plt.xticks(range(0,N+1,500), [f'{i*CFG.dt}' for i in range(0,N+1,500)])
            plt.xlabel('Time (ms)', fontsize=14)
            plt.title('A. Weighted Input', fontsize=18)
            plt.subplot(1,2,2)
            plt.plot(self.V[0, :, :].detach().cpu().numpy(), linewidth=1.0)
            plt.xticks(range(0,N+1,500), [f'{i*CFG.dt}' for i in range(0,N+1,500)])
            plt.xlabel('Time (ms)', fontsize=14)
            plt.title('B. Voltage Response', fontsize=18)

        T = torch.sigmoid((self.V - Vt) / Kp)
        return T