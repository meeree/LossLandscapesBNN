# Biological neural network models.
from config import CFG
import matplotlib.pyplot as plt
import torch
from torch import nn
import snntorch as snn
from snntorch import surrogate

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
    
# A BNN with S copies of the weight + random noise.
class Noisy_Weights_BNN(nn.Module):
    def __init__(self, S, input_dim=28*28, output_dim=10):
        super().__init__()
        NeuronModel = eval(CFG.neuron_model)
        
        self.dims = [input_dim] + CFG.hidden_layers + [output_dim]
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
            if CFG.use_DNN:
                self.layers.append(torch.nn.Sigmoid())
            else:
                self.layers.append(NeuronModel(d2))
                
        self.Ws = torch.nn.ParameterList(self.Ws) # Makes pytorch notice these.
        self.ticker = 0 # For picking parameter to optimize.
            
       
    # If T_sim is positive, we split simulation into temporal packets based on T_sim. 
    # This is important because of memory constraints.
    # If direction is not None, it specifies a direction along which to sample; 
    # this replaces sampling with normal distribution. Useful for evaluating loss landscape.
    def forward(self, batch : torch.Tensor, stddev, T_sim=-1, direction=None) -> torch.Tensor:
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
            
        for layer in self.layers:
            layer.INIT = True
        tsteps = batch.shape[1]
        if T_sim < 0:
            T_sim = tsteps # One temporal packet
            
        
        n_packets = (tsteps + T_sim - 1) // T_sim # Round up
        res = torch.zeros((self.S, batch.shape[0], tsteps, self.dims[-1])).cuda()
        for p in range(n_packets):
            # Get next temporal packet of input and simulate result..
            start, end = T_sim*p, T_sim*(p+1)
            T = batch[:, start:end, :].unsqueeze(0).repeat(self.S, 1, 1, 1)
                      
            for idx, (W, layer) in enumerate(zip(self.Ws_noisy, self.layers)):
                # Note that W is already transposed.
                z = torch.matmul(T, W)
                T = layer(z.reshape((-1, z.shape[2], z.shape[3]))) 
                T = T.reshape((self.S, -1, T.shape[1], T.shape[2]))
                layer.INIT = False
            res[:, :, start:end, :] = T
        return res
    
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
    
import torch.jit as jit
class HH_Gap(jit.ScriptModule):
    def __init__(self, L):
        super().__init__()
        self.L = L
        self.V = torch.ones(())
        self.gna = CFG.gna; self.gk = CFG.gk; self.gl = CFG.gl;
        self.Ena = CFG.Ena; self.Ek = CFG.Ek; self.El = CFG.El;
        self.Iapp = CFG.Iapp; self.Vt = CFG.Vt; self.Kp = CFG.Kp; 
        self.dt = CFG.dt;
        self.INIT = True
       
    @jit.script_method
    def forward(self, z):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record(torch.cuda.current_stream(torch.cuda.current_device()))
        B, N = z.shape[:2] # Batch size and number of timesteps
        
        # Gating variables
        K = torch.zeros((3, B, N, self.L)).cuda()
        K[2] = 1.0 # Start h gating variable at 1 since it is inverse.
        self.V = torch.ones((B, N, self.L)).cuda() * -70.0
        
        aK = torch.zeros((3, B, self.L)).cuda()
        bK = torch.zeros((3, B, self.L)).cuda()
        
        # Offsets and divisors for gating varialbe update rates.
        # For their uses, see below.
        offs = torch.tensor([35.0, -25.0, 90.0, 35.0, -25.0, 34.0]).view(-1, 1, 1).cuda()
        divs = torch.tensor([-9.0, -9.0, -12.0, 9.0, 9.0, 12.0]).view(-1, 1, 1).cuda()
        muls = torch.tensor([0.182, 0.02, 0.0, -0.124, -0.002, 0.0]).view(-1, 1, 1).cuda()
        
        if self.INIT:
            self.V = torch.ones((B, N, self.L)).cuda() * -70.0

        for k in range(1, N):
           # Optimization: concatenate all gating variables in one big tensor since their updates are very similar.
           m = K[0, :, k-1, :]
           n = K[1, :, k-1, :]
           h = K[2, :, k-1, :]

           # Calculate V intermediate channel quantities.
           pow1 = self.gna * (m ** 3) * h
           pow2 = self.gk * n ** 4
           G_scaled = (self.dt / 2) * (pow1 + pow2 + self.gl)
           E = pow1 * self.Ena + pow2 * self.Ek + self.gl * self.El
           
           # V update.
           self.V[:, k, :] = (self.dt * (E + self.Iapp + z[:, k-1, :]) +  (1 - G_scaled) * self.V[:, k-1, :]) / (1 + G_scaled)
           
           # Calculate gating variable intermediate rate quantities.
           v_off = self.V[:, k, :] + offs
           EXP = torch.exp(v_off / divs) # Optimization: do all exponentials at once. I've found this to shave ~20% time off.           
           scaled_frac = muls / (1 - EXP) # Optimization: compute these terms in a batch. MOST HAVE FORM: k * (v - off) / (1 - exp).
           
           aK[:2] = v_off[:2] * scaled_frac[:2]
           aK[2] = 0.25 * EXP[2]
           bK[:2] = v_off[3:5] * scaled_frac[3:5]          
           bK[2] = 0.25 * EXP[5]
           aK = torch.nan_to_num(aK, 0.0) # Handle division by zero in EXP.
           bK = torch.nan_to_num(bK, 0.0)
           
           # Gating Variable update.
           sum_scaled = self.dt/2 * (aK+bK)
           K[:, :, k, :] = (self.dt * aK + (1 - sum_scaled) * K[:, :, k-1, :]) / (1 + sum_scaled) # Note similarity with V update above

        T = torch.sigmoid((self.V - self.Vt) / self.Kp)
        end.record(torch.cuda.current_stream(torch.cuda.current_device()))
        torch.cuda.synchronize()
        # fraction_of_second = 1000 / (N * self.dt)
        # time = start.elapsed_time(end) / 1000
        # print(time, ', for full second : ', fraction_of_second * time)
        return T
    
class LIF(nn.Module):
    def __init__(self, L):
        super().__init__()
        self.L = L
        self.V = None
        self.INIT = True

    def forward(self, z):
        B, N = z.shape[:2] # Batch size and number of timesteps
        if self.INIT:
            self.V = torch.zeros_like(z)
        T = torch.zeros_like(self.V)
        lif = snn.Leaky(beta=CFG.lif_beta, spike_grad = surrogate.fast_sigmoid()).to('cuda')
        
        # Simulate LIF neuron over time.
        noise = torch.normal(torch.zeros_like(self.V), 0.0)
        for k in range(1, N):
            T[:, k, :], self.V[:, k, :] = lif(z[:, k-1, :] + CFG.Iapp + noise[:, k, :], self.V[:, k-1, :])
        return T