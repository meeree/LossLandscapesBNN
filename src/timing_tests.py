from bnn import HH_Gap
import torch
from config import CFG
from matplotlib import pyplot as plt
import os
print(f'Using GPU: {torch.cuda.get_device_name(0)}')
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def single_layer(batch_size, timesteps, L):
    neurons = HH_Gap(L).cuda()
    inp = torch.zeros((batch_size, timesteps, L)).cuda()
    

    neurons(inp) # Warm-up

    neurons.eval()
    
    for i in range(5):
        with torch.no_grad():
            out = neurons(inp) 
        
    V = neurons.V[0, :, :].cpu().detach().numpy()
    plt.plot(V)
    plt.show()
print(torch.cuda.current_stream(torch.cuda.current_device()))
    
CFG.dt = 0.2
CFG.Iapp = 1
B = 1000
T = 5000
L = 100
single_layer(B, T, L)