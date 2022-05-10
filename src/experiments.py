# Experiments with different parameters for training, etc.
from train_bbp import Trainer
from config import CFG
import config
import numpy as np
import matplotlib.pyplot as plt
import json
from os.path import exists
import torch
from bnn import BNN

# # # Analyze pretrained model. No actual training. 
# # CFG.Iapp = 0.0
# # #CFG.hidden_layers = [100, 20]
# #CFG.dt = 0.07
# # CFG.Kp = 0.8
# # CFG.Vt = 18.0
# # CFG.poisson_max_firings_per = 10
# # CFG.poisson_n_timesteps_spike = 2
# # CFG.lr = 0.1
# # CFG.n_samples_train = 0
# # CFG.n_samples_val = 1000 
# CFG.n_samples_train = 0
# CFG.n_samples_val = 1000
# CFG.test_batch_sz = 500
# trainer = Trainer(False, 'C:/Users/jhazelde/BNBP-dev/src/pytorch_impl/RERUN_HH_2000_2_model_200_I0_0.500000_0.001000.pt')
# trainer.validate()

# plt.figure(figsize=(30,30))
# W1 = trainer.model.Ws[0].cpu().weight.data.numpy()
# vmin, vmax = W1.min(), W1.max()
# print(vmin, vmax)
# W1 = W1.reshape(10, 10, 28, 28)

# for i in range(10):
#     for j in range(10):     
#         plt.subplot(10, 10, i + j * 10 + 1)
#         plt.imshow(W1[i, j, :, :], cmap='seismic', interpolation='bilinear')
#         plt.box(False)
#         plt.xticks([])
#         plt.yticks([])

# plt.show()

# plt.imshow(trainer.model.Ws[1].cpu().weight.data.numpy(), aspect = 'auto', cmap='seismic', interpolation='none')
# plt.colorbar()
# plt.show()

# exit()

# TODO : run for 20 epochs with 2000 samples
# run for 20 epochs with 4000 samples 
# Run this again with 200 hidden neurons
# Put all three on same plot over 20 epochs
# ALSO INCLUDE BASELINE

# CFG.n_samples_val = 500
# CFG.test_batch_sz = 250
# CFG.train_batch_sz = 50
# #CFG.dt = 0.02 # 2 times bigger DT
# for lr in [0.001]:
#     CFG.lr = lr
# #    CFG.hidden_layers = [100, 100]
#     for n_samples in [3000]:
#         print(n_samples)
#         CFG.n_samples_train = n_samples
        
#         def train():    
#             trainer = Trainer(False)
#             for epch in range(20):
#                 trainer.train(epch, 10)
#                 trainer.validate()        
        
#         CFG.plot = False
#         CFG.identifier = f'FULL_EPOCHS_1_lr_{lr}_'
#         trainer = Trainer(True)
#         trainer.train(0, 10)
        
#         # CFG.identifier = f'INCREASED_APCUR_lr_{lr}_'
#         # CFG.Iapp = 2.0
#         # train()
        
#         # Reset
#         # CFG.Iapp = 0.5
        
#         # CFG.lat_inhibition = True
#         # CFG.identifier = f'INHIBITION_NO_GRAD_lr_{lr}_'
#         # train()
        
#         # # Reset
#         # CFG.lat_inhibition = False
        
#         # CFG.beta_n_modified = True
#         # CFG.identifier = f'BETA_N_lr_{lr}_'
#         # train()
        
#         # # Reset
#         # CFG.beta_n_modified = False
     

#config.CFG.use_DNN = True
CFG.n_samples_val = 500
CFG.test_batch_sz = 250
CFG.train_batch_sz = 50
#CFG.dt = 0.02 # 2 times bigger DT  
for lr in [0.002]:
    CFG.lr = lr
#    CFG.hidden_layers = [100, 100]
    for n_samples in [60000]:
        CFG.identifier = f'RERUN_{lr}_2'
        CFG.n_samples_train = 2000
        CFG.plot = False
        print(CFG.use_DNN)
        trainer = Trainer()
        for i in range(n_samples // 2000):
            print(i)
            CFG.train_offset = i * 2000
            trainer.load_mnist() # Reload mnist with new train_offset.
            trainer.train(i, 10)
        # trainer.validate()

exit()

CFG.n_samples_val = 500
CFG.test_batch_sz = 250 

CFG.train_batch_sz = 10
for lr in [0.1]:
    CFG.lr = lr
    for n_samples in [4000]:
        print(n_samples)
        CFG.n_samples_train = n_samples
        CFG.identifier = str(n_samples) + '_ACCURACY_OVER_BATCHES'
        trainer = Trainer()
        trainer.train(0, 20)
        trainer.validate()
        
exit()
            
            
            
            
            
CFG.n_samples_train = 1
CFG.n_samples_val = 0 
CFG.train_batch_sz = 1 
CFG.dt = 0.05
CFG.sim_t = 4000
#CFG.plot = False


sliding_grad_1 = torch.load('../data/20_20_40___05_01_2022_R_10_10_slidingGrad1.pt')
sliding_grad_2 = torch.load('../data/20_20_40___05_01_2022_R_10_10_slidingGrad2.pt')
sliding_grad_1 = sliding_grad_1.cpu().detach().numpy()
sliding_grad_2 = sliding_grad_2.cpu().detach().numpy()

model = BNN()
model.load_state_dict(torch.load('../data/model_12_55_30___03_29_2022_48000_DATASET_lr_0.002__23.pt'))
model = model.cuda()
trainer = Trainer()
# Pick a random training sample to look at. 
idx = int(torch.randint(0, len(trainer.train_dataset), ()))
sample, target = trainer.train_dataset[idx]
target = target.cuda()
sample = sample.unsqueeze(0)
        
# Feed forward.
T2_out = model(sample.cuda())

for idx in range(5):
    plt.subplot(3,1,3)
    V1 = model.layers[0].V[0, :, idx]
    V1 = torch.sigmoid((V1 - CFG.Vt) / CFG.Kp).cpu().detach().numpy()
    plt.plot(V1)
    plt.xticks(range(0,CFG.sim_t+1,500), [f'{i*CFG.dt}' for i in range(0,CFG.sim_t+1,500)])
    
    plt.subplot(3,1,2)
    V2 = model.layers[1].V[0, :, :]
    V2 = torch.sigmoid((V2 - CFG.Vt) / CFG.Kp).cpu().detach().numpy()
    plt.plot(V2)
    plt.xticks([])
    
    plt.subplot(3,1,1)
    plt.plot(sliding_grad_1[:, 28*28*idx:28*28*(idx+1)])
    plt.xticks([])
    plt.savefig(f'../figures/grad_hidden_{idx}.pdf')
    plt.show()
            
            

changes = torch.load('../data/20_20_40___05_01_2022_R_10_10_slidingGrad2.pt')
print(changes.shape)
plt.imshow(torch.abs(changes).detach().cpu().numpy().transpose(), aspect='auto', cmap='hot', vmax=0.1)
plt.colorbar()
plt.xticks(range(0,401,50), [f'{i*CFG.dt*10}' for i in range(0,401,50)])
plt.xlabel('Time (ms)')
plt.ylabel('Weight (flattened)')
plt.title('D. Output Gradients Raster Plot', fontsize=15)
plt.savefig('../figures/changes_grid_2.pdf')
plt.show()

plt.plot(changes.cpu().numpy())
plt.xticks(range(0,401,50), [f'{i*CFG.dt*10}' for i in range(0,401,50)])
plt.title('C. Output Instantenous Gradients', fontsize=15)
plt.xlabel('Time (ms)')
plt.ylabel('Gradient')
plt.savefig('../figures/changes_plot_2.pdf')
plt.show()
            


def custom_plotter(trainer, batch, expected):
    print(torch.argmax(expected[0]))

trainer = Trainer(False, '../data/model_12_55_30___03_29_2022_48000_DATASET_lr_0.002__23.pt')
trainer.measure_sliding_gradients(10)
exit()
trainer.train(custom_plotter = custom_plotter) 
exit()