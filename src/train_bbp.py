# Implements training on BNNs.
from config import CFG
import config
from spike_train_mnist import SpikeTrainMNIST
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from bnn import BNN
import torch
from torch import nn
import time
from datetime import datetime
import json
import matplotlib as mpl
torch.set_default_dtype(torch.float32)

# Configuration is done by modifying global CFG dictionary -- no need to send it as a parameter.
class Trainer:
    # If pretrained is a string, we read the model from a pretrained file. This can be used to train it more or to validate it.
    def __init__(self, save = True, pretrained = ''):
        now = datetime.now()
        time_str = now.strftime("%H_%M_%S___%m_%d_%Y")
        self.identifier = time_str + '_' + CFG.identifier
        self.model = BNN().to('cuda')
        if pretrained != '':
            self.load_model_from_file(pretrained)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = CFG.lr)

        print("Model parameters:")
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                print(name)
        
        # Serialize configuration dictionary.
        self.save = save        
        if self.save:
            fname = f'../data/cfg_{self.identifier}.txt'
            with open(fname, 'w') as fout:
                simple_dict = dict(CFG)
                print(config.CFG['use_DNN'])
                for key in simple_dict:
                    simple_dict[key] = simple_dict[key][0]
                fout.write(json.dumps(simple_dict, indent=1))
            
        # Load MNIST and spiketrain MNIST validation and training sets.
        self.load_mnist()

        
    def load_mnist(self):
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
        self.train_dataset = None # Deallocate        
        self.val_dataset = None # Deallocate
        self.train_dataset = SpikeTrainMNIST(train_mnist, 'train')
        self.val_dataset = SpikeTrainMNIST(test_mnist, 'validation')
        
    def load_model_from_file(self, pretrained):
        self.model.load_state_dict(torch.load(pretrained))
        self.model = self.model.to('cuda')

    # If batches_val is nonzero, every batches_val batches we will validate and save the model to a file.
    def train(self, epoch=0, batches_val=-1, custom_plotter=None):
        loss_fun = nn.MSELoss().to('cuda')
        loss_record = []
        start_time = time.time()
        train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=CFG.train_batch_sz, shuffle=True)
        batch_idx = 0
           
        # Train.
        for batch, expected in train_loader:
            
            changes = torch.load('../data/changes_4.pt')
            print(changes.shape)
            plt.imshow(torch.abs(changes).detach().cpu().numpy().transpose(), aspect='auto', cmap='hot', vmax=0.1)
            plt.colorbar()
         #   plt.xticks(range(0,CFG.sim_t+1,500), [f'{i*CFG.dt}' for i in range(0,CFG.sim_t+1,500)])
            plt.xlabel('Time (ms)')
            plt.ylabel('Weight (flattened)')
            plt.title('D. Gradients Raster Plot', fontsize=15)
            plt.savefig('../figures/changes_grid.pdf')
            plt.show()
            
            plt.plot(changes.cpu().numpy())
        #    plt.xticks(range(0,CFG.sim_t+1,500), [f'{i*CFG.dt}' for i in range(0,CFG.sim_t+1,500)])
            plt.title('C. Instantenous Gradients', fontsize=15)
            plt.xlabel('Time (ms)')
            plt.ylabel('Gradient')
            plt.savefig('../figures/changes_plot.pdf')
            plt.show()
            
            changes = torch.zeros(CFG.sim_t  // 25, 28 * 28 * 100).to('cuda')
            for k in range(2100 // 25, CFG.sim_t // 25):
                print(k)
                self.optimizer.zero_grad()   
                V2_out = self.model(batch.to('cuda'))
                out_avg = torch.mean(V2_out[:, k*25:(k+1)*25, :], dim=1)
                print(out_avg.shape)
                loss = loss_fun(out_avg, expected.to('cuda'))
                loss_record.append(loss.detach())
                           
                loss.backward()
                
                changes[k, :] = self.model.Ws[0].weight.grad[:, :].reshape(-1)
                
                plt.imshow(V2_out[0, :, :].detach().cpu().numpy().transpose(), aspect='auto', cmap='seismic')
                plt.colorbar()
                plt.xticks(range(0,CFG.sim_t+1,500), [f'{i*CFG.dt}' for i in range(0,CFG.sim_t+1,500)])
                plt.xlabel('Time (ms)')
                plt.ylabel('Neuron index')
                plt.title('B. Raster Plot', fontsize=15)
                plt.savefig('../figures/spiking_grid.pdf')
                plt.show()
                
                plt.plot(V2_out[0, :, :].detach().cpu().numpy())
                plt.xticks(range(0,CFG.sim_t+1,500), [f'{i*CFG.dt}' for i in range(0,CFG.sim_t+1,500)])
                plt.xlabel('Time (ms)')
                plt.ylabel('Neuron output')
                plt.title('A. Output Layer Traces', fontsize=15)
                plt.savefig('../figures/spiking_plot.pdf')
                plt.show()
                
            torch.save(changes, '../data/changes_4.pt')
            exit()
            
            # Set NaN values to zero! This is a hack way to fix this issue, but I think
            # NaNs only occur very rarely due to 0/0 in gating vars, so it is not a big issue.
            for W in self.model.Ws:
                W.weight.grad[torch.isnan(W.weight.grad)] = 0.0
            self.optimizer.step()
            
            if custom_plotter is not None:
                custom_plotter(self, batch, expected)
            
            if batch_idx % 20 == 0:
                print(batch_idx, float(loss.detach()), time.time() - start_time)
                start_time = time.time()
                
                v = V2_out[0, :, :].detach().cpu().numpy()
                plt.plot(v)
                plt.title("%d" % batch_idx)
                plt.show()
            
                if batch_idx % 20 == 0:
                    plt.imshow(self.model.Ws[0].weight.grad.cpu().numpy(), aspect='auto', cmap='seismic')
                    plt.colorbar()
                    plt.show()
                    
                    plt.imshow(self.model.Ws[1].weight.grad.cpu().numpy(), aspect='auto', cmap='seismic', vmin=-0.00025, vmax=0.0)
                    plt.colorbar()
                    plt.show()
                    
                    plt.plot(loss_record)
                    plt.title('Loss %d' % batch_idx)
                    plt.xlabel('Batch index')
                    plt.ylabel('Loss')
                    plt.show()  
                    
            batch_idx += 1
            
            if batches_val > 0 and batch_idx % batches_val == 0:
                self.validate(epoch, batch_idx)
                if self.save:
                    torch.save(self.model.state_dict(), f'../data/model_{self.identifier}_{epoch}_{batch_idx}.pt')

            
        if self.save:
           torch.save(self.model.state_dict(), f'../data/model_{self.identifier}_{epoch}.pt')
            
        plt.figure(dpi=600)
        plt.plot(loss_record)
        plt.title('Loss - Simple MNIST Example')
        plt.xlabel('Batch index')
        plt.ylabel('Loss')
        plt.show()
        
    def validate(self, epoch=0, batch_idx=-1, use_val_dataset=True):
        dataset = self.val_dataset if use_val_dataset else self.train_dataset
        val_loader = torch.utils.data.DataLoader(dataset, batch_size=CFG.test_batch_sz, shuffle=False)
        n_hit = 0
        n_total = 0
        start = time.time()
        for batch, expected in val_loader:
            if (n_total + 1) % 51 == 0:
                print(time.time() - start, n_total, n_hit / n_total * 100.0)
                start = time.time()
           
            with torch.no_grad():
                out_avg = torch.mean(self.model(batch.to('cuda')), dim=1)
                guess = torch.argmax(out_avg, dim=1).cpu()
                labels = torch.argmax(expected, dim=1)
                n_hit += torch.sum(guess == labels)
         #   print(time.time() - start, batch.shape)
            start = time.time()
            n_total += batch.shape[0]
            
        if self.save:
            fout_str = f'../data/accuracy_{self.identifier}_{epoch}'
            if batch_idx >= 0:
                fout_str += f'_{batch_idx}'
            fout_str += '.txt'
            accuracy_out = open(fout_str, 'w')
            print("%f" % (n_hit / n_total * 100.0), file=accuracy_out)
            accuracy_out.close()
        print("%f" % (n_hit / n_total * 100.0))
        return float(n_hit / n_total * 100.0)