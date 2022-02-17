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

torch.set_default_dtype(torch.float32)

# Configuration is done by modifying global CFG dictionary -- no need to send it as a parameter.
class Trainer:
    # If pretrained is a string, we read the model from a pretrained file. This can be used to train it more or to validate it.
    def __init__(self, pretrained = ''):
        now = datetime.now()
        time_str = now.strftime("%H_%M_%S___%m_%d_%Y")
        self.identifier = time_str + '_' + CFG.identifier
        self.model = BNN()
        if len(pretrained) > 0:
            self.model.load_state_dict(torch.load(pretrained))
        self.model = self.model.to('cuda')
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = CFG.lr)

        print("Model parameters:")
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                print(name)
        
        # Serialize configuration dictionary.
        config.serialize(f'../data/cfg_{self.identifier}.txt')
        
        # Load MNIST and spiketrain MNIST validation and training sets.
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
        self.train_dataset = SpikeTrainMNIST(train_mnist, 'train')
        self.val_dataset = SpikeTrainMNIST(test_mnist, 'validation')

    def train(self, epoch=0):
        loss_fun = nn.MSELoss().to('cuda')
        loss_record = []
        start_time = time.time()
        train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=CFG.train_batch_sz, shuffle=True)
        batch_idx = 0
           
        # Train.
        for batch, expected in train_loader:
            self.optimizer.zero_grad()   
            V2_out = self.model(batch.to('cuda'))
            out_avg = torch.mean(V2_out, dim=1)
            loss = loss_fun(out_avg, expected.to('cuda'))
            loss_record.append(loss.detach())
                       
            loss.backward()
            
            # Set NaN values to zero! This is a hack way to fix this issue, but I think
            # NaNs only occur very rarely due to 0/0 in gating vars, so it is not a big issue.
            for W in self.model.Ws:
                W.weight.grad[torch.isnan(W.weight.grad)] = 0.0
            self.optimizer.step()
            
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
                    
                    plt.imshow(self.model.Ws[1].weight.grad.cpu().numpy(), aspect='auto', cmap='seismic')
                    plt.colorbar()
                    plt.show()
                    
                    plt.plot(loss_record)
                    plt.title('Loss %d' % batch_idx)
                    plt.xlabel('Batch index')
                    plt.ylabel('Loss')
                    plt.show()  
            
            batch_idx += 1
            
        torch.save(self.model.state_dict(), f'../data/self.model_{self.identifier}_{epoch}.pt')
            
        plt.figure(dpi=600)
        plt.plot(loss_record)
        plt.title('Loss - Simple MNIST Example')
        plt.xlabel('Batch index')
        plt.ylabel('Loss')
        plt.show()
        
    def validate(self, epoch=0):
        val_loader = torch.utils.data.DataLoader(self.val_dataset, batch_size=CFG.test_batch_sz, shuffle=False)
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
            print(time.time() - start, batch.shape)
            start = time.time()
            n_total += batch.shape[0]
            
        accuracy_out = open(f'../data/accuracy_{self.identifier}_{epoch}.txt', 'w')
        print("%f : %f" % (CFG.lr, n_hit / n_total * 100.0))
        print("%f : %f" % (CFG.lr, n_hit / n_total * 100.0), file=accuracy_out)
        accuracy_out.close()