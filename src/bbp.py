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
now = datetime.now()
time_str = now.strftime("%H_%M_%S___%m_%d_%Y")
config.serialize(f'../data/cfg_{time_str}.txt')

model = BNN()
model.load_state_dict(torch.load('C:/Users/jhazelde/BNBP-dev/src/pytorch_impl/HH_2000_model_200_I0_1.5_0.100000.pt'))

plt.imshow(model.Ws[1].weight.data.numpy(), cmap='seismic', aspect='auto')
plt.show()

plt.figure(figsize=(30,30))
W1 = model.Ws[0].weight.data.numpy()
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

plt.imshow(model.Ws[1].weight.data.numpy(), aspect = 'auto', cmap='seismic', interpolation='none')
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

model = BNN()
print("Model parameters:")
for name, param in model.named_parameters():
    if param.requires_grad:
        print(name)
        
model = model.to('cuda')
optimizer = torch.optim.Adam(model.parameters(), lr = CFG.lr)
loss_fun = nn.MSELoss().to('cuda')
for epch in range(10):
#for lr in [0.0001, 0.001, 0.01, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 1.0, 2.0, 10.0]:
    accuracy_out = open(f'../data/accuracy_{time_str}.txt', 'w')

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
        for W in model.Ws:
            W.weight.grad[torch.isnan(W.weight.grad)] = 0.0
        optimizer.step()
        
        if batch_idx % 20 == 0:
            print(batch_idx, float(loss.detach()), time.time() - start_time)
            start_time = time.time()
            
            v = V2_out[0, :, :].detach().cpu().numpy()
            plt.plot(v)
            plt.title("%d" % batch_idx)
            plt.show()
        
            if batch_idx % 20 == 0:
                plt.imshow(model.Ws[0].weight.grad.cpu().numpy(), aspect='auto', cmap='seismic')
                plt.colorbar()
                plt.show()
                
                plt.imshow(model.Ws[1].weight.grad.cpu().numpy(), aspect='auto', cmap='seismic')
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
    