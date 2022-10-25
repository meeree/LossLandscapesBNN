# -*- coding: utf-8 -*-
"""
Created on Sat Oct 15 19:08:49 2022

@author: jhazelde
"""

import torch
import os
from torch import nn
from matplotlib import pyplot as plt
import numpy as np
from progressbar import ProgressBar
print(f'Using GPU: {torch.cuda.get_device_name(0)}')
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

losses, param_glob = None, None
grad_global = []

class WeakGradRaw(torch.autograd.Function):
    S, std, method = 1, 0.0, None
    
    # Forward call copies the input S times and adds noise to each copy except first copy.
    @staticmethod
    def forward(ctx, input):
        dim_repeats = [WeakGradRaw.S] + [1 for s in input.shape[1:]]
        output = input.repeat(dim_repeats)
        
        # Generate noise for noisy sampling with normal distribution.
        # The first sample does not get offset so we have the baseline loss without change.
        offsets = torch.normal(torch.zeros_like(output), WeakGradRaw.std)
        offsets[:input.shape[0]] = 0.0 # Don't offset first sample.   
        output = output + offsets 
        
        global param_glob
        param_glob = offsets.detach()
        ctx.save_for_backward(input, offsets)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, offsets, = ctx.saved_tensors
        
        # note we drop grad_output entirely in exchange for weak grad.
        loss_smpls = losses.reshape(WeakGradRaw.S, -1, *losses.shape[1:])
        smpls = offsets.reshape(WeakGradRaw.S, -1, *offsets.shape[1:])
        grad = WeakGradRaw.method(loss_smpls, [smpls], [input], WeakGradRaw.std)
        global grad_global
        grad_global.append(grad[0].clone())
        return grad[0]
    
class WeakGrad(nn.Module):
    def __init__(self, S, std, method = 'compute_grad_regression'):
        from gradient_methods import compute_grad_regression, compute_grad_log_trick
        super().__init__()
        self.func = WeakGradRaw.apply
        self.S = S
        self.std = std
        WeakGradRaw.S = self.S
        WeakGradRaw.std = self.std
        WeakGradRaw.method = eval(method)

    def forward(self, x): 
        return self.func(x)
    
layers = [nn.Sequential(nn.Linear(10 if i == 1 else 1, 10 if i == 0 else 1), nn.Sigmoid()) for i in range(3)]
for layer in layers:
    layer[0].weight.data = torch.zeros_like(layer[0].weight.data)
    layer[0].bias.data = torch.zeros_like(layer[0].bias.data)
    
#Evaluate loss landscape by inserting a weak gradient.
# plt.figure(dpi=500)
# for i in range(len(layers)):
#     plt.subplot(3, 3, i+1)
#     layers = [nn.Sequential(nn.Linear(1, 1, bias=False), nn.Sigmoid()) for i in range(10)]
#     layers[i] = nn.Sequential(nn.Linear(1, 1, bias = False), WeakGrad(1000, 10.0), nn.ReLU())
#     net = nn.Sequential(*layers)
#     inp = torch.ones(1,1)
#     l0, losses = evaluate(inp)
#     plt.plot(param_glob, losses.detach(), 'o')
# plt.show()

def simple_ann(weak_idx = None):
    from gradient_methods import compute_smoothed_loss
    if weak_idx is not None:
        layers[weak_idx] = nn.Sequential(layers[weak_idx][0], WeakGrad(int(1e2), 1e-1, 'compute_grad_log_trick'), layers[weak_idx][1])
    
    net = nn.Sequential(*layers)
    
    def evaluate(inp):
        global losses
        batch_size = inp.shape[0]
        out = net(inp).squeeze()
        inp = inp.repeat(WeakGradRaw.S, 1)
        losses = (out.squeeze() - inp.squeeze())**2
        losses = losses.reshape(-1, batch_size)
        losses = torch.mean(losses, 1)
        l0 = losses[0] if len(losses.shape) > 0 else losses
        return l0, losses
    
    optim = torch.optim.Adam(net.parameters(), lr=0.002)
    torch.manual_seed(0)
    
    loss_record = []
    smooth_loss_record = []
    grads, weights = [], []
    for i in range(3000):
        optim.zero_grad()
        batch_size = 1
        inp = torch.ones(batch_size,1) * 0.7
        l0, losses = evaluate(inp)
        smooth_loss_record.append(compute_smoothed_loss(losses).detach().item())
        loss_record.append(l0)
        l0.backward()
        optim.step()
    
        weights.append([torch.mean(w.detach()) for w in net.parameters()]) # Note the ::2 because we want weights, not biases
        grads.append([torch.mean(w.grad) for w in net.parameters()]) 
    
    for record, name in zip([grads, weights], ['Gradient', 'Value']):
        arr = np.array(record)
        plt.subplot(1, 2, 1)
        plt.imshow(arr.T[::2], aspect='auto', cmap = 'hot', interpolation = 'none')
        plt.title('Weights')
        plt.xlabel('Batch Index')
        plt.colorbar()
        plt.subplot(1, 2, 2)
        plt.imshow(arr.T[1::2], aspect='auto', cmap = 'hot', interpolation = 'none')
        plt.title('Biases')
        plt.suptitle(name)
        plt.xlabel('Batch Index')
        plt.colorbar()
        plt.show()
    
    plt.plot(loss_record, label = 'Loss')
    plt.plot(smooth_loss_record, label = 'Smoothed loss')
    plt.ylabel('Loss (output)')
    plt.xlabel('Batch Index')
    plt.legend()
    plt.show()

# CNN Cifar10 Code
if __name__ == '__main__':
    import torchvision
    import torchvision.transforms as transforms
    import PIL
    import ssl
    method_str = 'AutoGrad'
    method_full_str = 'compute_grad_regression' if method_str == 'RegGrad' else 'compute_grad_log_trick'
    ssl._create_default_https_context = ssl._create_unverified_context
    
    train_transform = transforms.Compose(
        [transforms.RandomHorizontalFlip(p=0.5),
         transforms.RandomAffine(degrees=(-5, 5), translate=(0.1, 0.1), scale=(0.9, 1.1), resample=PIL.Image.BILINEAR),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    test_transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    train_set, val_set = torch.utils.data.random_split(dataset, [40000, 10000])
    
    batch_size = 128
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=8)
    
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=8)
    
    classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    
    import torch.nn.functional as F
    
    class Net(nn.Module):
      def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 128, 5, padding=2)
        self.conv2 = nn.Conv2d(128, 128, 5, padding=2)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv4 = nn.Conv2d(256, 256, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.bn_conv1 = nn.BatchNorm2d(128)
        self.bn_conv2 = nn.BatchNorm2d(128)
        self.bn_conv3 = nn.BatchNorm2d(256)
        self.bn_conv4 = nn.BatchNorm2d(256)
        self.bn_dense1 = nn.BatchNorm1d(1024)
        self.bn_dense2 = nn.BatchNorm1d(512)
        self.dropout_conv = nn.Dropout2d(p=0.25)
        self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(256 * 8 * 8, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 10)
        self.weak = WeakGrad(1000, 1e1, method_full_str)
    
      def conv_layers(self, x):
        out = F.relu(self.bn_conv1(self.conv1(x)))
        out = F.relu(self.bn_conv2(self.conv2(out)))
        out = self.pool(out)
        out = self.dropout_conv(out)
        out = F.relu(self.bn_conv3(self.conv3(out)))
        out = F.relu(self.bn_conv4(self.conv4(out)))
        out = self.pool(out)
        out = self.dropout_conv(out)
        return out
    
      def dense_layers(self, x):
        out = F.relu(self.bn_dense1(self.fc1(x)))
        out = self.dropout(out)
        out = F.relu(self.bn_dense2(self.fc2(out)))
        out = self.dropout(out)
        out = self.fc3(out)
        return out
    
      def forward(self, x):
        out = self.conv_layers(x)
        out = out.view(-1, 256 * 8 * 8)
        out = self.weak(self.dense_layers(out))
        return out
    
    net = Net()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Device:', device)
    net.to(device)
    
    num_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print("Number of trainable parameters:", num_params)
    
    
    import torch.optim as optim
    
    criterion = nn.CrossEntropyLoss(reduction='none') # No reduction (mean) applied!
    optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True, min_lr=0)
    
    loss_hist, acc_hist = [], []
    loss_record, accuracy_record = [], []
    loss_hist_val, acc_hist_val = [], []
    
    def eval_loss_weak_grad(batch, labels):
        global losses
        outputs = net(batch)
        
        repeat_dims = [WeakGradRaw.S] + [1 for s in labels.shape[1:]]
        start_dim = labels.shape[0]
        labels = labels.repeat(*repeat_dims)
                             
        losses = criterion(outputs, labels) # Same shape as outputs.
        
        # Break losses into shape (S, batch_size, ...)
        losses = losses.reshape(-1, batch_size, *losses.shape[1:])
        mean_dims = list(range(len(losses.shape)))[1:]
        losses = torch.mean(losses, mean_dims) # Reduce loss in all dims except first.
        loss = losses[0] # Loss without any randomness in network.
        return loss, outputs[:start_dim]
    
    for data in train_loader:
        batch, labels = data
        batch, labels = batch.to(device), labels.to(device)
        break
    
    T = np.linspace(0.0, 1.0, 140)
    net.load_state_dict(torch.load('../data/net_CIFAR10_AutoGrad_0_0.pt'))
    init_W, init_b = net.fc3.weight.data.clone(), net.fc3.bias.data.clone()
    net.load_state_dict(torch.load('../data/net_CIFAR10_AutoGrad_0_310.pt'))
    direction_W = net.fc3.weight - init_W
    direction_b = net.fc3.bias - init_b
    
    for use_smooth in [True, False]:
        loss_interpolate = []
        for idx, (t, data) in enumerate(zip(T, train_loader)):
            # batch, labels = data
            # batch, labels = batch.to(device), labels.to(device)
            
            from gradient_methods import compute_smoothed_loss
            print(idx, direction_W.sum())
            net.fc3.weight.data = init_W + t * direction_W 
            net.fc3.bias.data = init_b + t * direction_b
            net.load_state_dict(torch.load(f'../data/net_CIFAR10_AutoGrad_{idx}.pt'))
            loss, outputs = eval_loss_weak_grad(batch, labels) 
            smooth_loss = compute_smoothed_loss(losses)
            if use_smooth:
                loss_interpolate.append(smooth_loss.cpu().item())
            else:
                loss_interpolate.append(loss.cpu().item())
            
        plt.plot(list(range(len(T))), loss_interpolate)
        plt.title('Fixed Batch. ' + ('Smooth' if use_smooth else 'Singular') + ' Loss')
        plt.xlabel('Epoch')
        plt.ylabel('$\mathcal{L}_{\\theta}^{smooth}(x)$' if use_smooth else '$f(N_{\\theta}(x))$')
        plt.show()
    exit()
        
    
    for epoch in range(140):
      model_str = f'../data/net_CIFAR10_{method_str}_{epoch}.pt'
      torch.save(net.state_dict(), model_str)
      running_loss = 0.0
      correct = 0
      pbar = ProgressBar()
      for idx, data in enumerate(pbar(train_loader)):
        if idx == len(train_loader) - 1: # Batch size is messed up on last sample. Skip for now.
            break 
        
        batch, labels = data
        batch, labels = batch.to(device), labels.to(device)
        
        optimizer.zero_grad()
        loss, outputs = eval_loss_weak_grad(batch, labels)
        loss.backward()
        loss_record.append(loss)
        optimizer.step()
        
        # compute training statistics
        _, predicted = torch.max(outputs, 1)
        correct_batch = (predicted == labels).sum().item()
        accuracy_record.append(100 * correct_batch / batch_size)
        correct += correct_batch
        running_loss += loss.item()
    
      avg_loss = running_loss / len(train_set)
      avg_acc = correct / len(train_set)
      loss_hist.append(avg_loss)
      acc_hist.append(avg_acc)
           
      if epoch % 10 == 0:
          for v, name in zip([loss_record, accuracy_record], ['Loss', 'Train Accuracy (%)']):
              plt.figure()
              plt.plot(v)
              plt.title(f'Epoch = {epoch}, {name}, {method_str} S = {WeakGradRaw.S}, std = {WeakGradRaw.std}')
              plt.savefig(f'cifar10_fig_{method_str}_{epoch}_{name}.pdf')
              plt.xlabel('Batch Index')
              plt.show()
          
      # plt.figure(dpi=500)
      # plt.plot([torch.mean(g).cpu().item() for g in grad_global])
      # plt.title(f'Epoch = {epoch}, Mean Gradient; {method_str}, S = {WeakGradRaw.S}, std = {WeakGradRaw.std}')
      # plt.show()
    
      # validation statistics
      if False:
          net.eval()
          with torch.no_grad():
              loss_val = 0.0
              correct_val = 0
              pbar = ProgressBar()
              for idx, data in enumerate(pbar(val_loader)):
                  if idx == len(val_loader) - 1: # Batch size is messed up on last sample. Skip for now.
                      break 
                  batch, labels = data
                  batch, labels = batch.to(device), labels.to(device)
    
                  loss, outputs = eval_loss_weak_grad(batch, labels)
                  _, predicted = torch.max(outputs, 1)
                  correct_val += (predicted == labels).sum().item()
                  loss_val += loss.item()
              avg_loss_val = loss_val / len(val_set)
              avg_acc_val = correct_val / len(val_set)
              loss_hist_val.append(avg_loss_val)
              acc_hist_val.append(avg_acc_val)
          net.train()
    
          scheduler.step(avg_loss_val)
          print('[epoch %d] loss: %.5f accuracy: %.4f val loss: %.5f val accuracy: %.4f' % (epoch + 1, avg_loss, avg_acc, avg_loss_val, avg_acc_val))
