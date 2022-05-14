# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 20:19:57 2022

@author: jhazelde
"""

import numpy as np
import matplotlib.pyplot as plt
from bnn import BNN
from config import CFG
import torch
import torchvision 
from torchaudio.datasets import SPEECHCOMMANDS
import torchaudio
import os
from progressbar import ProgressBar

if __name__ == '__main__':
    def memory_usage_psutil():
        # return the memory usage in MB
        import psutil
        process = psutil.Process(os.getpid())
        mem = process.memory_info().rss / float(2 ** 20)
        return mem
    
    class SubsetSC(SPEECHCOMMANDS):
        def __init__(self, subset: str = None):
            super().__init__("../", download=True)
    
            def load_list(filename):
                filepath = os.path.join(self._path, filename)
                with open(filepath) as fileobj:
                    return [os.path.normpath(os.path.join(self._path, line.strip())) for line in fileobj]
    
            if subset == "validation":
                self._walker = load_list("validation_list.txt")
            elif subset == "testing":
                self._walker = load_list("testing_list.txt")
            elif subset == "training":
                excludes = load_list("validation_list.txt") + load_list("testing_list.txt")
                excludes = set(excludes)
                self._walker = [w for w in self._walker if w not in excludes]
    
    # Create training and testing split of the data. We do not use validation in this tutorial.
    train_set = SubsetSC("training")
    test_set = SubsetSC("testing")
    
    def plot_waveforms_mel():
        i = int(torch.randint(0, 100, ()))
        waveform, sample_rate, label, speaker_id, utterance_number = train_set[i]
                
        print("Shape of waveform: {}".format(waveform.size()))
        print("Sample rate of waveform: {}".format(sample_rate))
        
        plt.subplot(1,2,1)
        plt.plot(waveform.t().numpy())
        
        spect = torchaudio.transforms.MelSpectrogram(n_fft=800, n_mels=100)
        stft = spect(waveform.squeeze())[:, :-1]
        stft = torch.log(stft + 1)
        stft = stft.repeat_interleave(dim=1, repeats=50)
        print(f"Shape of spectrogram: {stft.shape}")
        
        plt.subplot(1,2,2)
        plt.imshow(stft.numpy(), aspect='auto', cmap='seismic')
        plt.xticks(range(0, stft.shape[1]+1, 500))
        plt.yticks(range(0, stft.shape[0]+1, 20))
        plt.colorbar()
        plt.suptitle(label)
        plt.show()
    
    labels = sorted(list(set(datapoint[2] for datapoint in train_set)))
    def label_to_index(word):
        # Return the position of the word in labels
        return torch.tensor(labels.index(word))
    
    
    def index_to_label(index):
        # Return the word corresponding to the index in labels
        # This is the inverse of label_to_index
        return labels[index]
    
    def pad_sequence(batch):
        # Make all tensor in a batch the same length by padding with zeros
        batch = [item.t() for item in batch]
        batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.)
        return batch.permute(0, 1, 2)
    
    
    def collate_fn(batch):
        # A data tuple has the form:
        # waveform, sample_rate, label, speaker_id, utterance_number
        tensors, targets = [], []
    
        # Gather in lists, and encode labels as indices
        spect = torchaudio.transforms.MelSpectrogram(n_fft=800, n_mels=100)
        for waveform, _, label, *_ in batch:
            stft = spect(waveform.squeeze())[:, :-1] # Compute mel spectrogram.
            stft = torch.log(stft + 1)               # Take log to regularize range.
            stft = stft.repeat_interleave(dim=1, repeats=50) # Increase time dim to 2000.
            tensors += [stft]
            targets += [label_to_index(label)]
    
        # Group the list of tensors into a batched tensor
        tensors = pad_sequence(tensors)
        targets = torch.stack(targets)
    
        return tensors, targets
    
    CFG.lr = 0.004
    CFG.train_batch_sz = 50
    CFG.test_batch_sz = 400
    pin_memory = True
    CFG.plot = False
    
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=CFG.train_batch_sz,
        shuffle=True,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
    )
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=CFG.test_batch_sz,
        shuffle=True,
        drop_last=False,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
    )
    
    pbar = ProgressBar()
    model = BNN(100, 35).cuda()
    loss_fun = torch.nn.CrossEntropyLoss().cuda()
    loss_fun = torch.nn.MSELoss().cuda()
    optim = torch.optim.Adam(model.parameters(), lr = CFG.lr)
    loss_record = []
    print(len(train_loader))
    train_n_correct, train_n_total = 0, 0
    print(f"Memory usage just before training: {memory_usage_psutil()}")
    for batch, target in train_loader:
        idx = len(loss_record)
        if idx % 5 == 0:
            torch.save(model, f'../data/AUDIO_MSE_model_{idx}.pt')
            with torch.no_grad():
                n_correct, n_total = 0, 0
                for test_batch, test_target in test_loader:
                    out = torch.mean(model(test_batch.cuda()), dim = 1)
                    guess = torch.argmax(out, dim = 1)
                    n_correct += int((guess == test_target.cuda()).sum())
                    n_total += CFG.test_batch_sz
                    if n_total >= 800:
                        break
                print(f"{idx}: Validation Accuracy: {n_correct / float(n_total) * 100}%")
                if idx > 0:
                    print(f"{idx}: Running Training Accuracy: {train_n_correct / float(train_n_total) * 100}%")
        
        optim.zero_grad()
        out = torch.mean(model(batch.cuda()), dim=1)
        guess = torch.argmax(out, dim = 1)
        one_hot = torch.nn.functional.one_hot(target, num_classes=35) * 1.0
        
        loss = loss_fun(out, one_hot.cuda())
        loss_record.append(loss.detach())
        train_n_correct += int((guess == target.cuda()).sum())
        train_n_total += CFG.train_batch_sz
                      
        loss.backward()
        
        for W in model.Ws:
            W.weight.grad[torch.isnan(W.weight.grad)] = 0.0
                
        optim.step()
        
        plt.plot(loss_record)
        plt.show()
        
        print(loss.detach())