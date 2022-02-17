from config import CFG
import torch
from torch.distributions.exponential import Exponential
from torch.utils.data import Dataset
from os.path import exists

torch.set_default_dtype(torch.float32)

def to_spiketrain (output, sample, total_timesteps, max_firings, n_timesteps_spike):
    for pix_id, s in enumerate(sample.flatten()):
        if s < 0.01:
            continue # No spikes, pixel is black.
            
        rate = (max_firings * s) / total_timesteps
        exp = Exponential(rate)
        i = 0
        while i < total_timesteps:
            period = exp.sample()
            i += int(period)
            end_pt = min(total_timesteps, i+n_timesteps_spike)
            output[i:end_pt, pix_id] = 1.0
            i = end_pt

class SpikeTrainMNIST(Dataset):
    # phase should be one of 'test', 'train', 'validation'
    def __init__(self, mnist_dset, phase):
        offset = 0
        if phase == 'test':
            n_samples = CFG.n_samples_test
            offset = CFG.n_samples_val # Split testing data into (validation U testing) disjoint union.
        elif phase == 'train':
            n_samples = CFG.n_samples_train
        elif phase == 'validation':
            n_samples = CFG.n_samples_val
        else:
            print(f'ERROR: Invalid phase for MNIST data: {phase}')
            raise ValueError(phase)
        print(f'Loading spiketrains for phase: {phase}, n_samples = {n_samples}')
        
        # If in validation, use first n_samples_val
        self.spiketrains = torch.zeros((n_samples, CFG.sim_t, 28*28))
        self.labels = torch.nn.functional.one_hot(mnist_dset.targets[offset:], num_classes=10) * 1.0
        fname = '../data/spiketrains'
        fname += '_' + phase
        fname += '_' + str(n_samples)
        fname += '_' + str(CFG.sim_t)
        fname += '_' + str(CFG.poisson_max_firings_per)        
        fname += '_' + str(CFG.poisson_n_timsteps_spike)
        fname += '.pt'
                
        if exists(fname):
            self.spiketrains = torch.load(fname)    
        else:
            for i in range(n_samples):
                img = mnist_dset[offset + i]
                if (i+1) % 500 == 0:
                    print("%.1f%%" % (100 * i / n_samples))
                to_spiketrain(self.spiketrains[i, :, :], img[0][0, :, :], CFG.sim_t, CFG.poisson_max_firings_per, CFG.poisson_n_timsteps_spike)
            torch.save(self.spiketrains, fname)

    def __len__(self):
        return self.spiketrains.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.spiketrains[idx, :, :], self.labels[idx, :]