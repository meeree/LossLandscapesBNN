# Approximate gradient methods introduced in my work. 

import numpy as np
import torch

########################################################################################
# Compute regression-based gradients for the layers of model given loss samples.
# This is done by fitting a linear regression to the losses given the sampling offsets. 
#
# - losses: The losses associated with each sample. 
#                  It is required that first loss is the loss at the origin.
# - param_offsets: The offsets applied for each sample to the model parameters. 
# - parameters: The parameters (weights, biases, etc) of the original model. 
#                  Only needed for correctly reshaping array. 
# - stddev: Standard deviation used for sampling. Unused in this case.
#
########################################################################################

def compute_grad_regression(losses, param_offsets, parameters, stddev):
    losses = losses.cpu().detach().numpy()
    l0 = losses[0] # Loss at origin. Note that Noisy_BNN always samples first point at origin!
    grads = []
    for i in range(len(param_offsets)):
        smpls, param = param_offsets[i], parameters[i]
        S = smpls.shape[0]
        flat_noise = smpls.reshape((S, -1))
        flat_noise = flat_noise.cpu().detach().numpy()
        
        M, _, _, _ = np.linalg.lstsq(flat_noise, losses - l0)
        M = M.reshape(param.shape)
        grads.append(torch.from_numpy(M).to(param.device))
        
    return grads
        
########################################################################################

########################################################################################
# Compute Gaussian smoothed gradients for the layers of model given loss samples.
# This is done by Monte-Carlo estimation where we approximate the gradient as a 
# weighted sum over the losses which are sampled from a normal distribution.
# d/dv_i of log(p_theta(v)) is (mu_i - v_i) / sigma_i^2, we scale by this in weighted sum.
########################################################################################

def compute_grad_log_trick(losses, param_offsets, parameters, stddev):
    grads = []
    S = losses.shape[0]
    for i in range(len(param_offsets)):
        smpls = param_offsets[i].reshape(S, -1)
        
        # Sum 1/S * loss(v_i) * (mu_i - v_i) / sigma_i^2 
        grad = torch.sum(losses.reshape(S, 1) * smpls, 0) / (S * stddev**2)
        grads.append(grad.reshape(parameters[i].shape))

    return grads

########################################################################################

########################################################################################
# Compute Gaussian smoothed LOSS, not gradients! This is very simply done 
# by Monte-Carlo estimation where we average the loss samples.
# Loss samples are thus assumed to be from normal distribution.
########################################################################################

def compute_smoothed_loss(losses):
    return torch.mean(losses)

########################################################################################
