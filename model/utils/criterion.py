import torch
import torch.nn as nn
        
def narrow_gaussian(x, ell):
    return torch.exp(-0.5 * (x / ell) ** 2)

# The spice must flow
def approx_count_nonzero(x, ell=1e-3):
    return len(x) - narrow_gaussian(x, ell).sum(dim=-1)

def mean_over_frames(input, target):
    assert(input.shape == target.shape)
    T = input.shape[1]

    difference = input - target
    zeros = T - approx_count_nonzero(difference)
    mof = zeros / T
    return mof.mean()