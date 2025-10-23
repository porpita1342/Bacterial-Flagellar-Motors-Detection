import torch
import torch.nn.functional as F
import random
import torch.nn as nn
from torch.distributions import Beta

class Mixup(nn.Module):
    def __init__(self, mix_beta, mixadd=False):

        super(Mixup, self).__init__()
        self.beta_distribution = Beta(mix_beta, mix_beta)
        self.mixadd = mixadd

    def forward(self, X, Y, Z=None):

        bs = X.shape[0]
        n_dims = len(X.shape)
        perm = torch.randperm(bs)
        coeffs = self.beta_distribution.rsample(torch.Size((bs,))).to(X.device)
        X_coeffs = coeffs.view((-1,) + (1,)*(X.ndim-1))
        Y_coeffs = coeffs.view((-1,) + (1,)*(Y.ndim-1))
        
        X = X_coeffs * X + (1-X_coeffs) * X[perm]

        if self.mixadd:
            Y = (Y + Y[perm]).clip(0, 1)
        else:
            Y = Y_coeffs * Y + (1 - Y_coeffs) * Y[perm]
                
        if Z is not None:
            return X, Y, Z

        return X, Y

def rotate(x, mask= None, dims= ((-3,-2), (-3,-1), (-2,-1)), p= 1.0):
    """
    Rotate pixels.
    Mask is the target block
    """
    bs= x.shape[0]
    for d in dims:
        if random.random() < p:
            k = random.randint(0,3)
            x = torch.rot90(x, k=k, dims=d)
            if mask is not None:
                mask = torch.rot90(mask, k=k, dims=d) 

    if mask is not None:
        return x, mask
    else:
        return x
    

def flip_3d(x, mask= None, dims=(-3,-2,-1), p= 0.5):
    """
    Flip along axis.
    """
    axes = [i for i in dims if random.random() < p]
    if axes:
        x = torch.flip(x, dims=axes)
        if mask is not None:
            mask = torch.flip(mask, dims=axes)
        
    if mask is not None:
        return x, mask
    else:
        return x
    

