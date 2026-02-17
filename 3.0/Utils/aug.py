import torch
import random
import torch.nn as nn
from torch.distributions import Beta

class Mixup(nn.Module):
    def __init__(self, mix_beta, mixadd=False):

        super(Mixup, self).__init__()
        self.beta_distribution = Beta(mix_beta, mix_beta)
        self.mixadd = mixadd

    def forward(self, X, Y, Z=None):

        perm = torch.randperm(X.shape[0])
        coeffs = self.beta_distribution.rsample(torch.Size((X.shape[0],))).to(X.device)
        X_coeffs = coeffs.view((-1,) + (1,)*(X.ndim-1))
        Y_coeffs = coeffs.view((-1,) + (1,)*(Y.ndim-1))

        X = X_coeffs * X + (1-X_coeffs) * X[perm]

        if self.mixadd:
            Y = (Y + Y[perm]).clip(0, 1)
        else:
            Y = Y_coeffs * Y + (1 - Y_coeffs) * Y[perm]

        if Z is not None:
            Z_coeffs = coeffs.view((-1,) + (1,)*(Z.ndim-1))
            if self.mixadd:
                Z = (Z + Z[perm]).clip(0, 1)
            else:
                Z = Z_coeffs * Z + (1 - Z_coeffs) * Z[perm]
            return X, Y, Z

        return X, Y

def rotate(x, mask= None, dims= ((-3,-2), (-3,-1), (-2,-1)), p= 1.0):
    """
    Rotate pixels.
    Mask is the target block
    """
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
