import torch

def relu(z):
    z[z < 0] = 0
    return z

def relu2(z):
    z[z < 0] = 0
    z[z > 0] = z[z > 0] ** 2
    return z

def tanh(z):
    exp_z = torch.exp(z)
    exp_minus_z = torch.exp(-z)
    return (exp_z - exp_minus_z) / (exp_z + exp_minus_z)

def bspline(z):
    bs = z.clone()
    side = torch.logical_or(z <= -1, z > 1)
    left = torch.logical_and(z > -1, z <= 0)
    right = torch.logical_and(z <= 1, z > 0)
    bs[left] = 1 + z[left]
    bs[right] = 1 - z[right]
    bs[side] = 0
    return bs