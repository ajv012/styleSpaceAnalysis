import torch
from torch import nn
from torch import autograd
import math 


class path_reg_loss(nn.Module):
    def __init__(self):
        
        super(path_reg_loss, self).__init__()

    def forward(self, fake_img, latents, mean_path_length, decay=0.01):
        noise = torch.randn_like(fake_img) / math.sqrt(fake_img.shape[2] * fake_img.shape[3])
        grad, = autograd.grad(outputs=(fake_img * noise).sum(), inputs=latents, create_graph=True)
        path_lengths = torch.sqrt(grad.pow(2).sum(2).mean(1))

        path_mean = mean_path_length + decay * (path_lengths.mean() - mean_path_length)

        path_penalty = (path_lengths - path_mean).pow(2).mean()

        return path_penalty, path_mean.detach(), path_lengths