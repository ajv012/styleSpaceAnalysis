import torch
from torch import nn
import torchvision.transforms.functional as F

class adv_loss(nn.Module):
    def __init__(self):
        super(adv_loss, self).__init__()

    def forward(self, real_pred: torch.Tensor, fake_pred: torch.Tensor, disc = True):
        if disc:
            return F.softplus(-real_pred).mean() + F.softplus(fake_pred).mean()
        else:
            return F.softplus(-fake_pred).mean()