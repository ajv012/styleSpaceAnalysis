import torch
from torch import nn
import torchvision.transforms.functional as F

class adv_loss(nn.Module):
    def __init__(self):
        super(adv_loss, self).__init__()

    def forward(self, real_pred: torch.Tensor, fake_pred: torch.Tensor):
        real_loss = F.softplus(-real_pred)
        fake_loss = F.softplus(fake_pred)

        return real_loss.mean() + fake_loss.mean()