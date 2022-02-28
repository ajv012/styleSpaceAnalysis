import torch
from torch import nn
import torchvision.transforms.functional as F
from torch import autograd
from models.discriminator import conv2d_gradfix

class adv_loss(nn.Module):
    def __init__(self):
        super(adv_loss, self).__init__()

    def forward(self, real_pred: torch.Tensor, fake_pred: torch.Tensor, disc = True):
        if disc:
            return F.softplus(-real_pred).mean() + F.softplus(fake_pred).mean()
        else:
            return F.softplus(-fake_pred).mean()
    
    def d_r1_loss(self, real_pred, real_img):
        with conv2d_gradfix.no_weight_gradients():
            grad_real, = autograd.grad(
                outputs=real_pred.sum(), inputs=real_img, create_graph=True
            )
        grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()

        return grad_penalty