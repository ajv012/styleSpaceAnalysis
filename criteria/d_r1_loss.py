import torch
from torch import nn
from torch import autograd
from models.discriminator import conv2d_gradfix

class d_r1_loss(nn.Module):
    def __init__(self, args):
        super(d_r1_loss, self).__init__()
        self.args = args

    def forward(self, real_pred: torch.Tensor, real_img):
        with conv2d_gradfix.no_weight_gradients():
            grad_real, = autograd.grad(
                outputs=real_pred.sum(), inputs=real_img, create_graph=True
            )
        grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()
        r1_loss = self.args.r1 / 2 * grad_penalty * self.args.d_reg_every + 0 * real_pred[0]

        return r1_loss