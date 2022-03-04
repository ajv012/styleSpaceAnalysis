import torch
from torch import nn
from torchvision import models


class clf_loss(nn.Module):
    def __init__(self, classifier, args):
        
        super(clf_loss, self).__init__()

        self.model_ft = classifier
        self.loss_func = nn.KLDivLoss().to(args.device)
        

    def forward(self, x: torch.Tensor, y_hat: torch.Tensor):
        return self.loss_func(self.model_ft(x), self.model_ft(y_hat))