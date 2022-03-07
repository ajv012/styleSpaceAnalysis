import torch
from torch import nn
from torchvision import models


class clf_loss(nn.Module):
    def __init__(self, classifier, args):
        
        super(clf_loss, self).__init__()

        self.model_ft = classifier
        self.loss_func = nn.KLDivLoss().to(args.device)
        # TODO: change to classifier output size argument
        self.resize_input = torch.nn.AdaptiveAvgPool2d((512, 512))
        

    def forward(self, x: torch.Tensor, y_hat: torch.Tensor):
        return self.loss_func(self.model_ft(self.resize_input(x)), self.model_ft(self.resize_input(y_hat)))
