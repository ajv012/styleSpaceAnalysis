import torch
from torch import nn
from torchvision import models


class clf_loss(nn.Module):
    def __init__(self, classifier, args):
        
        super(clf_loss, self).__init__()

        self.model_ft = classifier
        self.loss_func = nn.KLDivLoss(log_target=True).to(args.device)
        # TODO: change to classifier output size argument
        self.resize_input = torch.nn.AdaptiveAvgPool2d((512, 512))
        

    def forward(self, x: torch.Tensor, y_hat: torch.Tensor):
        log_prob_x = torch.nn.functional.log_softmax(self.model_ft(self.resize_input(x)), dim=-1)
        log_prob_y_hat = torch.nn.functional.log_softmax(self.model_ft(self.resize_input(y_hat)), dim=-1)
        return self.loss_func(log_prob_x, log_prob_y_hat)
