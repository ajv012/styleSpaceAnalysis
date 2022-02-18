import torch
from torchvision import models
import torch.nn as nn 


class clf(torch.nn.Module):
    r"""
    A simple encoder and fully connected layer for classification
    """

    def __init__(self, num_classes):
        super(clf, self).__init__()
        self.model_ft = models.resnet18(pretrained=True)
        self.num_ftrs = self.model_ft.fc.in_features
        self.model_ft.fc = nn.Linear(self.num_ftrs, num_classes)

    def forward(self, x):
        x = self.model_ft(x)
        return x