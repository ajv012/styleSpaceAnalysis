from torch import nn
from torchvision import models
import torch

class Classifier(nn.Module):
    def __init__(self, args, num_classes = 2, network = "resnet"):
        
        super(Classifier, self).__init__()

        self.model_ft = models.resnet18(pretrained=True)
        self.num_ftrs = self.model_ft.fc.in_features
        self.model_ft.fc = nn.Linear(self.num_ftrs, num_classes)

        print("Initialized classifier")

    def forward(self, x: torch.Tensor):
        return self.model_ft(x)