from torch import nn
from torchvision import models
import torch

class Classifier(nn.Module):
    def __init__(self, args, num_classes = 2, network = "resnet", path_to_weights = "./checkpoint/checkpoint_2.pt"):

        self.model_ft = models.resnet18(pretrained=False)
        self.num_ftrs = self.model_ft.fc.in_features
        self.model_ft.fc = nn.Linear(self.num_ftrs, num_classes)

        checkpoint = torch.load(path_to_weights)
        self.model_ft.load_state_dict(checkpoint['model_state_dict'])

        self.model_ft.eval()

    def forward(self, x: torch.Tensor):
        return self.model_ft(x)