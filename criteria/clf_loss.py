import torch
from torch import nn
from torchvision import models


class clf_loss(nn.Module):
    def __init__(self, args, num_classes = 2, network = "resnet", path_to_weights = "./checkpoint/checkpoint_2.pt"):
        
        super(clf_loss, self).__init__()

        self.model_ft = models.resnet18(pretrained=False)
        self.num_ftrs = self.model_ft.fc.in_features
        self.model_ft.fc = nn.Linear(self.num_ftrs, num_classes)

        checkpoint = torch.load(path_to_weights)
        self.model_ft.load_state_dict(checkpoint['model_state_dict'])

        self.model_ft.eval()

        self.loss_func = nn.KLDivLoss().to(args.device)
        

    def forward(self, x: torch.Tensor, y_hat: torch.Tensor):
        return self.loss_func(self.model_ft(x), self.model_ft(y_hat))