import torch
import torch.nn as nn
# Parameter
from torch.nn.parameter import Parameter
# Tensor
from torchvision.transforms.functional import *

# CNNMnist
class CNNMnist(nn.Module):
    # (16, 1, 28, 28)
    def __init__(self, args):
        super(CNNMnist, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.mp1 = nn.MaxPool2d(2, 2)
        self.mp2 = nn.MaxPool2d(2, 2)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.fc1 = nn.Linear(64 * 7 * 7, args.num_classes)  # Adjusted based on the output from the last MaxPool2d

    def forward(self, x):
        x = self.conv1(x)
        x = self.mp1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.mp2(x)
        x = self.relu2(x)
        x = x.view(-1, 64 * 7 * 7)
        x = self.fc1(x)
        return x
# Amplification
class PowerAmplification(nn.Module):
        def __init__(
            self, in_features: int, alpha: float = None, device=None, dtype=None
        ) -> None:
            super(PowerAmplification, self).__init__()
            factory_kwargs = {"device": device, "dtype": dtype}
            self.in_features = in_features
            if alpha is not None:
                self.alpha = Parameter(torch.tensor([alpha], **factory_kwargs))
            else:
                self.alpha = Parameter(torch.rand(1, **factory_kwargs))

        def forward(self, input: Tensor) -> Tensor:
            alpha = self.alpha.expand(self.in_features)
            return torch.pow(input, alpha)

# Inversion
class Inversion(nn.Module):
        def __init__(self, args):
            super(Inversion, self).__init__()
            self.in_channels = args.num_classes
            self.deconv1 = nn.ConvTranspose2d(self.in_channels, 64, 7, 1)
            self.deconv2 = nn.ConvTranspose2d(64, 32, 4, 2, 1)
            self.deconv3 = nn.ConvTranspose2d(32, 1, 4, 2, 1)
            self.relu1 = nn.ReLU()
            self.relu2 = nn.ReLU()
            self.sigmoid = nn.Sigmoid()
        def forward(self, x):
            x = x.view(-1, self.in_channels, 1, 1)
            x = self.deconv1(x)
            x = self.relu1(x)
            x = self.deconv2(x)
            x = self.relu2(x)
            x = self.deconv3(x)
            x = self.sigmoid(x)
            return x