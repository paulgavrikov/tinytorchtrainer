import torch
import torch.nn as nn
from typing import Any


__all__ = ['LowResLeNet5', 'lowres_lenet5']


class LowResLeNet5(nn.Module):

    def __init__(self, in_channels: int = 3, num_classes: int = 10, activation_fn=None) -> None:
        super(LowResLeNet5, self).__init__()

        if activation_fn is None:
            activation_fn = nn.Sigmoid

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 6, kernel_size=5, padding=2),
            activation_fn(),
            nn.AvgPool2d(kernel_size=(2, 2), stride=2),
            nn.Conv2d(6, 16, kernel_size=5),
            activation_fn(),
            nn.AdaptiveAvgPool2d(5),
        )
        self.fc = nn.Sequential(
            nn.LazyLinear(120),
            activation_fn(),
            nn.Linear(120, 84),
            activation_fn(),
            nn.Linear(84, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def lowres_lenet5(**kwargs: Any) -> LowResLeNet5:
    return LowResLeNet5(**kwargs)
