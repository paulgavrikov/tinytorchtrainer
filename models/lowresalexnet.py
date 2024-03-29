import torch
import torch.nn as nn
from typing import Any


__all__ = ['LowResAlexNet', 'lowres_alexnet']


class LowResAlexNet(nn.Module):

    def __init__(self, in_channels: int = 3, num_classes: int = 10, activation_fn=None) -> None:
        super(LowResAlexNet, self).__init__()

        if activation_fn is None:
            activation_fn = nn.ReLU

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
            activation_fn(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            activation_fn(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            activation_fn(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            activation_fn(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            activation_fn(inplace=True),
            nn.AdaptiveMaxPool2d(4),
        )
        #self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.fc = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 4 * 4, 4096),
            activation_fn(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            activation_fn(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        #x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def lowres_alexnet(**kwargs: Any) -> LowResAlexNet:
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
    """
    return LowResAlexNet(**kwargs)
