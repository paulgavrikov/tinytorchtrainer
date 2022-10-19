# Modified from https://raw.githubusercontent.com/facebookresearch/open_lth/main/models/cifar_resnet.py
# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from re import S
from statistics import mode
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

class ResNet(nn.Module):
    """A residual neural network as originally designed for CIFAR-10."""

    class Block(nn.Module):
        """A ResNet block."""

        @staticmethod
        def make_conv(f_in: int, f_out: int, stride: int, depthwise: bool=False, spatial: bool=False):
            if not spatial and not depthwise:
                return nn.Conv2d(f_in, f_out, kernel_size=3, stride=stride, padding=1, bias=False)
            elif not spatial and depthwise:
                return nn.Sequential(
                    nn.Conv2d(f_in, f_in, kernel_size=3, stride=stride, padding=1, bias=False, groups=f_in),
                    nn.BatchNorm2d(f_in),
                    nn.Conv2d(f_in, f_out, kernel_size=1, bias=False),
                )
            elif spatial and depthwise:
                return nn.Sequential(
                    nn.Conv2d(f_in, f_in, kernel_size=(1, 3), stride=(1, stride), padding=(0, 1), bias=False, groups=f_in),
                    nn.Conv2d(f_in, f_in, kernel_size=(3, 1), stride=(stride, 1), padding=(1, 0), bias=False, groups=f_in),
                    nn.BatchNorm2d(f_in),
                    nn.Conv2d(f_in, f_out, kernel_size=1, bias=False),
                )
            elif spatial and not depthwise:
                return nn.Sequential(
                    nn.Conv2d(f_in, f_out, kernel_size=(1, 3), stride=(1, stride), padding=(0, 1), bias=False),
                    nn.Conv2d(f_out, f_out, kernel_size=(3, 1), stride=(stride, 1), padding=(1, 0), bias=False),
                )
            else:
                raise ValueError("Invalid combination of depthwise and spatial.")


        def __init__(self, f_in: int, f_out: int, downsample=False, depthwise=False, spatial=False):
            super().__init__()

            stride = 2 if downsample else 1
            
            self.conv1 = ResNet.Block.make_conv(f_in, f_out, stride, depthwise, spatial)
            self.bn1 = nn.BatchNorm2d(f_out)
            self.conv2 = ResNet.Block.make_conv(f_out, f_out, 1, depthwise, spatial)
            self.bn2 = nn.BatchNorm2d(f_out)

            self.activation = nn.ReLU(inplace=True)

            # No parameters for shortcut connections.
            if downsample or f_in != f_out:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(f_in, f_out, kernel_size=1, stride=2, bias=False),
                    nn.BatchNorm2d(f_out)
                )
            else:
                self.shortcut = nn.Sequential()

        def forward(self, x):
            out = x
            out = self.conv1(out)
            out = self.bn1(out)
            out = self.activation(out)
            
            out = self.conv2(out)
            out = self.bn2(out)

            out += self.shortcut(x)
            return self.activation(out)

    def __init__(self, plan, spatial=False, depthwise=False, in_channels=3, num_classes=10, **kwargs):
        super().__init__()

        # Initial convolution.
        current_filters = plan[0][0]

        self.conv = make_conv(in_channels, current_filters, stride=, depthwise=depthwise, spatial=spatial)
        self.bn = nn.BatchNorm2d(current_filters)
        self.activation = nn.ReLU(inplace=True)

        # The subsequent blocks of the ResNet.
        blocks = []
        for segment_index, (filters, num_blocks) in enumerate(plan):
            for block_index in range(num_blocks):
                downsample = segment_index > 0 and block_index == 0
                blocks.append(ResNet.Block(current_filters, filters, downsample, spatial=spatial, depthwise=depthwise))
                current_filters = filters

        self.blocks = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)

        # Final fc layer. Size = number of filters in last segment.
        self.fc = nn.Linear(plan[-1][0], num_classes)

    def forward(self, x):
        out = x
        out = self.conv(out)
        out = self.bn(out)
        out = self.activation(out)
        out = self.blocks(out)
        out = self.pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

    @property
    def output_layer_names(self):
        return ['fc.weight', 'fc.bias']

    @staticmethod
    def get_model_from_name(name, **kwargs):
        """The naming scheme for a ResNet is 'cifar_resnet_N[_W]'.

        The ResNet is structured as an initial convolutional layer followed by three "segments"
        and a linear output layer. Each segment consists of D blocks. Each block is two
        convolutional layers surrounded by a residual connection. Each layer in the first segment
        has W filters, each layer in the second segment has 32W filters, and each layer in the
        third segment has 64W filters.

        The name of a ResNet is 'cifar_resnet_N[_W]', where W is as described above.
        N is the total number of layers in the network: 2 + 6D.
        The default value of W is 16 if it isn't provided.

        For example, ResNet-20 has 20 layers. Exclusing the first convolutional layer and the final
        linear layer, there are 18 convolutional layers in the blocks. That means there are nine
        blocks, meaning there are three blocks per segment. Hence, D = 3.
        The name of the network would be 'cifar_resnet_20' or 'cifar_resnet_20_16'.
        """

        depthwise = False
        skip_spatial_convs = False
        spatial = False
        if "_dw" in name:
            depthwise = True
            name = name.replace("_dw", "")

        if "_di" in name:
            depthwise = True
            skip_spatial_convs = True
            name = name.replace("_di", "")

        spatial = False
        if "_sp" in name:
            spatial = True
            name = name.replace("_sp", "")

        name = name.split('_')

        W = 16 if len(name) == 2 else int(name[2])
        D = int(name[1])
        if (D - 2) % 3 != 0:
            raise ValueError('Invalid ResNet depth: {}'.format(D))
        D = (D - 2) // 6
        plan = [(W, D), (2*W, D), (4*W, D)]

        resnet = ResNet(plan, spatial=spatial, depthwise=depthwise, **kwargs)

        if skip_spatial_convs:
            replace_conv2d(resnet, nn.Sequential, filter_cb=lambda m: m.kernel_size != (1, 1) and m.stride == (1, 1))
            replace_conv2d(resnet, partial(nn.Upsample, scale_factor=0.5, mode="bilinear"), filter_cb=lambda m: m.kernel_size != (1, 1) and m.stride == (2, 2))
        return resnet

def replace_conv2d(module, new_cls, filter_cb=None, **kwargs):
    for name, child in module.named_children():
   
        if type(child) == nn.Conv2d and (not filter_cb or (filter_cb and filter_cb(child))) and name != "conv":
            new_module = new_cls()
            setattr(module, name, new_module)

    for child in module.children():
        replace_conv2d(child, new_cls, filter_cb, **kwargs)