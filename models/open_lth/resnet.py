# Modified from https://raw.githubusercontent.com/facebookresearch/open_lth/main/models/cifar_resnet.py
# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn
from functools import partial
import re

class ResNet(nn.Module):
    """A residual neural network as originally designed for CIFAR-10."""

    class Block(nn.Module):
        """A ResNet block."""

        @staticmethod
        def make_conv(f_in: int, f_out: int, stride: int, conv_type: str, kernel_size:int=3, force_no_padding: bool=False, padding_mode: str="zeros"):
            assert padding_mode in ["zeros", "reflect", "replicate", "circular"], f"Invalid padding mode {padding_mode}."

            padding = (kernel_size // 2)
            if force_no_padding:
                padding = 0
            
            if not conv_type:
                return nn.Conv2d(f_in, f_out, kernel_size=kernel_size, stride=stride, padding=padding, padding_mode=padding_mode, bias=False)
            elif conv_type == "depthwise_separable":
                return nn.Sequential(
                    nn.Conv2d(f_in, f_in, kernel_size=kernel_size, stride=stride, padding=padding, padding_mode=padding_mode, bias=False, groups=f_in),
                    nn.Conv2d(f_in, f_out, kernel_size=1, bias=False),
                )
            else:
                raise ValueError(f"Invalid conv_type {conv_type}.")


        def __init__(self, f_in: int, f_out: int, activation_fn, downsample=False, **conv_args):
            super().__init__()

            stride = 2 if downsample else 1
            self.downsample = downsample
            
            self.conv1 = ResNet.Block.make_conv(f_in, f_out, stride, **conv_args)
            self.bn1 = nn.BatchNorm2d(f_out)
            self.conv2 = ResNet.Block.make_conv(f_out, f_out, 1, **conv_args)
            self.bn2 = nn.BatchNorm2d(f_out)

            self.force_no_padding = conv_args.get("force_no_padding", False)

            self.activation = activation_fn()

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

            if self.force_no_padding:
              if self.downsample:
                out = nn.functional.pad(out, (2, 1, 2, 1))
              else:
                out = nn.functional.pad(out, (2, 2, 2, 2))

            out += self.shortcut(x)
            return self.activation(out)

    def __init__(self, plan, in_channels=3, num_classes=10, activation_fn=None, **conv_args):
        super().__init__()

        if activation_fn is None:
            activation_fn = partial(nn.ReLU, inplace=True)

        # Initial convolution.
        current_filters = plan[0][0]

        self.conv = ResNet.Block.make_conv(in_channels, current_filters, stride=1, **conv_args)
        self.bn = nn.BatchNorm2d(current_filters)
        self.activation = activation_fn()

        # The subsequent blocks of the ResNet.
        blocks = []
        for segment_index, (filters, num_blocks) in enumerate(plan):
            for block_index in range(num_blocks):
                downsample = segment_index > 0 and block_index == 0
                blocks.append(ResNet.Block(current_filters, filters, activation_fn=activation_fn, downsample=downsample, **conv_args))
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
        return ["fc.weight", "fc.bias"]

    @staticmethod
    def get_model_from_name(name, **kwargs):
        use_1x1_stem = False
        kernel_size = 3
        conv_type = None
        padding_mode = "zeros"
        force_no_padding = False

        if "_dw" in name:
            conv_type = "depthwise_separable"
            name = name.replace("_dw", "")
          
        if "_1x1stem" in name:
            use_1x1_stem = True
            name = name.replace("_1x1stem", "")
            
        exp_match = re.search(r"_k[0-9]+", name)
        if exp_match:
            kernel_size = int(exp_match.group(0)[2:])
            name = name.replace(exp_match.group(0), "")

        exp_match = re.search(r"_padding(.*)", name)
        if exp_match:
            padding_mode = exp_match.group(1)
            name = name.replace(exp_match.group(0), "")

        if "_nopad" in name:
            force_no_padding = True
            name = name.replace("_nopad", "")

        name = name.split("_")
        
        assert len(name) <= 3, f"Extra args not understood {name}"

        W = 16 if len(name) == 2 else int(name[2])
        D = int(name[1])
        if (D - 2) % 3 != 0:
            raise ValueError("Invalid ResNet depth: {}".format(D))
        D = (D - 2) // 6
        plan = [(W, D), (2*W, D), (4*W, D)]

        resnet = ResNet(plan, conv_type=conv_type, kernel_size=kernel_size, padding_mode=padding_mode, force_no_padding=force_no_padding, **kwargs)

        # must be first, before any other replace!
        if use_1x1_stem:
            resnet.conv = nn.Conv2d(in_channels=resnet.conv.in_channels, out_channels=resnet.conv.out_channels, kernel_size=1, bias=False)

        return resnet
