import torch.nn as nn

__all__ = [
    "LowResResNet",
    "lowres_resnet14",
    "lowres_resnet18",
    "lowres_resnet18_noresidual",
    "lowres_resnet34",
    "lowres_resnet50",
    "lowres_resnet101",
    "lowres_resnet152",
    "lowres_resnet200",
    "lowres_resnet1202",
    "lowres_preact_resnet14",
    "lowres_preact_resnet18",
    "lowres_preact_resnet34",
    "lowres_wide_resnet50_2",
    "lowres_wide_resnet101_2",
    "lowres_resnext50_32x4d",
    "lowres_resnext101_32x8d"
]


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        activation_fn,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
        skip_residual=False
    ):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = activation_fn(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
        self.skip_residual = skip_residual

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if not self.skip_residual:
            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity
        out = self.relu(out)

        return out


class DepthwiseBasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        activation_fn,
        stride=1,
        downsample=None,
        groups=-1,
        base_width=64,
        dilation=1,
        norm_layer=None,
        skip_residual=False
    ):
        super(DepthwiseBasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != -1 or base_width != 64:
            raise ValueError("DepthwiseBasicBlock only supports groups=-1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.dwconv1 = conv3x3(inplanes, inplanes, stride=stride, groups=inplanes)
        self.bn1 = norm_layer(inplanes)
        self.pwconv1 = conv1x1(inplanes, planes)
        self.relu = activation_fn(inplace=True)
        self.dwconv2 = conv3x3(planes, planes, groups=planes)
        self.bn2 = norm_layer(planes)
        self.pwconv2 = conv1x1(planes, planes)
        self.bn3 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
        self.skip_residual = skip_residual

    def forward(self, x):
        identity = x

        out = self.dwconv1(x)
        out = self.bn1(out)
        out = self.pwconv1(out)
        out = self.relu(out)

        out = self.dwconv2(out)
        out = self.bn2(out)
        out = self.pwconv2(out)
        out = self.bn3(out)

        if not self.skip_residual:
            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        inplanes,
        planes,
        activation_fn,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
        skip_residual=False
    ):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != -1:
            width = int(planes * (base_width / 64.0)) * groups
        else:
            width = int(planes * (base_width / 64.0)) 
            groups = width
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = activation_fn(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.skip_residual = skip_residual

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if not self.skip_residual:
            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity
        out = self.relu(out)

        return out


class PreactBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, activation_fn, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, skip_residual=False):
        super(PreactBasicBlock, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')

        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")

        # Both self.conv1 and self.downsample layers downsample the input when stride != 1

        self.bn1 = norm_layer(inplanes)
        self.relu1 = activation_fn(inplace=True)
        self.conv1 = conv3x3(inplanes, planes, stride)

        self.bn2 = norm_layer(planes)
        self.relu2 = activation_fn(inplace=True)
        self.conv2 = conv3x3(planes, planes)

        self.downsample = downsample
        self.stride = stride
        self.skip_residual = skip_residual

    def forward(self, x):
        identity = x

        out = self.bn1(x)
        out = self.relu1(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv2(out)

        if not self.skip_residual:
            if self.downsample is not None:
                identity = self.downsample(x)

        out += identity

        return out


class LowResResNet(nn.Module):
    def __init__(
        self,
        block,
        layers,
        in_channels=3,
        num_classes=10,
        zero_init_residual=False,
        groups=1,
        width_per_group=64,
        replace_stride_with_dilation=None,
        norm_layer=None,
        skip_residual=False,
        activation_fn=None
    ):
        super(LowResResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        if activation_fn is None:
            activation_fn = nn.ReLU

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                "or a 3-element tuple, got {}".format(replace_stride_with_dilation)
            )
        self.groups = groups
        self.base_width = width_per_group

        # CIFAR10: kernel_size 7 -> 3, stride 2 -> 1, padding 3->1
        self.conv1 = nn.Conv2d(in_channels, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        # END

        self.bn1 = norm_layer(self.inplanes)
        self.relu = activation_fn(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], activation_fn, skip_residual=skip_residual)
        self.layer2 = self._make_layer(
            block, 128, layers[1], activation_fn, stride=2, dilate=replace_stride_with_dilation[0], skip_residual=skip_residual
        )
        self.layer3 = self._make_layer(
            block, 256, layers[2], activation_fn, stride=2, dilate=replace_stride_with_dilation[1], skip_residual=skip_residual
        )
        
        if len(layers) >= 4:
            self.layer4 = self._make_layer(
                block, 512, layers[3], activation_fn, stride=2, dilate=replace_stride_with_dilation[2], skip_residual=skip_residual
            )
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(512 * block.expansion, num_classes)
        else:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(256 * block.expansion, num_classes)


    def _make_layer(self, block, planes, blocks, activation_fn, stride=1, dilate=False, skip_residual=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                activation_fn,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
                skip_residual=skip_residual
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    activation_fn,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                    skip_residual=skip_residual
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        if hasattr(self, "layer4"):
            x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)

        return x


def _resnet(block, layers, **kwargs):
    return LowResResNet(block, layers, **kwargs)


def lowres_resnet14(**kwargs):
    """Constructs a ResNet-14 model."""
    return _resnet(BasicBlock, [2, 2, 2], **kwargs)

def lowres_resnet1202(**kwargs):
    """Constructs a ResNet-1202 model."""
    return _resnet(BasicBlock, [200, 200, 200], **kwargs)

def lowres_resnet18_noresidual(**kwargs):
    """Constructs a ResNet-18 model without residual connections."""
    return _resnet(BasicBlock, [2, 2, 2, 2], skip_residual=True, **kwargs)


def lowres_resnet18(**kwargs):
    """Constructs a ResNet-18 model."""
    return _resnet(BasicBlock, [2, 2, 2, 2], **kwargs)


def lowres_resnet34(**kwargs):
    """Constructs a ResNet-34 model."""
    return _resnet(BasicBlock, [3, 4, 6, 3], **kwargs)


def lowres_resnet50(**kwargs):
    """Constructs a ResNet-50 model."""
    return _resnet(Bottleneck, [3, 4, 6, 3], **kwargs)


def lowres_resnet101(**kwargs):
    """Constructs a ResNet-101 model."""
    return _resnet(Bottleneck, [3, 4, 23, 3], **kwargs)


def lowres_resnet152(**kwargs):
    """Constructs a ResNet-152 model."""
    return _resnet(Bottleneck, [3, 8, 36, 3], **kwargs)

def lowres_resnet200(**kwargs):
    """Constructs a ResNet-200 model."""
    return _resnet(Bottleneck, [3, 24, 36, 3], **kwargs)

# ResNeXt


def lowres_resnext50_32x4d(**kwargs):
    r"""ResNeXt-50 32x4d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_.
    """
    kwargs["groups"] = 32
    kwargs["width_per_group"] = 4
    return _resnet(Bottleneck, [3, 4, 6, 3], **kwargs)


def lowres_resnext101_32x8d(**kwargs):
    r"""ResNeXt-101 32x8d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_.
    """
    kwargs["groups"] = 32
    kwargs["width_per_group"] = 8
    return _resnet(Bottleneck, [3, 4, 23, 3], **kwargs)


# Wide


def lowres_wide_resnet50_2(**kwargs):
    r"""Wide ResNet-50-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_.
    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.
    """
    kwargs["width_per_group"] = 64 * 2
    return _resnet(Bottleneck, [3, 4, 6, 3], **kwargs)


def lowres_wide_resnet101_2(**kwargs):
    r"""Wide ResNet-101-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_.
    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.
    """
    kwargs["width_per_group"] = 64 * 2
    return _resnet(Bottleneck, [3, 4, 23, 3], **kwargs)

# Pre-Act

def lowres_preact_resnet14(**kwargs):
    """Constructs a Pre-Act ResNet-14 model."""
    return _resnet(PreactBasicBlock, [2, 2, 2], **kwargs)


def lowres_preact_resnet18(**kwargs):
    """Constructs a Pre-Act ResNet-18 model."""
    return _resnet(PreactBasicBlock, [2, 2, 2, 2], **kwargs)


def lowres_preact_resnet34(**kwargs):
    """Constructs a Pre-Act ResNet-34 model."""
    return _resnet(PreactBasicBlock, [3, 4, 6, 3], **kwargs)

# DepthWise

def lowres_resnet14_dw(**kwargs):
    """Constructs a DepthWise ResNet-14 model."""
    kwargs["groups"] = -1
    return _resnet(DepthwiseBasicBlock, [2, 2, 2], **kwargs)


def lowres_resnet18_dw(**kwargs):
    """Constructs a DepthWise ResNet-18 model."""
    kwargs["groups"] = -1
    return _resnet(DepthwiseBasicBlock, [2, 2, 2, 2], **kwargs)


def lowres_resnet34_dw(**kwargs):
    """Constructs a DepthWise ResNet-34 model."""
    kwargs["groups"] = -1
    return _resnet(DepthwiseBasicBlock, [3, 4, 6, 3], **kwargs)


def lowres_resnet50_dw(**kwargs):
    """Constructs a DepthWise ResNet-50 model."""
    kwargs["groups"] = -1
    return _resnet(Bottleneck, [3, 4, 6, 3], **kwargs)


def lowres_resnet101_dw(**kwargs):
    """Constructs a DepthWise ResNet-101 model."""
    kwargs["groups"] = -1
    return _resnet(Bottleneck, [3, 4, 23, 3], **kwargs)


def lowres_resnet152_dw(**kwargs):
    """Constructs a DepthWise ResNet-152 model."""
    kwargs["groups"] = -1
    return _resnet(Bottleneck, [3, 8, 36, 3], **kwargs)


def lowres_wide_resnet50_2_dw(**kwargs):
    r"""DepthWise Wide ResNet-50-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_.
    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.
    """
    kwargs["groups"] = -1
    kwargs["width_per_group"] = 64 * 2
    return _resnet(Bottleneck, [3, 4, 6, 3], **kwargs)


def lowres_wide_resnet101_2_dw(**kwargs):
    r"""DepthWise Wide ResNet-101-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_.
    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.
    """
    kwargs["groups"] = -1
    kwargs["width_per_group"] = 64 * 2
    return _resnet(Bottleneck, [3, 4, 23, 3], **kwargs)