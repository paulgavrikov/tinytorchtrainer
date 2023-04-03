import logging

from .lowresdensenet import lowres_densenet121, lowres_densenet161, lowres_densenet169, lowres_densenet201, lowres_densenet264
from .lowresgooglenet import lowres_googlenet
from .lowresinception import lowres_inception_v3
from .lowresmobilenetv2 import lowres_mobilenet_v2
from .lowresresnet import lowres_resnet14, lowres_resnet18, lowres_resnet18_noresidual, lowres_resnet34, \
    lowres_resnet50, lowres_resnet101, lowres_resnet152, lowres_resnet200, lowres_resnet1202, lowres_preact_resnet14, lowres_preact_resnet18, \
    lowres_preact_resnet34, lowres_wide_resnet50_2, lowres_wide_resnet101_2, \
    lowres_resnext50_32x4d, lowres_resnext101_32x8d, lowres_resnet14_dw, lowres_resnet18_dw, lowres_resnet34_dw, lowres_resnet50_dw, \
    lowres_resnet101_dw, lowres_resnet152_dw, lowres_wide_resnet50_2_dw, lowres_wide_resnet101_2_dw
from .lowresvgg import lowres_vgg11_bn, lowres_vgg13_bn, lowres_vgg16_bn, lowres_vgg19_bn, lowres_vgg11, lowres_vgg13, \
    lowres_vgg16, lowres_vgg19
from .lowresresnet9 import lowres_resnet9
from .lowresalexnet import lowres_alexnet
from .lowreslenet import lowres_lenet5
from .convnext import convnext_tiny, convnext_small, convnext_base, convnext_large, convnext_xlarge
from .open_lth.resnet import ResNet
from .cifardensenet import densenet40_12, densenet40_12_bc

from functools import partial

all_classifiers = {
    "lowres_vgg11_bn": lowres_vgg11_bn,
    "lowres_vgg13_bn": lowres_vgg13_bn,
    "lowres_vgg16_bn": lowres_vgg16_bn,
    "lowres_vgg19_bn": lowres_vgg19_bn,
    "lowres_vgg11": lowres_vgg11,
    "lowres_vgg13": lowres_vgg13,
    "lowres_vgg16": lowres_vgg16,
    "lowres_vgg19": lowres_vgg19,
    "lowres_resnet14": lowres_resnet14,
    "lowres_resnet18": lowres_resnet18,
    "lowres_resnet18_noresidual": lowres_resnet18_noresidual,
    "lowres_resnet34": lowres_resnet34,
    "lowres_resnet50": lowres_resnet50,
    "lowres_resnet101": lowres_resnet101,
    "lowres_resnet152": lowres_resnet152,
    "lowres_resnet200": lowres_resnet200,
    "lowres_resnet1202": lowres_resnet1202,
    "lowres_preact_resnet14": lowres_preact_resnet14,
    "lowres_preact_resnet18": lowres_preact_resnet18,
    "lowres_preact_resnet34": lowres_preact_resnet34,
    "lowres_wide_resnet50_2": lowres_wide_resnet50_2,
    "lowres_wide_resnet101_2": lowres_wide_resnet101_2,
    "lowres_resnext50_32x4d": lowres_resnext50_32x4d,
    "lowres_resnext101_32x8d": lowres_resnext101_32x8d,
    "lowres_resnet14_dw": lowres_resnet14_dw,
    "lowres_resnet18_dw": lowres_resnet18_dw,
    "lowres_resnet34_dw": lowres_resnet34_dw,
    "lowres_resnet50_dw": lowres_resnet50_dw,
    "lowres_resnet101_dw": lowres_resnet101_dw,
    "lowres_resnet152_dw": lowres_resnet152_dw,
    "lowres_wide_resnet50_2_dw": lowres_wide_resnet50_2_dw,
    "lowres_wide_resnet101_2_dw": lowres_wide_resnet101_2_dw, 
    "lowres_resnet9": lowres_resnet9,
    "lowres_densenet121": lowres_densenet121,
    "lowres_densenet161": lowres_densenet161,
    "lowres_densenet169": lowres_densenet169,
    "lowres_densenet201": lowres_densenet201,
    "lowres_densenet264": lowres_densenet264,
    "lowres_mobilenet_v2": lowres_mobilenet_v2,
    "lowres_googlenet": lowres_googlenet,
    "lowres_inception_v3": lowres_inception_v3,
    "lowres_alexnet": lowres_alexnet,
    "lowres_lenet5": lowres_lenet5,
    "convnext_tiny": convnext_tiny, 
    "convnext_small": convnext_small, 
    "convnext_base": convnext_base, 
    "convnext_large": convnext_large, 
    "convnext_xlarge": convnext_xlarge,
    "densenet40_12": densenet40_12,
    "densenet40_12_bc": densenet40_12_bc,
}


def torchvision_loader(name, in_channels, num_classes, **kwargs):
    assert in_channels == 3, "TorchVision only supports 3-channel inputs"
    logging.warn(f"Ignoring {kwargs}")
    import torchvision.models
    return torchvision.models.__dict__[name](num_classes=num_classes, pretrained=False)


def timm_loader(name, in_channels, num_classes, **kwargs):
    assert in_channels == 3, "TIMM only supports 3-channel inputs"
    logging.warn(f"Ignoring {kwargs}")
    import timm
    return timm.create_model(model_name=name, num_classes=num_classes, pretrained=False)


def get_model(name):
    if name.startswith("lowres_") or name.startswith("convnext_") or name.startswith("densenet"):
        return all_classifiers.get(name)
    elif name.startswith("torchvision_"):
        return partial(torchvision_loader, name=name.replace("torchvision_", ""))
    elif name.startswith("timm_"):
        return partial(timm_loader, name=name.replace("timm_", ""))
    elif name.startswith("openlth_"):
        return partial(ResNet.get_model_from_name, name=name.replace("openlth_", ""))
