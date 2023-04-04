dependencies = ["torch", "numpy"]
import models

def custom_loader(model_name, pretrained=False, **kwargs):

    assert not pretrained, "Pretrained models not supported"

    if "in_channels" not in kwargs:
        kwargs["in_channels"] = 3
    if "num_classes" not in kwargs:
        kwargs["num_classes"] = 10

    model = models.get_model(model_name)(**kwargs)
    return model