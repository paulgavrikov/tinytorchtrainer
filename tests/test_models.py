import models
import torch.nn as nn


def test_model_init():
    for model_name in  models.all_classifiers:
        model = models.get_model(model_name)(in_channels=3, num_classes=10)
        assert model is not None

def test_model_init_with_activation():

    activation_fn = nn.GELU

    for model_name in  models.all_classifiers:
        model = models.get_model(model_name)(in_channels=3, num_classes=10, activation_fn=activation_fn)
        
        assert model is not None

        contains_act = False
        for module in model.modules():
            if type(module) == activation_fn:
                contains_act = True
                break
        assert contains_act
