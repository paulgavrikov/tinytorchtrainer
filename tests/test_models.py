import models
import torch.nn as nn
import torch


def test_model_init():
    x = torch.random(1, 3, 32, 32)
    for model_name in models.all_classifiers:
        print(model_name)
        model = models.get_model(model_name)(in_channels=3, num_classes=10)
        model(x)
        assert model is not None


# def test_model_init_with_activation():

#     activation_fn = nn.ELU

#     for model_name in  models.all_classifiers:
#         print(model_name)
#         model = models.get_model(model_name)(in_channels=3, num_classes=10, activation_fn=activation_fn)
        
#         assert model is not None

#         contains_act = False
#         for module in model.modules():
#             assert type(module) is not nn.ReLU
#             if type(module) == activation_fn:
#                 contains_act = True
                
#         assert contains_act
