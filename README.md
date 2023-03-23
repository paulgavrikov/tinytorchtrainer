# tinytorchtrainer
[![pytest](https://github.com/paulgavrikov/tinytorchtrainer/actions/workflows/pytest.yml/badge.svg)](https://github.com/paulgavrikov/tinytorchtrainer/actions/workflows/pytest.yml)
[![CC BY-SA 4.0][cc-by-sa-shield]][cc-by-sa]

[cc-by-sa]: http://creativecommons.org/licenses/by-sa/4.0/
[cc-by-sa-image]: https://licensebuttons.net/l/by-sa/4.0/88x31.png
[cc-by-sa-shield]: https://img.shields.io/badge/License-CC%20BY--SA%204.0-lightgrey.svg

Successor of pytorch-pretrained-cnns minus pytorch lightning


Scripts:

- `train.py`: Training with various parameters
- `eval.py`: Runs evaluation of a given checkpoint with validation dataset
- `eval_adv_rob.py`: Runs evaluation of a given checkpoint with validation dataset under adversarial attacks of AutoAttack
- `plot_convolutions.py`: Plots the 3x3 conv filters of a given checkpoint
- `plot_feature_maps.py`: Plots the feature maps of selected layers of a given checkpoints for a given input sample
- `init_data.py`: Prepares the data independent of training (usefull before sweeps)
- `ckpt_info.py`: Dumps info stored in a checkpoint
- `to_onnx.py`: Exports checkpoints to ONNX

# Credits

Code partially taken from https://github.com/huyvnphan/PyTorch_CIFAR10, https://github.com/1M50RRY/resnet18-preact, https://github.com/pytorch/vision/, https://github.com/andreasveit/densenet-pytorch/.
