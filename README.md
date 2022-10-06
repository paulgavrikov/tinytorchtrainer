# tinytorchtrainer
[![pytest](https://github.com/paulgavrikov/tinytorchtrainer/actions/workflows/pytest.yml/badge.svg)](https://github.com/paulgavrikov/tinytorchtrainer/actions/workflows/pytest.yml)

Successor of pytorch-pretrained-cnns minus pytorch lightning


Scripts:

- `train.py`: Training with various parameters
- `eval.py`: Runs evaluation of a given checkpoint with validation dataset
- `eval_adv_rob.py`: Runs evaluation of a given checkpoint with validation dataset under adversarial attacks of AutoAttack
- `plot_convolutions.py`: Plots the 3x3 conv filters of a given checkpoint
- `plot_feature_maps.py`: Plots the feature maps of selected layers of a given checkpoints for a given input sample
- `init_data.py`: Prepares the data independent of training (usefull before sweeps)
- `ckpt_info.py`: Dumps info stored in a checkpoint
