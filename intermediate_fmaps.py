import models
import torch
import argparse


class LayerHook:

    def __init__(self):
        self.hook_handle = None

    def pull(self):
        if self.hook_handle is not None:
            self.hook_handle.remove()

    def register_hook(self, module, info_text=""):
        if self.hook_handle is not None:
            self.hook_handle.remove()

        def hook(_, inp, out):
            print(info_text)
            print(module)
            print(inp[0].shape, " -> ", out[0].shape)
            print()
        self.hook_handle = module.register_forward_hook(hook)


def main(args):
    model = models.get_model(args.model)(in_channels=3, num_classes=10, activation_fn=None)
    for (name, module) in filter(lambda t: type(t[1]) == torch.nn.Conv2d, model.named_modules()):
        lh = LayerHook()
        lh.register_hook(module, name)
    model(torch.rand(1, 3, 32, 32))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=str)
    _args = parser.parse_args()
    main(_args)