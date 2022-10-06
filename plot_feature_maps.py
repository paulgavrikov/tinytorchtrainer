import torch
import sys
import argparse
from train import Trainer
import os
import data
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np
from PIL import Image
from utils import LayerHook


def _hide_border(ax):
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())
    ax.imshow(np.zeros((1, 1, 3)))


def plot_filters(layers, cols=32, sparse_thresh=0, show_labels=True, **kwargs):
    rows = len(layers)
    fig, axes = plt.subplots(rows, cols, figsize=(8, rows * 8 / cols), squeeze=False)
    for i, (ax_row, (layer_name, layer_filters)) in enumerate(zip(axes, layers)):
        if show_labels:
            ax_row[0].set_ylabel(layer_name, rotation=0, ha="right", va="center")
        t = abs(layer_filters).max()
        list(map(_hide_border, ax_row))
        for i, (ax, f) in enumerate(zip(ax_row, layer_filters / t)):
            ax.imshow(f.reshape(3, 3), vmin=-1, vmax=1, cmap=LinearSegmentedColormap.from_list("CyanOrange", ["C0", "white", "C1"]))
    fig.align_ylabels(axes)
    return fig


def plot_filters_from_model(model, **kwargs):
    filters_filter = filter(lambda t: type(t[1]) == torch.nn.Conv2d and t[1].kernel_size == (3, 3), model.named_modules())
    iterator = map(lambda t: (t[0], t[1].weight.detach().view(-1, 9).cpu().numpy()), filters_filter)
    fig = plot_filters(list(iterator), **kwargs)
    return fig


def main(args):
    ckpt = torch.load(args.load_checkpoint, map_location="cpu")
    saved_args = argparse.Namespace()

    for k, v in ckpt["args"].items():
        vars(saved_args)[k] = v
    vars(saved_args)["load_checkpoint"] = args.load_checkpoint
    vars(saved_args)["verbose"] = False
    vars(saved_args)["device"] = "cpu"
    dataset = data.get_dataset(saved_args.dataset)(os.path.join(saved_args.dataset_dir, saved_args.dataset))

    vars(saved_args)["model_in_channels"] = dataset.in_channels
    vars(saved_args)["model_num_classes"] = dataset.num_classes

    trainer = Trainer(saved_args)

    img = Image.open(args.sample)
    x = dataset.val_transform(img).unsqueeze(0)

    trainer.model.eval()

    hooks = []
    for layer_name in args.layers.split(","):
        hook = LayerHook()
        hook.register_hook(getattr(trainer.model, layer_name), False)
        hooks.append((layer_name, hook))

    with torch.no_grad():
        y_pred = trainer.model(x)
        pred = torch.nn.Sigmoid()(y_pred)
        print("Predicted label", pred.argmax(1).item(), "with confidence", pred.max().item())

    for layer_name, hook in hooks:
        feature_maps = hook.pull()

        t = abs(feature_maps).max()

        print(f"Layer {layer_name} has {feature_maps.shape[0]} feature maps")

        N = int(np.log2(feature_maps.shape[0]))

        fig = plt.figure(figsize=(N, N))
        grid = ImageGrid(fig, 111,
                 nrows_ncols=(N, N),
                 axes_pad=0.1,
                )

        for ax, im in zip(grid, feature_maps):
            # Iterating over the grid returns the Axes.
            _hide_border(ax)
            ax.imshow(im, vmin=-t, vmax=t, cmap="seismic")
        os.makedirs("out/feature_maps", exist_ok=True)
        fig.savefig(f"out/feature_maps/{layer_name}.png", bbox_inches="tight")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("load_checkpoint", type=str, default=None)
    parser.add_argument("sample", type=str, default=None)
    parser.add_argument("layers", type=str, default=None)
    _args = parser.parse_args()
    main(_args)
    sys.exit(0)
