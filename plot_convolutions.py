import torch
import sys
import argparse
from train import Trainer
import os
import data
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from utils import str2bool


def _hide_border(ax):
    ax.spines["bottom"].set_visible(False)
    ax.spines["top"].set_visible(False) 
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
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
    dataset = data.get_dataset(saved_args.dataset)(os.path.join(saved_args.dataset_dir, saved_args.dataset), 1, 0)

    vars(saved_args)["model_in_channels"] = dataset.in_channels
    vars(saved_args)["model_num_classes"] = dataset.num_classes

    trainer = Trainer(saved_args)
    fig = plot_filters_from_model(trainer.model, show_labels=args.show_labels)
    fig.savefig(args.save_file, bbox_inches="tight")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_checkpoint", type=str, default=None)
    parser.add_argument("--save_file", type=str, default="plot_convolutions.png")
    parser.add_argument("--show_labels", type=str2bool, default=True)
    _args = parser.parse_args()
    main(_args)
    sys.exit(0)
