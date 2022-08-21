from ctypes import LibraryLoader
import torch
import sys
import argparse
from train import Trainer
from autoattack import AutoAttack
from utils import NormalizedModel
import os
import data


def main(args):
    ckpt = torch.load(args.load_checkpoint, map_location="cpu")
    saved_args = argparse.Namespace()

    for k, v in ckpt["args"].items():
        vars(saved_args)[k] = v

    vars(saved_args)["load_checkpoint"] = args.load_checkpoint
    vars(saved_args)["device"] = args.device

    loader_batch = args.n_samples
    if args.n_samples == -1:
        loader_batch = saved_args.batch_size

    dataset = data.get_dataset(saved_args.dataset)(os.path.join(
            saved_args.dataset_dir, saved_args.dataset), loader_batch, saved_args.num_workers)

    vars(saved_args)["model_in_channels"] = dataset.in_channels
    vars(saved_args)["model_num_classes"] = dataset.num_classes

    trainer = Trainer(saved_args)

    all_x = []
    all_y = []
    loader = None

    if args.data_split == "val":
        loader = dataset.val_dataloader()
    elif args.data_split == "train":
        loader = dataset.train_dataloader()

    for x, y in loader:
        all_x.append(x.to(trainer.device))
        all_y.append(y.to(trainer.device))

        if args.n_samples == -1:
            break

    all_x = torch.vstack(all_x)
    all_y = torch.hstack(all_y)

    model = NormalizedModel(trainer.model, dataset.mean, dataset.std).to(trainer.device)

    all_x = all_x * model.std + model.mean  # unnormalize samples for AA

    adversary = AutoAttack(model, norm=args.norm, eps=args.eps)

    _ = adversary.run_standard_evaluation(all_x, all_y)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_checkpoint", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--norm", type=str, default="Linf")
    parser.add_argument("--eps", type=float, default=8/255)
    parser.add_argument("--data_split", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--n_samples", type=int, default=-1)
    _args = parser.parse_args()
    main(_args)
    sys.exit(0)
