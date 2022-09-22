import torch
import sys
import argparse
from train import Trainer
import data
import os


def main(args):
    ckpt = torch.load(args.load_checkpoint, map_location="cpu")
    saved_args = argparse.Namespace()

    for k, v in ckpt["args"].items():
        vars(saved_args)[k] = v

    dataset = data.get_dataset(saved_args.dataset)(os.path.join(
            saved_args.dataset_dir, saved_args.dataset))

    vars(saved_args)["load_checkpoint"] = args.load_checkpoint
    vars(saved_args)["device"] = args.device
    vars(saved_args)["batch_size"] = args.batch_size
    vars(saved_args)["model_in_channels"] = dataset.in_channels
    vars(saved_args)["model_num_classes"] = dataset.num_classes

    trainer = Trainer(saved_args)

    criterion = torch.nn.CrossEntropyLoss()

    metrics = trainer.validate(trainer.model, dataset.val_dataloader(saved_args.batch_size, saved_args.num_workers), criterion, trainer.device)
    print(metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_checkpoint", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=256)
    _args = parser.parse_args()
    main(_args)
    sys.exit(0)
