import torch
import sys
import argparse
from train import Trainer
import shutil


def main(args):
    ckpt = torch.load(args.load_checkpoint, map_location="cpu")
    saved_args = argparse.Namespace()

    for k, v in ckpt["args"].items():
        vars(saved_args)[k] = v

    vars(saved_args)["load_checkpoint"] = args.load_checkpoint
    vars(saved_args)["device"] = args.device
    vars(saved_args)["batch_size"] = args.batch_size
    
    trainer = Trainer(saved_args, ".temp")
    metrics = trainer.validate(trainer.model, trainer.dataset.val_dataloader(), trainer.criterion, trainer.device)
    print(metrics)
    shutil.rmtree(".temp")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_checkpoint", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=256)
    _args = parser.parse_args()
    main(_args)
    sys.exit(0)
