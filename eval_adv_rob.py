import torch
import sys
import argparse
from train import Trainer
import shutil
from autoattack import AutoAttack
from utils import NormalizedModel



def main(args):
    ckpt = torch.load(args.load_checkpoint, map_location="cpu")
    saved_args = argparse.Namespace()

    for k, v in ckpt["args"].items():
        vars(saved_args)[k] = v

    vars(saved_args)["load_checkpoint"] = args.load_checkpoint
    vars(saved_args)["device"] = args.device
    
    trainer = Trainer(saved_args, ".temp")
    
    all_x = []
    all_y = []
    for x, y in trainer.dataset.val_dataloader(): 
        all_x.append(x)
        all_y.append(y)

    all_x = torch.vstack(all_x).to(trainer.device)
    all_y = torch.hstack(all_y).to(trainer.device)

    model = NormalizedModel(trainer.model, trainer.dataset.mean, trainer.dataset.std).to(trainer.device)
    
    all_x = all_x * model.std + model.mean  # unnormalize samples for AA

    adversary = AutoAttack(model, norm=args.norm, eps=args.eps)

    _ = adversary.run_standard_evaluation(all_x, all_y)

    shutil.rmtree(".temp")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_checkpoint", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--norm", type=str, default="Linf")
    parser.add_argument("--eps", type=float, default=8/255)
    _args = parser.parse_args()
    main(_args)
    sys.exit(0)
