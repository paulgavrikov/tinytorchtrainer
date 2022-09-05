import torch
import sys
import argparse
from train import Trainer
import os
import data


def main(args):
    assert args.batch_size <= args.n_probes

    ckpt = torch.load(args.load_checkpoint, map_location="cpu")
    saved_args = argparse.Namespace()

    for k, v in ckpt["args"].items():
        vars(saved_args)[k] = v

    vars(saved_args)["load_checkpoint"] = args.load_checkpoint
    vars(saved_args)["device"] = args.device

    dataset = data.get_dataset(saved_args.dataset)(os.path.join(
            saved_args.dataset_dir, saved_args.dataset), 1, 0)

    vars(saved_args)["model_in_channels"] = dataset.in_channels
    vars(saved_args)["model_num_classes"] = dataset.num_classes

    trainer = Trainer(saved_args)
    loader = None

    if args.data_split == "val":
        loader = dataset.val_dataloader()
    elif args.data_split == "train":
        loader = dataset.train_dataloader()

    trainer.model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for i, (x, y) in enumerate(loader):

            if args.n_samples != -1 and args.n_samples <= i:
                break

            x = x.to(trainer.device)
            orig_pred = torch.nn.Sigmoid()(trainer.model(x).detach())
            y = orig_pred.argmax().item()


            for b in range(args.n_probes // args.batch_size):
                x = x.repeat(args.batch_size, 1, 1, 1)

                delta = torch.FloatTensor(x.shape).uniform_(-args.eps, args.eps).to(trainer.device)

                y_pred = trainer.model(x + delta)

                correct_batch = (y_pred.argmax(axis=1) == y).sum().item()
                total_batch = len(x)

                print(f"[{i+1}/{args.n_samples if args.n_samples != -1 else len(loader)}] \
                        Batch Accuracy: {correct_batch/total_batch} with confidence {orig_pred.max().item()}")

                correct += correct_batch
                total += len(x)

    print(f"Accuracy: {correct/total}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_checkpoint", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    # parser.add_argument("--norm", type=str, default="Linf")
    parser.add_argument("--eps", type=float, default=8/255)
    parser.add_argument("--data_split", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--n_samples", type=int, default=-1)
    parser.add_argument("--n_probes", type=int, default=16384)
    parser.add_argument("--batch_size", type=int, default=512)
    _args = parser.parse_args()
    main(_args)
    sys.exit(0)
