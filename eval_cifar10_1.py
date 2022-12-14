import torch
import sys
import argparse
import os
import data
from utils import none2str, str2bool
from train import load_trainer
import wandb
import models
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import numpy as np
import torchvision.transforms as transforms
import os


class CIFAR10_1(Dataset):

    def __init__(self, data_root, version="v6", transform=None):
        self.samples = np.load(os.path.join(data_root, f"cifar10.1_{version}_data.npy"))
        self.targets = np.load(os.path.join(data_root, f"cifar10.1_{version}_labels.npy"))

        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x = self.samples[idx]
        if self.transform:
            x = self.transform(x)
        return x, self.targets[idx]


class CIFAR10_1Data:

    def __init__(self, root_dir, version="v6"):
        super().__init__()
        self.root_dir = root_dir
        self.mean = (0.49139968, 0.48215841, 0.44653091)
        self.std = (0.24703223, 0.24348513, 0.26158784)
        self.num_classes = 10
        self.in_channels = 3
        self.version = version
        self.val_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std),
            ]
        )

    def val_dataloader(self, batch_size, num_workers, shuffle=False, drop_last=False, pin_memory=True, **kwargs):
        dataset = CIFAR10_1(data_root=self.root_dir, version=self.version, transform=self.val_transform)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
            drop_last=drop_last,
            pin_memory=pin_memory,
            **kwargs
        )
        return dataloader


def main(args):

    if args.wandb_project:
        wandb.init(config=vars(args), project=args.wandb_project)

    model = load_trainer(args).model
    model.eval()

    correct = 0 
    total = 0
    with torch.no_grad():
        for x, y in CIFAR10_1Data(os.path.join(args.dataset_dir, "cifar10_1"), "v6").val_dataloader(args.batch_size, 0):
            x = x.to(args.device)
            y = y.to(args.device)

            y_pred = model(x)
            correct += (y_pred.argmax(1) == y).sum()
            total += len(y)

    acc = (correct / total).item()

    results = {}
    results["test/acc"] = acc
    print(f"accuracy: {acc}")

    if args.wandb_project:
        wandb.log(results)
    print(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("load_checkpoint", type=str, default=None)
    parser.add_argument("--dataset_dir", type=str, default="/workspace/data/datasets")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=2000)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--wandb_project", type=none2str, default=None)
    _args = parser.parse_args()
    main(_args)
    sys.exit(0)
