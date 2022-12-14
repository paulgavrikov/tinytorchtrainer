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
import pandas as pd
from tqdm import tqdm
from PIL import Image


class CIFAR10Corruptions(Dataset):

    def list_corruptions():
        return ['fog', 'spatter', 'zoom_blur', 'defocus_blur', 'speckle_noise', 'jpeg_compression', 'frost', 'gaussian_noise', 'brightness', 
                              'elastic_transform', 'contrast', 'gaussian_blur', 'snow', 'shot_noise', 'saturate', 'glass_blur', 'motion_blur', 'pixelate', 'impulse_noise']

    def __init__(self, path: str, corruption: str, severity: int, transform=None):
        """
        Args:
            path (string): Path to numpy arrays.
            corruption (string): Corruption to load.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        assert corruption in CIFAR10Corruptions.list_corruptions()

        self.samples = np.load(os.path.join(path, corruption + ".npy"))[(severity - 1) * 10000:severity * 10000]
        self.labels = np.load(os.path.join(path, "labels.npy"))[(severity - 1) * 10000:severity * 10000]
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = Image.fromarray(self.samples[idx])

        if self.transform:
            sample = self.transform(sample)

        return sample, self.labels[idx]


def main(args):

    if args.wandb_project:
        wandb.init(config=vars(args), project=args.wandb_project)

    model = load_trainer(args).model
    model.eval()
    rows = []

    mean = (0.49139968, 0.48215841, 0.44653091)
    std = (0.24703223, 0.24348513, 0.26158784)

    val_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

    with torch.no_grad():
        for corruption in tqdm(CIFAR10Corruptions.list_corruptions()):
            for severity in range(1, 6):
                dataset = CIFAR10Corruptions(args.dataset_dir, corruption, severity, transform=val_transform)

                loader = DataLoader(
                    dataset,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                )

                correct = 0
                total = 0
                for (x, y) in loader:
                    inputs, targets = x.to(args.device), y.to(args.device)

                    logits = model(inputs)
                    correct += (logits.argmax(1) == targets).sum().item()
                    total += len(targets)

                results = {
                    "corruption": corruption, 
                    "severity": severity, 
                    "accuracy": correct / total
                }
                rows.append(results)
                if args.wandb_project:
                    wandb.log({f"{corruption}/{severity}": correct / total})
    
    if args.wandb_project:
        wandb.finish()

    log_file = args.log_file
    if log_file is None:
        log_file = os.path.join(os.path.dirname(args.load_checkpoint), f"cifar10c.csv")

    if os.path.isfile(log_file):
        os.remove(log_file)

    df = pd.DataFrame(rows)
    df.to_csv(log_file, index=False, header=True)

    print(df.pivot(index="severity", columns="corruption", values="accuracy"))

    print(f"mean accuracy: {df.accuracy.mean()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("load_checkpoint", type=str, default=None)
    parser.add_argument("--dataset_dir", type=str, default="/workspace/data/datasets/cifar10c/CIFAR-10-C")
    parser.add_argument("--log_file", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--wandb_project", type=none2str, default=None)
    _args = parser.parse_args()
    main(_args)
    sys.exit(0)
