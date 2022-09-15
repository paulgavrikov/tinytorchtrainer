import argparse
import data
import os
import sys


def main(args):
    dataset = data.get_dataset(args.dataset)(os.path.join(args.dataset_dir, args.dataset), 1, 0)
    _ = dataset.val_dataloader()
    _ = dataset.train_dataloader()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--dataset_dir", type=str, default=None)
    _args = parser.parse_args()
    main(_args)
    sys.exit(0)
