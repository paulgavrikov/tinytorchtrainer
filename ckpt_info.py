import torch
import sys
import argparse
from pprint import pprint

def main(args):
    ckpt = torch.load(args.load_checkpoint, map_location="cpu")
    print("METRICS:")
    pprint(ckpt["metrics"])
    print("ARGS:")
    pprint(ckpt["args"])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("load_checkpoint", type=str, default=None)
    _args = parser.parse_args()
    main(_args)
    sys.exit(0)
