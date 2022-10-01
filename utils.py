from git import RemoteProgress
import torch
from tqdm import tqdm
import argparse
from datetime import datetime
import os
import pandas as pd
import wandb
import numpy as np
import random
import logging
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo


class MockContextManager:

    def __init__(self, **kwargs) -> None:
        pass

    def __enter__(self):
        pass

    def __exit__(self, type, value, traceback):
        pass


class MockScaler:

    def scale(self, loss):
        return loss

    def step(self, **kwargs):
        pass

    def update(self, **kwargs):
        pass


class CSVLogger:
    
    def __init__(self, log_file):
        self.rows = []
        self.log_file = log_file 
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)

    def log(self, epoch, step, row, silent=False):
        row = {"timestamp": datetime.timestamp(datetime.now()), "epoch": epoch, "step": step, **row}
        self.rows.append(row)
        pd.DataFrame(self.rows).to_csv(self.log_file, index=False)


class ConsoleLogger:

    def log(self, epoch, step, row, silent=False):
        if not silent:
            logging.info(f"[{datetime.now()}] Epoch {epoch} - {row}")


class WandBLogger:
    
    def __init__(self, project, args):
        wandb.init(config=args, project=project)

    def log(self, epoch, step, row, silent=False):
        row = {"timestamp": datetime.timestamp(datetime.now()), "epoch": epoch, "step": step, **row}
        wandb.log(row)
        

class CheckpointCallback:
    
    CKPT_PATTERN = "epoch=%d-step=%d.ckpt"
    
    def __init__(self, path, mode="all", args=None):
        
        assert mode in ["all", None]
        
        self.path = path 
        self.mode = mode
        self.args = args
        
        os.makedirs(self.path, exist_ok=True)

    def save(self, epoch, step, model, metrics):
        if self.mode == "all":
            out_path = os.path.join(self.path, self.CKPT_PATTERN % (epoch, step))
            logging.debug(f"saving {out_path}")
            torch.save(
                {
                    "state_dict": model.state_dict(), 
                    "metrics": {"epoch": epoch, "step": step, **metrics}, 
                    "args": self.args
                }, out_path)


    
class NormalizedModel(torch.nn.Module):
    
    def __init__(self, model, mean, std):
        super(NormalizedModel, self).__init__()
        self.model = model
        self.mean = torch.nn.Parameter(torch.Tensor(mean).view(-1, 1, 1), requires_grad=False)
        self.std = torch.nn.Parameter(torch.Tensor(std).view(-1, 1, 1), requires_grad=False)
        
    def forward(self, x):
        out = (x - self.mean) / self.std 
        out = self.model(out)
        return out
    
    
def none2str(value):  # from https://stackoverflow.com/questions/48295246/how-to-pass-none-keyword-as-command-line-argument
    if value == "None":
        return None
    return value


def str2bool(v):
    """
    Converts string to bool type; enables command line 
    arguments in the format of "--arg1 true --arg2 false"
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def prepend_key_prefix(d, prefix):
    return dict((prefix + key, value) for (key, value) in d.items())


def seed_everything(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def get_arg(args, key, fallback=None):
    if key in vars(args):
        return vars(args)[key]
    return fallback


def get_gpu_stats():
    nvmlInit()
    stats = []
    for i in range(torch.cuda.device_count()):
        handle = nvmlDeviceGetHandleByIndex(i)
        info = nvmlDeviceGetMemoryInfo(handle)
        stats.append(info.used)
    return stats

