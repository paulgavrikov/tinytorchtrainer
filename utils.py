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
import torch.nn.functional as F


class LabelSmoothingLoss(torch.nn.Module):
    def __init__(self, smoothing=0.0, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
            true_dist += self.smoothing / pred.size(self.dim)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))

    
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

    def step(self, optimizer, **kwargs):
        optimizer.step()

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
        wandb.init(config=args, project=project, notes=get_arg(args, "wandb_notes", None))

    def log(self, epoch, step, row, silent=False):
        row = {"timestamp": datetime.timestamp(datetime.now()), "epoch": epoch, "step": step, **row}
        wandb.log(row)


class CheckpointCallback:

    CKPT_PATTERN = "epoch=%d-step=%d.ckpt"

    def __init__(self, path, args=None):
        self.path = path 
        self.mode = args.checkpoints
        self.args = args
        self.last_best = 0
        self.last_path = None
        self.target_metric = get_arg(args, "checkpoints_metric", "val/acc")
        self.target_metric_target = get_arg(args, "checkpoints_metric_target", "max")
        os.makedirs(self.path, exist_ok=True)
        logging.info(f"saving checkpoints to {self.path}")

    def save(self, epoch, step, model, metrics):
        if self.mode == "all":
            self.last_path = os.path.join(self.path, self.CKPT_PATTERN % (epoch, step))
            logging.info(f"saving {self.last_path}")
            torch.save(
                {
                    "state_dict": model.state_dict(), 
                    "metrics": {"epoch": epoch, "step": step, **metrics}, 
                    "args": vars(self.args)
                }, self.last_path)
        elif self.mode == "last":
            self.last_path = os.path.join(self.path, os.path.join(os.path.split(self.CKPT_PATTERN)[0], "last.ckpt"))
            logging.info(f"saving {self.last_path}")
            torch.save(
                {
                    "state_dict": model.state_dict(), 
                    "metrics": {"epoch": epoch, "step": step, **metrics}, 
                    "args": vars(self.args)
                }, self.last_path)
        elif self.mode == "best" and (self.target_metric in metrics) and ((metrics[self.target_metric] > self.last_best and self.target_metric_target == "max") or \
                (metrics[self.target_metric] < self.last_best and self.target_metric_target == "min")):
            self.last_best = metrics[self.target_metric]
            self.last_path = os.path.join(self.path, os.path.join(os.path.split(self.CKPT_PATTERN)[0], "best.ckpt"))
            logging.info(f"saving {self.last_path}")
            torch.save(
                {
                    "state_dict": model.state_dict(), 
                    "metrics": {"epoch": epoch, "step": step, **metrics}, 
                    "args": vars(self.args)
                }, self.last_path)



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


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def cutmix_batch(batch_x, batch_y, beta):
    lam = np.random.beta(beta, beta)
    rand_index = torch.randperm(batch_x.size()[0]).to(batch_x.device)
    target_a = batch_y
    target_b = batch_y[rand_index]
    bbx1, bby1, bbx2, bby2 = rand_bbox(batch_x.size(), lam)
    batch_x[:, :, bbx1:bbx2, bby1:bby2] = batch_x[rand_index, :, bbx1:bbx2, bby1:bby2]
    # adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (batch_x.size()[-1] * batch_x.size()[-2]))

    return batch_x, batch_y, target_a, target_b, lam


def cutmix_loss(y_pred, target_a, target_b, lam, loss_fn=F.cross_entropy):
    loss = loss_fn(y_pred, target_a) * lam + loss_fn(y_pred, target_b) * (
                        1. - lam)
    return loss


class LayerHook:

    def __init__(self):
        self.storage = None
        self.hook_handle = None

    def pull(self):
        if self.hook_handle is not None:
            self.hook_handle.remove()
        return self.storage

    def register_hook(self, module, store_input=True):
        if self.hook_handle is not None:
            self.hook_handle.remove()
        self.storage = None

        def hook(_, inp, out):
            if store_input:
                self.storage = inp.detach().cpu()
            else:
                self.storage = out[0].detach().cpu()
        self.hook_handle = module.register_forward_hook(hook)
