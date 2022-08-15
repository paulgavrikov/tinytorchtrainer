from git import RemoteProgress
import torch
from tqdm import tqdm
import argparse
from datetime import datetime
import os
import pandas as pd


class CSVLogger:
    
    def __init__(self, log_file):
        self.rows = []
        self.log_file = log_file 

    def log(self, epoch, step, row):
        row = {"timestamp": datetime.timestamp(datetime.now()), "epoch": epoch, "step": step, **row}
        print(row)
        self.rows.append(row)
        pd.DataFrame(self.rows).to_csv(self.log_file, index=False)
        

class CheckpointCallback:
    
    CKPT_PATTERN = "epoch=%epoch%step=%step%.ckpt"
    
    def __init__(self, path, mode="all"):
        
        assert mode in ["all", None]
        
        self.path = path 
        self.mode = mode
        os.makedirs(self.path, exist_ok=True)

    def save(self, epoch, step, model, metrics):
        if self.mode == "all":
            out_path = os.path.join(self.path, self.CKPT_PATTERN.replace("%epoch%", str(epoch)).replace("%step%", str(step)))
            torch.save({"state_dict": model.state_dict(), "metrics": metrics, "epoch": epoch, "step": step}, out_path)


class CloneProgress(RemoteProgress):
    def update(self, op_code, cur_count, max_count=None, message=''):
        pbar = tqdm(total=max_count)
        pbar.update(cur_count)

    
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
    if value == 'None':
        return None
    return value


def str2bool(v):
    """
    Converts string to bool type; enables command line 
    arguments in the format of '--arg1 true --arg2 false'
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def prepend_key_prefix(d, prefix):
    return dict((prefix + key, value) for (key, value) in d.items())