import torch
import models
import data
import os
import argparse
import yaml
import sys
import logging
import numpy as np
from tqdm import tqdm
from datetime import datetime
from rich.progress import track

from utils import (
    CSVLogger,
    ConsoleLogger,
    WandBLogger,
    CheckpointCallback,
    MockContextManager,
    MockScaler,
    LabelSmoothingLoss
    none2str,
    str2bool,
    prepend_key_prefix,
    seed_everything,
    get_arg,
    get_gpu_stats,
    cutmix_batch,
    cutmix_loss
)


class Trainer:
    def __init__(self, args):
        self.model = Trainer.prepare_model(
            args, args.model_in_channels, args.model_num_classes
        )
        self.device = args.device
        self.args = args
        self.loggers = []

    def prepare_model(args, in_channels, num_classes):

        logging.info(f"Initializing {args.model}")

        model = models.get_model(args.model)(
            in_channels=in_channels, num_classes=num_classes
        )

        if args.load_checkpoint is not None:

            logging.info(f"Loading state from {args.load_checkpoint}")

            state = torch.load(args.load_checkpoint, map_location="cpu")

            if "state_dict" in state:
                state = state["state_dict"]

            model.load_state_dict(state)

        if get_arg(args, "reset_head"):
            logging.info("Resetting head")
            model.fc.reset_parameters()

        if get_arg(args, "reset_all_but_conv2d_3x3"):
            for mname, module in filter(lambda t: len(list(t[1].children())) == 0, model.named_modules()):
                if type(module) is not torch.nn.Conv2d or module.kernel_size == (1, 1):
                    logging.info(f"Resetting {mname} {module}")
                    try:
                        module.reset_parameters()
                    except:
                        logging.info("... failed")

        if get_arg(args, "freeze_layers"):
            for mname, module in filter(lambda t: len(list(t[1].children())) == 0, model.named_modules()):
                if type(module).__name__ in args.freeze_layers.split(","):
                    for pname, param in module.named_parameters():
                        logging.info(f"Freezing {mname}/{pname} {module}")
                        param.requires_grad = False

        if get_arg(args, "freeze_conv2d_3x3"):
            for mname, module in filter(lambda t: len(list(t[1].children())) == 0, model.named_modules()):
                if type(module) is torch.nn.Conv2d and module.kernel_size != (1, 1):
                    for pname, param in module.named_parameters():
                        logging.info(f"Freezing {mname}/{pname} {module}")
                        param.requires_grad = False

        model.to(args.device)

        logging.info("TRAINABLE PARAMETERS:")
        for name, p in filter(lambda p: p[1].requires_grad, model.named_parameters()):
            logging.info(f" > {name} {p.shape}")
        logging.info(
            f"TOTAL: {sum(list(map(lambda p: p.numel(), filter(lambda p: p.requires_grad, model.parameters()))))}"
        )

        return model

    def train(self, model, trainloader, opt, criterion, device, scaler, context, scheduler=None):

        correct = 0
        total = 0
        total_loss = 0

        model.train()
        for x, y in track(trainloader, description="Training  ", total=len(trainloader)):
            x = x.to(device)
            y = y.to(device)

            r_cutmix = np.random.rand(1)
            use_cutmix = False
            if r_cutmix < get_arg(self.args, "cutmix_prob", 0):
                use_cutmix = True
                x, y, target_a, target_b, lam = cutmix_batch(x, y, self.args.cutmix_beta)

            opt.zero_grad()
            with context():
                y_pred = model(x)
                if not use_cutmix:
                    loss = criterion(y_pred, y)
                else:
                    loss = cutmix_loss(y_pred, target_a, target_b, lam)
            scaler.scale(loss).backward()

            batch_loss = loss.item() * len(y)
            batch_correct = (y_pred.argmax(axis=1) == y).sum().item()

            correct += batch_correct
            total_loss += batch_loss
            total += len(y)
            self.steps += 1

            scaler.step(opt)
            scaler.update()
        if scheduler:
            scheduler.step()

        return {"acc": correct / total, "loss": total_loss / total}

    def validate(self, model, valloader, criterion, device, scaler, context):
        correct = 0
        total = 0
        total_loss = 0

        model.eval()
        with torch.no_grad():
            for x, y in track(valloader, description="Validating", total=len(valloader)):
                x = x.to(device)
                y = y.to(device)

                with context():
                    y_pred = model(x)
                    loss = criterion(y_pred, y)
                total_loss += loss.item() * len(y)

                correct += (y_pred.argmax(axis=1) == y).sum().item()
                total += len(y)

        return {"acc": correct / total, "loss": total_loss / total}

    def warmup_bn(self, model, trainloader, device):
        with torch.no_grad():
            model.train()
            for x, _ in track(trainloader, description="Warm-Up   ", total=len(trainloader)):
                x = x.to(device)
                model(x)

    def _log(self, metrics_dict, **kwargs):
        for logger in self.loggers:
                logger.log(self.epoch, self.steps, metrics_dict, **kwargs)

    def fit(self, dataset, output_dir=None):
        torch.backends.cudnn.benchmark = get_arg(self.args, "cudnn_benchmark", False)

        trainloader = dataset.train_dataloader(self.args.batch_size, self.args.num_workers)
        valloader = dataset.val_dataloader(self.args.batch_size, self.args.num_workers)

        if self.args.optimizer == "sgd":
            self.opt = torch.optim.SGD(
                filter(lambda x: x.requires_grad, self.model.parameters()),
                lr=self.args.learning_rate,
                momentum=self.args.momentum,
                nesterov=self.args.nesterov,
                weight_decay=self.args.weight_decay,
            )
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.opt, step_size=get_arg(self.args, "scheduler_step", 30), gamma=0.1
            )
        elif self.args.optimizer == "adam":
            self.opt = torch.optim.Adam(
                filter(lambda x: x.requires_grad, self.model.parameters()),
                lr=self.args.learning_rate,
                weight_decay=self.args.weight_decay,
            )
            self.scheduler = None
        elif self.args.optimizer == "rmsprop":
            self.opt = torch.optim.RMSprop(
                filter(lambda x: x.requires_grad, self.model.parameters()),
                lr=self.args.learning_rate,
                weight_decay=self.args.weight_decay,
            )
            self.scheduler = None
        elif self.args.optimizer == "adamw":
            self.opt = torch.optim.AdamW(
                filter(lambda x: x.requires_grad, self.model.parameters()),
                lr=self.args.learning_rate,
                weight_decay=self.args.weight_decay,
            )
            self.scheduler = None

        self.criterion = LabelSmoothingLoss(smoothing=get_arg(self.args, "label_smoothing", 0)).to(self.device)

        if get_arg(self.args, "use_amp", False):
            self.scaler = torch.cuda.amp.GradScaler()
            self.context = torch.cuda.amp.autocast
        else:
            self.scaler = MockScaler()
            self.context = MockContextManager

        self.loggers = []
        self.loggers.append(ConsoleLogger())
        if output_dir:
            self.loggers.append(CSVLogger(os.path.join(output_dir, "metrics.csv")))
        if self.args.wandb_project:
            self.loggers.append(WandBLogger(self.args.wandb_project, self.args))

        if output_dir:
            self.checkpoint = CheckpointCallback(
                os.path.join(output_dir, "checkpoints"),
                args=self.args,
            )

        self.epoch = 0
        self.steps = 0

        if get_arg(self.args, "warmup_bn", False):
            self.warmup_bn(self.model, trainloader, self.device)

        if get_arg(self.args, "initial_val", False):
            val_metrics = self.validate(
                self.model, 
                valloader, 
                self.criterion, 
                self.device, 
                self.scaler, 
                self.context
            )
            self._log(prepend_key_prefix(val_metrics, "val/"))

        if output_dir:
            self.checkpoint.save(0, 0, self.model, {})

        val_acc_max = 0
        best_epoch = 0

        for epoch in range(self.args.max_epochs):

            self.epoch = epoch

            train_metrics = self.train(
                self.model,
                trainloader,
                self.opt,
                self.criterion,
                self.device,
                self.scaler, 
                self.context,
                self.scheduler
            )
            val_metrics = self.validate(
                self.model, 
                valloader, 
                self.criterion, 
                self.device, 
                self.scaler, 
                self.context
            )

            metrics = {
                **prepend_key_prefix(train_metrics, "train/"),
                **prepend_key_prefix(val_metrics, "val/"),
                "val/acc_max": val_acc_max,
                "best_epoch": best_epoch,
            }

            if val_acc_max < metrics["val/acc"]:
                val_acc_max = metrics["val/acc"]
                best_epoch = epoch

                metrics["val/acc_max"] = val_acc_max
                metrics["best_epoch"] = best_epoch

            self._log(metrics)
            if output_dir:
                self.checkpoint.save(epoch, self.steps, self.model, metrics)


def main(args):

    assert not (args.freeze_layers and args.freeze_conv2d_3x3)

    logging.basicConfig(level=logging.INFO)
    if get_arg(args, "verbose"):
        logging.basicConfig(level=logging.info)

    output_dir = args.output_dir
    for k, v in vars(args).items():
        if f"%{k}%" in output_dir:
            output_dir = output_dir.replace(f"%{k}%", v if type(v) == str else str(v))

    str_date_time = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_dir = output_dir.replace("%timestamp%", str_date_time)

    seed_everything(args.seed)

    dataset = data.get_dataset(args.dataset)(
        os.path.join(args.dataset_dir, args.dataset)
    )

    best_is_gpu = False

    if args.device == "auto":

        try:
            mps_available = torch.backends.mps.is_available()
        except:
            mps_available = False

        if torch.cuda.is_available():
            best_is_gpu = True
        elif mps_available:
            logging.info(f"Autoselected device: Apple Silicon")
            vars(args)["device"] = "mps"
        else:
            logging.info(f"Autoselected device: CPU")
            vars(args)["device"] = "cpu"


    if args.device == "auto_gpu_by_memory" or best_is_gpu:
        best_device = f"cuda:{np.argmin(get_gpu_stats())}"
        logging.info(f"Autoselected device: {best_device}")
        vars(args)["device"] = best_device


    if args.model_in_channels == -1:
        vars(args)["model_in_channels"] = dataset.in_channels

    if args.model_num_classes == -1:
        vars(args)["model_num_classes"] = dataset.num_classes

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "hparams.yaml"), "w") as file:
        yaml.dump(vars(args), file)

    Trainer(args).fit(dataset, output_dir=output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str)
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--dataset_dir", type=str, default="~/datasets/")
    parser.add_argument(
        "--output_dir", type=str, default="output/%dataset%/%model%/version_%seed%"
    )
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--cudnn_benchmark", type=str2bool, default=True)
    parser.add_argument("--use_amp", type=str2bool, default=False)
    parser.add_argument("--checkpoints", type=none2str, default=None, choices=["all", "best", "None", None])
    parser.add_argument("--checkpoints_metric", type=str, default="val/acc")
    parser.add_argument("--checkpoints_metric_target", type=str, default="max", choices=["max", "min"])

    parser.add_argument("--load_checkpoint", type=none2str, default=None)
    parser.add_argument("--reset_head", type=str2bool, default=False)
    parser.add_argument("--reset_all_but_conv2d_3x3", type=str2bool, default=False)
    parser.add_argument("--freeze_conv2d_3x3", type=str2bool, default=False)

    parser.add_argument("--warmup_bn", type=str2bool, default=False)
    parser.add_argument("--initial_val", type=str2bool, default=False)

    parser.add_argument("--model_in_channels", type=int, default=-1)
    parser.add_argument("--model_num_classes", type=int, default=-1)

    parser.add_argument("--max_epochs", type=int, default=125)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=0)

    # optimizer
    parser.add_argument("--optimizer", type=str, default="sgd", choices=["adam", "sgd", "adamw", "rmsprop"])

    # scheduler
    parser.add_argument("--scheduler_step", type=int, default=30)

    parser.add_argument("--cutmix_prob", type=float, default=0)
    parser.add_argument("--cutmix_beta", type=float, default=1)

    parser.add_argument("--label_smoothing", type=float, default=0.1)
    
    parser.add_argument("--learning_rate", type=float, default=1e-2)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--nesterov", type=str2bool, default=True)
    parser.add_argument("--freeze_layers", type=none2str, default=None)

    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--verbose", type=str2bool, default=False)

    parser.add_argument("--wandb_project", type=none2str, default=None)
    parser.add_argument("--wandb_notes", type=none2str, default=None)
    parser.add_argument("--wandb_extra_1", type=none2str, default=None)
    parser.add_argument("--wandb_extra_2", type=none2str, default=None)
    parser.add_argument("--wandb_extra_3", type=none2str, default=None)

    _args = parser.parse_args()
    main(_args)
    sys.exit(0)
