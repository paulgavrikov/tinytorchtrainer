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
import foolbox as fb


from utils import (
    CSVLogger,
    ConsoleLogger,
    WandBLogger,
    CheckpointCallback,
    MockContextManager,
    MockScaler,
    LabelSmoothingLoss,
    none2str,
    str2bool,
    prepend_key_prefix,
    seed_everything,
    get_arg,
    get_gpu_stats,
    cutmix_batch,
    cutmix_loss,
    eval_adv,
    adv_attack
)


def load_trainer(args):
    ckpt = torch.load(args.load_checkpoint, map_location="cpu")
    saved_args = argparse.Namespace()

    for k, v in ckpt["args"].items():
        vars(saved_args)[k] = v

    vars(saved_args)["load_checkpoint"] = args.load_checkpoint
    vars(saved_args)["device"] = args.device

    dataset = data.get_dataset(saved_args.dataset)(os.path.join(
            saved_args.dataset_dir, saved_args.dataset))

    vars(saved_args)["model_in_channels"] = dataset.in_channels
    vars(saved_args)["model_num_classes"] = dataset.num_classes

    trainer = Trainer(saved_args)
    return trainer


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

        activation = get_arg(args, "activation", None)

        model = models.get_model(args.model)(
            in_channels=in_channels, num_classes=num_classes, activation_fn=getattr(torch.nn, activation) if activation else None
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

        if get_arg(args, "freeze_all_but_bn", False):  
            for mname, module in filter(lambda t: len(list(t[1].children())) == 0, model.named_modules()):
                if type(module) is not torch.nn.BatchNorm2d:
                    for pname, param in module.named_parameters():
                        logging.info(f"Freezing {mname}/{pname} {module}")
                        param.requires_grad = False

        if get_arg(args, "freeze_layers") == "conv3x3":
            for mname, module in filter(lambda t: len(list(t[1].children())) == 0, model.named_modules()):
                if type(module) is torch.nn.Conv2d and module.kernel_size == (3, 3):
                    for pname, param in module.named_parameters():
                        logging.info(f"Freezing {mname}/{pname} {module}")
                        param.requires_grad = False
        elif get_arg(args, "freeze_layers") == "conv1x1":
            for mname, module in filter(lambda t: len(list(t[1].children())) == 0, model.named_modules()):
                if type(module) is torch.nn.Conv2d and module.kernel_size == (1, 1):
                    for pname, param in module.named_parameters():
                        logging.info(f"Freezing {mname}/{pname} {module}")
                        param.requires_grad = False
        elif get_arg(args, "freeze_layers"):
            for mname, module in filter(lambda t: len(list(t[1].children())) == 0, model.named_modules()):
                if type(module).__name__ in args.freeze_layers.split(","):
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
        
        logging.info(model)

        return model

    def train(self, model, trainloader, opt, criterion, device, scaler, context, scheduler=None, dataset=None):

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

            if get_arg(self.args, "adv_train", False):
                x = adv_attack(model=model, fb_attack=getattr(fb.attacks, self.args.adv_train_attack), 
                    attack_extras=eval(self.args.adv_train_attack_extras), dataset=dataset, device=self.device)

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
        elif self.args.optimizer == "adam":
            self.opt = torch.optim.Adam(
                filter(lambda x: x.requires_grad, self.model.parameters()),
                lr=self.args.learning_rate,
                weight_decay=self.args.weight_decay,
            )
        elif self.args.optimizer == "rmsprop":
            self.opt = torch.optim.RMSprop(
                filter(lambda x: x.requires_grad, self.model.parameters()),
                lr=self.args.learning_rate,
                weight_decay=self.args.weight_decay,
            )
        elif self.args.optimizer == "adamw":
            self.opt = torch.optim.AdamW(
                filter(lambda x: x.requires_grad, self.model.parameters()),
                lr=self.args.learning_rate,
                weight_decay=self.args.weight_decay,
            )
        else:
            raise NotImplementedError()

        steps_per_epoch = len(trainloader)

        if get_arg(self.args, "scheduler", "step") == "step":
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.opt, step_size=get_arg(self.args, "scheduler_step", 30) * steps_per_epoch, gamma=0.1
            )
        elif self.args.scheduler == "frankle_step":
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
                self.opt, milestones=[80*steps_per_epoch, 120*steps_per_epoch], gamma=0.1
            )
        elif self.args.scheduler == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.opt, T_max=self.args.max_epochs * steps_per_epoch
            )
        elif self.args.schedule is None:
            self.scheduler = None      
        else:
            raise NotImplementedError()

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
                self.scheduler,
                dataset
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

            if self.args.adv_train:
                 metrics["val/robust_acc"] = eval_adv(model=self.model, fb_attack=getattr(fb.attacks, self.args.adv_val_attack), 
                    attack_extras=eval(self.args.adv_val_attack_extras), dataset=dataset, device=self.device)

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
    vars(args)["output_dir"] = output_dir

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
    parser.add_argument("--checkpoints", type=none2str, default=None, choices=["all", "best", "None", "last", None])
    parser.add_argument("--checkpoints_metric", type=str, default="val/acc")
    parser.add_argument("--checkpoints_metric_target", type=str, default="max", choices=["max", "min"])

    parser.add_argument("--load_checkpoint", type=none2str, default=None)
    parser.add_argument("--reset_head", type=str2bool, default=False)
    parser.add_argument("--reset_all_but_conv2d_3x3", type=str2bool, default=False)
    parser.add_argument("--freeze_conv2d_3x3", type=str2bool, default=False)
    parser.add_argument("--freeze_all_but_bn", type=str2bool, default=False)

    parser.add_argument("--warmup_bn", type=str2bool, default=False)
    parser.add_argument("--initial_val", type=str2bool, default=False)

    parser.add_argument("--model_in_channels", type=int, default=-1)
    parser.add_argument("--model_num_classes", type=int, default=-1)

    parser.add_argument("--activation", type=none2str, default=None)

    parser.add_argument("--max_epochs", type=int, default=125)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)

    # optimizer
    parser.add_argument("--optimizer", type=str, default="sgd", choices=["adam", "sgd", "adamw", "rmsprop"])

    # scheduler
    parser.add_argument("--scheduler", type=none2str, default="step", choices=[None, "step", "cosine", "frankle_step"])
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

    # adversarial training
    parser.add_argument("--adv_train", type=str2bool, default=False)
    parser.add_argument("--adv_train_attack", type=str)
    parser.add_argument("--adv_train_attack_extras", type=none2str, default=None)
    parser.add_argument("--adv_val_attack", type=str)
    parser.add_argument("--adv_val_attack_extras", type=none2str, default=None)

    # wandb
    parser.add_argument("--wandb_project", type=none2str, default=None)
    parser.add_argument("--wandb_notes", type=none2str, default=None)
    parser.add_argument("--wandb_extra_1", type=none2str, default=None)
    parser.add_argument("--wandb_extra_2", type=none2str, default=None)
    parser.add_argument("--wandb_extra_3", type=none2str, default=None)

    _args = parser.parse_args()
    main(_args)
    sys.exit(0)
