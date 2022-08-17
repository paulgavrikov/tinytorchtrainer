import torch
import numpy as np
import random
import models
import data
import os
import argparse
import yaml
import sys
from utils import CSVLogger, CheckpointCallback, none2str, str2bool, prepend_key_prefix


class Trainer:

    def __init__(self, args, output_dir):
        self.dataset = data.get_dataset(args.dataset)(os.path.join(
            args.dataset_dir, args.dataset), args.batch_size, args.num_workers)
        self.model = Trainer.prepare_model(args, self.dataset.in_channels, self.dataset.num_classes)

        self.opt = torch.optim.SGD(
            filter(lambda x: x.requires_grad, self.model.parameters()), 
            lr=args.learning_rate, momentum=args.momentum, nesterov=args.nesterov, weight_decay=args.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.opt, step_size=30, gamma=0.1)
        self.criterion = torch.nn.CrossEntropyLoss()

        self.loggers = [CSVLogger(os.path.join(output_dir, "metrics.csv"))]

        self.checkpoint = CheckpointCallback(os.path.join(
            output_dir, "checkpoints"), mode=args.checkpoints, args=vars(args))

        self.epoch = 0
        self.steps = 0

        self.max_epochs = args.max_epochs

        self.device = args.device

    def prepare_model(args, in_channels, num_classes):
        model = models.get_model(args.model)(
            in_channels=in_channels, num_classes=num_classes)

        if args.load_checkpoint is not None:
            state = torch.load(args.load_checkpoint, map_location="cpu")

            if "state_dict" in state:
                state = state["state_dict"]

            model.load_state_dict(state)

        if args.reset_head:
            model.fc.reset_parameters()

        if args.freeze_layers:
            for module in model.modules():
                if type(module).__name__ in args.freeze_layers.split(","):
                    for param in module.parameters():
                        param.requires_grad = False

        model.to(args.device)
        return model

    def train(self, model, trainloader, opt, criterion, device, scheduler=None):

        correct = 0
        total = 0
        total_loss = 0
        model.train()
        for x, y in (trainloader):
            x = x.to(device)
            y = y.to(device)

            opt.zero_grad()
            y_pred = model(x)
            loss = criterion(y_pred, y)
            loss.backward()
            opt.step()

            total_loss += loss.item() * len(y)
            correct += (y_pred.argmax(axis=1) == y).sum().item()
            total += len(y)
            self.steps += 1

        if scheduler:
            scheduler.step()

        return {
            "acc": correct / total,
            "loss": total_loss / total
        }

    def validate(self, model, valloader, criterion, device):
        correct = 0
        total = 0
        total_loss = 0
        model.eval()
        with torch.no_grad():
            for x, y in (valloader):
                x = x.to(device)
                y = y.to(device)

                y_pred = model(x)
                loss = criterion(y_pred, y)
                total_loss += loss.item() * len(y)

                correct += (y_pred.argmax(axis=1) == y).sum().item()
                total += len(y)

        return {
            "acc": correct / total,
            "loss": total_loss / total
        }

    def fit(self):
        trainloader = self.dataset.train_dataloader()
        valloader = self.dataset.val_dataloader()

        val_metrics = self.validate(
            self.model, valloader, self.criterion, self.device)
        for logger in self.loggers:
            logger.log(0, 0, prepend_key_prefix(val_metrics, "val/"))
        self.checkpoint.save(0, 0, self.model, {})

        for epoch in range(self.max_epochs):

            self.epoch = epoch

            train_metrics = self.train(
                self.model, trainloader, self.opt, self.criterion, self.device, self.scheduler)
            val_metrics = self.validate(
                self.model, valloader, self.criterion, self.device)

            metrics = {**prepend_key_prefix(train_metrics, "train/"), **prepend_key_prefix(val_metrics, "val/")}
            for logger in self.loggers:
                logger.log(epoch, self.steps, metrics)
            self.checkpoint.save(epoch, self.steps, self.model, metrics)


def main(args):

    output_dir = args.output_dir
    for k, v in vars(args).items():
        if f"%{k}%" in output_dir:
            output_dir = output_dir.replace(
                f"%{k}%", v if type(v) == str else str(v))

    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "hparams.yaml"), "w") as file:
        yaml.dump(vars(args), file)

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    Trainer(args, output_dir).fit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str)
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--dataset_dir", type=str,
                        default="/workspace/data/datasets/")
    parser.add_argument("--output_dir", type=str,
                        default="output/%dataset%/%model%/version_%seed%")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--checkpoints", type=none2str,
                        default=None, choices=["all", None])
    parser.add_argument("--load_checkpoint", type=str, default=None)
    parser.add_argument("--reset_head", type=str2bool, default=False)

    parser.add_argument("--max_epochs", type=int, default=125)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=0)

    # optimizer
    parser.add_argument("--learning_rate", type=float, default=1e-2)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--nesterov", type=str2bool, default=True)
    parser.add_argument("--freeze_layers", type=none2str, default=None)

    parser.add_argument("--seed", type=int, default=0)
    _args = parser.parse_args()
    main(_args)
    sys.exit(0)
