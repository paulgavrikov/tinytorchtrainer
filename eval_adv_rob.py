from ctypes import LibraryLoader
import torch
import sys
import argparse
from train import Trainer
from autoattack import AutoAttack
from utils import NormalizedModel
import os
import data
from utils import none2str, str2bool
import wandb
import foolbox as fb


def parse_aa_log(log_file):
    results = {}
    prev_attack = ""
    with open(log_file, "r") as file:
        for line in file.readlines():
            if "accuracy" in line:
                acc = float(line.split(": ")[1].replace("%", "").strip().split(" ")[0]) / 100

                tag = None
                if "initial accuracy" in line:
                    tag = "clean"
                elif "after" in line:
                    tag = line.split(":")[0].split(" ")[-1].strip()
                    if len(prev_attack) == 0:
                        prev_attack = tag
                    else:
                        prev_attack += "+" + tag

                    tag = "AA-" + prev_attack
                else:
                    tag = "AA-robust"

                results[tag] = acc

    return results


def run_foolbox_attack(attack, model, x, y, eps, batch_size, dataset, device):
    fmodel = fb.PyTorchModel(model, bounds=(0, 1), device=device)

    pos = 0
    perturbed = 0
    while pos < len(x):
        _, _, success = attack(fmodel, x[pos:pos + batch_size], y[pos:pos + batch_size], epsilons=[eps])
        perturbed += success.float().sum(axis=-1)[0].item()
        pos += batch_size

    return 1 - (perturbed / len(x))


def clean_acc(model, x, y, batch_size):
    successes = 0
    pos = 0
    while pos < len(x):
        logits = model(x[pos:pos + batch_size])
        c = (logits.argmax(axis=1) == y[pos:pos + batch_size]).sum().item()
        successes += c
        pos += batch_size
    acc = successes / len(x)
    return acc


def main(args):

    if args.wandb_project:
        wandb.init(config=vars(args), project=args.wandb_project)

    ckpt = torch.load(args.load_checkpoint, map_location="cpu")
    saved_args = argparse.Namespace()

    for k, v in ckpt["args"].items():
        vars(saved_args)[k] = v

    vars(saved_args)["load_checkpoint"] = args.load_checkpoint
    vars(saved_args)["device"] = args.device

    loader_batch = args.n_samples
    if args.n_samples == -1:
        loader_batch = saved_args.batch_size

    dataset = data.get_dataset(saved_args.dataset)(os.path.join(
            args.dataset_dir, saved_args.dataset))

    vars(saved_args)["model_in_channels"] = dataset.in_channels
    vars(saved_args)["model_num_classes"] = dataset.num_classes

    trainer = Trainer(saved_args)

    all_x = []
    all_y = []
    loader = None

    if args.data_split == "val":
        loader = dataset.val_dataloader(loader_batch, saved_args.num_workers)
    elif args.data_split == "train":
        loader = dataset.train_dataloader(loader_batch, saved_args.num_workers)

    for x, y in loader:
        all_x.append(x.to(trainer.device))
        all_y.append(y.to(trainer.device))

        if args.n_samples != -1:
            break

    all_x = torch.vstack(all_x)
    all_y = torch.hstack(all_y)

    model = NormalizedModel(trainer.model, dataset.mean, dataset.std).to(trainer.device)
    model.eval()

    all_x = all_x * model.std + model.mean  # unnormalize samples for AA

    results = {}

    acc = clean_acc(model, all_x, all_y, args.batch_size)
    results["clean"] = acc
    print(f"Clean: {acc}")

    if args.aa:
        log_file = args.log_file
        if log_file is None:
            log_file = os.path.join(os.path.dirname(args.load_checkpoint), f"aa_{args.norm}_{args.eps}.log")

        if os.path.isfile(log_file):
            os.remove(log_file)

        adversary = AutoAttack(model, norm=args.norm, eps=args.eps / 255, log_path=log_file, device=args.device, version="standard")
        _ = adversary.run_standard_evaluation(all_x, all_y)
        for k, v in parse_aa_log(log_file).items():
            results[f"aa/{k}"] = v

    if args.pgd:
        acc = run_foolbox_attack(fb.attacks.LinfPGD(), model, all_x, all_y, args.eps / 255, args.batch_size, dataset, args.device)
        results["pgd/robust"] = acc
        print(f"PGD: {acc}")

    if args.fgsm:
        acc = run_foolbox_attack(fb.attacks.FGSM(), model, all_x, all_y, args.eps / 255, args.batch_size, dataset, args.device)
        results["fgsm/robust"] = acc
        print(f"FGSM: {acc}")

    if args.wandb_project:
        wandb.log(results)
    print(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("load_checkpoint", type=str, default=None)
    
    parser.add_argument("--dataset_dir", type=str, default="/workspace/data/datasets")
    parser.add_argument("--data_split", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--n_samples", type=int, default=-1)
    
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=4)
    
    parser.add_argument("--aa", type=str2bool, default=True)
    parser.add_argument("--fgsm", type=str2bool, default=False)
    parser.add_argument("--pgd", type=str2bool, default=False)
    
    parser.add_argument("--norm", type=str, default="Linf")
    parser.add_argument("--eps", type=float, default=1)
    
    parser.add_argument("--log_file", type=none2str, default=None)
    parser.add_argument("--wandb_project", type=none2str, default=None)
    _args = parser.parse_args()
    main(_args)
    sys.exit(0)
