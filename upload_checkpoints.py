import wandb
from tqdm import tqdm
import argparse
import os


def main(args):
    api = wandb.Api()
    runs = api.runs(args.wandb_project)

    for run in tqdm(filter(lambda r: r.state == "finished" and r.config["model"] == "lowres_mobilenet_v2", runs)):
        config = run.config
        output_dir = config["output_dir"]
            
        for k, v in (run.config).items():
            if f"%{k}%" in output_dir:
                output_dir = output_dir.replace(f"%{k}%", v if type(v) == str else str(v))
        checkpoint_best_dir = os.path.join(output_dir, f"checkpoints/best.ckpt") 
        wandb.init(project=args.wandb_project, id=run.id, resume="must", reinit=True)
        wandb.save(checkpoint_best_dir) 



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("wandb_project", type=str)
    _args = parser.parse_args()
    main(_args)
