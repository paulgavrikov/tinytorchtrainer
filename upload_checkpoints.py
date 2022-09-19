import wandb
from tqdm import tqdm
import argparse
import os


def main(args):
    api = wandb.Api()
    runs = api.runs(args.wandb_project)

    for run in tqdm(filter(lambda r: r.state == "finished", runs)):
        config = run.config
        output_dir = config["output_dir"]
        for k, v in config.items():
            if f"%{k}%" in output_dir:
                output_dir = output_dir.replace(f"%{k}%", v if type(v) == str else str(v))

        df = run.history()
        best_row = df.iloc[df["val/acc"].argmax()]

        checkpoint_dir = os.path.join(output_dir, f"checkpoints/epoch={int(best_row.epoch)}-step={int(best_row.step)}.ckpt")
        checkpoint_best_dir = os.path.join(output_dir, f"checkpoints/best.ckpt")

        os.system(f"cp {checkpoint_dir} {checkpoint_best_dir}")
 
        wandb.init(project=args.wandb_project, id=run.id, resume="must", reinit=True)
        wandb.save(checkpoint_best_dir) 



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("wandb_project", type=str)
    _args = parser.parse_args()
    main(_args)
