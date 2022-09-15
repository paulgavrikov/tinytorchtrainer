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

        artifact = wandb.Artifact("best.ckpt", type="model", metadata=best_row.to_dict())
        artifact.add_file(checkpoint_dir)
        run.log_artifact(artifact) 



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("wandb_project", type=str, default=None)
    _args = parser.parse_args()
    main(_args)
