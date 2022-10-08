import torch
import sys
import argparse
from train import Trainer
import data
import os


def main(args):
    ckpt = torch.load(args.load_checkpoint, map_location="cpu")
    saved_args = argparse.Namespace()

    for k, v in ckpt["args"].items():
        vars(saved_args)[k] = v

    dataset = data.get_dataset(saved_args.dataset)(os.path.join(
            saved_args.dataset_dir, saved_args.dataset))

    vars(saved_args)["load_checkpoint"] = args.load_checkpoint
    vars(saved_args)["device"] = "cpu"
    vars(saved_args)["model_in_channels"] = dataset.in_channels
    vars(saved_args)["model_num_classes"] = dataset.num_classes

    trainer = Trainer(saved_args)

    trainer.model.eval()

    # Input to the model
    x = torch.randn(1, 3, 32, 32, requires_grad=True)
    torch_out = trainer.model(x)

    trainer.model.train()


    # Export the model
    torch.onnx.export(trainer.model,               # model being run
                    x,                         # model input (or a tuple for multiple inputs)
                    args.output,   # where to save the model (can be a file or file-like object)
                    export_params=True,        # store the trained parameter weights inside the model file
                    opset_version=13,          # the ONNX version to export the model to
                    do_constant_folding=False,  # whether to execute constant folding for optimization
                    training=torch.onnx.TrainingMode.TRAINING, 
                    input_names = ['input'],   # the model's input names
                    output_names = ['output'], # the model's output names
                    dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                    'output' : {0 : 'batch_size'}})




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("load_checkpoint", type=str, default=None)
    parser.add_argument("output", type=str, default=None)
    _args = parser.parse_args()
    main(_args)
    sys.exit(0)
