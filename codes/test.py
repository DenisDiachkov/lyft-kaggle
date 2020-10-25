import argparse
import os

from pytorch_lightning import Trainer

from dataset import LyftLDM
from module import LyftModule
from net import LyftNet


def test_args(parent_parser):
    parser = argparse.ArgumentParser(
        parents=[parent_parser], add_help=False)
    # Model options
    parser.add_argument(
        "--history_num_frames", "-hnf", type=int, default=10)
    parser.add_argument(
        "--future_num_frames", "-fnf", type=int, default=50)

    # Train options 
    parser.add_argument(
        "--batch_size", "-bs", type=int, default=4)
    parser.add_argument(
        "--distributed_backend", "-db", default=None)
    parser.add_argument(
        "--pretrained_path", "-pp", type=str)
    args = parser.parse_args()
    return args


def get_module(args):
    model = LyftNet(args.history_num_frames, args.future_num_frames)
    return LyftModule(model, None, None, None)


def test(args, parser):
    args = test_args(parser)
    trainer = Trainer(
        gpus=args.gpu,
        deterministic=True,
        resume_from_checkpoint=args.pretrained_path,
        distributed_backend=args.distributed_backend,
        max_steps=20
    )
    trainer.test(
        get_module(args),
        datamodule=LyftLDM(args, os.environ["L5KIT_DATA_FOLDER"]))
