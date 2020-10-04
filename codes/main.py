import argparse
import os
import warnings
from multiprocessing import cpu_count
from datetime import datetime
from test import test

import utils
from train import train


def base_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", type=str, choices=["train", "test"], default="train")
    parser.add_argument(
        "--gpu", type=str)
    parser.add_argument(
        "--cpu", action="store_true")
    parser.add_argument(
        "--num_workers", "--jobs", "-j",
        type=int, choices=range(cpu_count()+1), default=cpu_count())
    parser.add_argument("--Wall", action="store_true")
    args, _ = parser.parse_known_args()
    utils.set_device(args)
    return args, parser


def main():
    os.environ["L5KIT_DATA_FOLDER"] = \
        os.path.realpath("../input/lyft-motion-prediction-autonomous-vehicles")
    args, parser = base_args()
    if args.Wall:
        warnings.simplefilter("error")
    if args.mode == "train":
        train(args, parser)
    elif args.mode == "test":
        test(args, parser)


if __name__ == "__main__":
    main()
