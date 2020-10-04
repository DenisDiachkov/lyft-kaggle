import argparse
from datetime import datetime
import os

import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as sched
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger as tb

from module import LyftModule
from net import LyftNet
from dataset import LyftLDM


def train_args(parent_parser):
    parser = argparse.ArgumentParser(
        parents=[parent_parser], add_help=False)
    
    # Dataset options
    parser.add_argument(
        "--shuffle", type=bool, default=True)

    # Model options
    parser.add_argument(
        "--history_num_frames", "-hnf", type=int, default=10)
    parser.add_argument(
        "--future_num_frames", "-fnf", type=int, default=50)

    # Train options 
    parser.add_argument(
        "--batch_size", "-bs", type=int, default=4)
    parser.add_argument(
        "--distributed_backend", "-db", type=str, default="dp")
    parser.add_argument(
        "--epochs", type=int, default=4)
    parser.add_argument(
        "--iterations_per_epoch", "-ipe", type=int)
    parser.add_argument(
        "--experiment_name", "-exn", type=str,
        default=datetime.now().strftime("%d-%m-%Y_%H%M%S"))
    parser.add_argument(
        "--resume", action="store_true")
    parser.add_argument(
        "--pretrained_path", "-pp", type=str)
    args = parser.parse_args()
    return args


def get_module(args):
    model = LyftNet(args.history_num_frames, args.future_num_frames)
    optimizer = optim.Adam(
        model.parameters(), lr=1e-4)
    scheduler = sched.CosineAnnealingWarmRestarts(
        optimizer, 150)
    criterion = nn.MSELoss()
    return LyftModule(model, optimizer, scheduler, criterion)


def train(args, parser):
    args = train_args(parser)
    tb_logger = tb(".", "experiments", version=args.experiment_name)
    trainer = Trainer(
        gpus=args.gpu,
        logger=tb_logger,
        num_sanity_val_steps=1,
        deterministic=True,
        limit_train_batches=1.0 if args.iterations_per_epoch is None
        else args.iterations_per_epoch,
        limit_val_batches=1.0 if args.iterations_per_epoch is None
        else args.iterations_per_epoch,
        row_log_interval=1,
        log_save_interval=1,
        resume_from_checkpoint=args.pretrained_path if args.resume else None,
        distributed_backend=args.distributed_backend,
    )
    trainer.fit(
        get_module(args),
        datamodule=LyftLDM(args, os.environ["L5KIT_DATA_FOLDER"]))
