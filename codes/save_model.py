import argparse
import os
from collections import OrderedDict

from pytorch_lightning import Trainer

from dataset import LyftLDM
from module import LyftModule
from net import LyftNet
import torch


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Model options
    parser.add_argument(
        "--history_num_frames", "-hnf", type=int, default=10)
    parser.add_argument(
        "--future_num_frames", "-fnf", type=int, default=50)
    parser.add_argument(
        "--pretrained_path", "-pp", type=str, default="/home/d/Desktop/lyft-kaggle/experiments/17-11-2020_192406/checkpoints/epoch=0.ckpt")
    args = parser.parse_args()

    net = LyftNet(args.history_num_frames, args.future_num_frames)
    net_state_dict = torch.load(args.pretrained_path)['state_dict']
    net_state_dict = OrderedDict([(k[6:], v) if k[:6] == 'model.' else (k, v) for k, v in net_state_dict.items()])
    torch.save(net_state_dict, 'submission_model.pth')
