from pytorch_lightning import LightningModule
import torch
from torchvision.models.resnet import resnet50, resnet34
import torch.nn as nn


class LyftNet(LightningModule):
    def __init__(
        self, 
        history_num_frames, 
        future_num_frames,
        pretrained=True,
        **kwargs
    ):
        super().__init__()
        num_history_channels = (history_num_frames+1) * 2
        num_in_channels = 3 + num_history_channels
        num_targets = 2 * future_num_frames

        resnet = resnet34(pretrained=pretrained)
        resnet.conv1 = nn.Conv2d(
            num_in_channels,
            resnet.conv1.out_channels,
            kernel_size=resnet.conv1.kernel_size,
            stride=resnet.conv1.stride,
            padding=resnet.conv1.padding,
            bias=False,
        )
        resnet.fc = nn.Linear(in_features=512, out_features=num_targets)
        self.resnet = resnet

    def forward(self, data):
        return self.resnet(data)