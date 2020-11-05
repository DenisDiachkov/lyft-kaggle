import argparse
import os
import warnings
from collections import Counter
from datetime import datetime
from multiprocessing import cpu_count
from pathlib import Path
from pprint import pprint

import numpy as np
import pytorch_lightning
import pytorch_lightning as pl
import torch
import torch.nn as nn
import zarr
from l5kit.configs import load_config_data
from l5kit.data import PERCEPTION_LABELS, ChunkedDataset, LocalDataManager
from l5kit.dataset import AgentDataset, EgoDataset
from l5kit.evaluation import write_pred_csv
from l5kit.evaluation.metrics import neg_multi_log_likelihood
from l5kit.geometry import transform_points
from l5kit.rasterization import build_rasterizer
from l5kit.visualization import TARGET_POINTS_COLOR, draw_trajectory
from prettytable import PrettyTable
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from torch.utils.data import DataLoader
from torchvision.models.resnet import resnet34, resnet50
from tqdm.notebook import tqdm

mode = "test"
gpu = 0
num_workers = cpu_count()
history_num_frames = 10
future_num_frames=50
batch_size=4
distributed_backend=None
pretrained_path=None


def get_test_cfg():
    return {
        "format_version": 4,
        "model_params": {
            "history_num_frames": history_num_frames,
            "history_step_size": 1,
            "history_delta_time": 0.1,
            "future_num_frames": future_num_frames,
            "future_step_size": 1,
            "future_delta_time": 0.1
        },
        
        "raster_params": {
            "raster_size": [300, 300],
            "pixel_size": [0.5, 0.5],
            "ego_center": [0.25, 0.5],
            "map_type": "py_semantic",
            "satellite_map_key": "aerial_map/aerial_map.png",
            "semantic_map_key": "semantic_map/semantic_map.pb",
            "dataset_meta_key": "meta.json",
            "filter_agents_threshold": 0.5,
            # Just to remove a warning
            "disable_traffic_light_faces": False
        },
    }


class LyftLDM(LightningDataModule):
    def __init__(self, data_root):
        super().__init__()
        self.cfg = get_test_cfg()
        self.data_root = data_root
        self.dm = LocalDataManager(data_root)
        self.rast = build_rasterizer(self.cfg, self.dm)

    def chunked_dataset(self, relative_path):
        dataset_path = self.dm.require(relative_path)
        zarr_dataset = ChunkedDataset(dataset_path)
        zarr_dataset.open()
        return zarr_dataset

    def test_dataloader(self):
        return self.get_dataloader(
            "scenes/test.zarr",
            False,
            np.load(f"{self.data_root}/scenes/mask.npz")["arr_0"]
        )

    def get_dataloader(self, zarr_dataset_path, shuffle, agent_mask=None):
        zarr_dataset = self.chunked_dataset(zarr_dataset_path)
        if agent_mask is None:
            agent_dataset = AgentDataset(self.cfg, zarr_dataset, self.rast)
        else:
            agent_dataset = AgentDataset(
                self.cfg, zarr_dataset, self.rast, agents_mask=agent_mask
            )
        return DataLoader(
            agent_dataset,
            shuffle=shuffle,
            batch_size=batch_size,
            num_workers=num_workers,
        )


class LyftNet(LightningModule):
    def __init__(
        self, 
        history_num_frames, 
        future_num_frames,
        pretrained=True,
        num_modes=3,
        **kw
    ):
        super().__init__()
        num_history_channels = (history_num_frames+1) * 2
        num_in_channels = 3 + num_history_channels
        num_targets = 2 * future_num_frames
        self.future_len = future_num_frames
        self.backbone = resnet34(pretrained=pretrained)
        
        self.backbone.conv1 = nn.Conv2d(
            num_in_channels,
            self.backbone.conv1.out_channels,
            kernel_size=self.backbone.conv1.kernel_size,
            stride=self.backbone.conv1.stride,
            padding=self.backbone.conv1.padding,
            bias=False,
        )
        backbone_out_features = 512
        self.head = nn.Sequential(
            nn.Linear(in_features=backbone_out_features, out_features=4096),
        )
        self.num_preds = num_targets * num_modes
        self.num_modes = num_modes
        self.logit = nn.Linear(4096, out_features=self.num_preds + num_modes)

    def forward(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)

        x = self.head(x)
        x = self.logit(x)

        bs, _ = x.shape
        pred, confidences = torch.split(x, self.num_preds, dim=1)
        pred = pred.view(bs, self.num_modes, self.future_len, 2)
        assert confidences.shape == (bs, self.num_modes)
        confidences = torch.softmax(confidences, dim=1)
        return pred, confidences


class LyftModule(LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)

    def test_step(self, batch, batch_idx):
        outputs = self(batch["image"])
        return {
            "future_coords_offsets_pd": outputs[0].cpu().numpy(),
            "timestamps": batch["timestamp"].cpu().numpy(),
            "agent_ids": batch["track_id"].cpu().numpy(),
            "confidences": outputs[1].cpu().numpy(),
        }

    def test_epoch_end(self, outputs):
        write_pred_csv('submission.csv',
            timestamps=np.concatenate(
                [output["timestamps"] for output in outputs]),
            track_ids=np.concatenate(
                [output["agent_ids"] for output in outputs]),
            coords=np.concatenate(
                [output["future_coords_offsets_pd"] for output in outputs]),
            confs=np.concatenate(
                [output["confidences"] for output in outputs])
        )
        return {}

    def configure_optimizers(self):
        return None


if __name__ == "__main__":
    os.environ["L5KIT_DATA_FOLDER"] = os.path.realpath(
        "../input/lyft-motion-prediction-autonomous-vehicles")
    trainer = Trainer(
        gpus=gpu,
        resume_from_checkpoint=pretrained_path,
        distributed_backend=distributed_backend,
    )
    trainer.test(
        LyftModule(LyftNet(history_num_frames, future_num_frames)),
        datamodule=LyftLDM(os.environ["L5KIT_DATA_FOLDER"]))
