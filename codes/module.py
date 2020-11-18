from pytorch_lightning import LightningModule
from l5kit.evaluation import write_pred_csv
import torch
import pytorch_lightning as pl
import numpy as np
from l5kit.evaluation.metrics import neg_multi_log_likelihood


class LyftModule(LightningModule):
    def __init__(self, model, optimizer, scheduler, criterion):
        super().__init__()
        # save all variables in __init__ signature to self.hparams
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        outputs = self(batch["image"])
        return {
            "loss": self.compute_loss(batch, outputs)
        }

    def validation_step(self, batch, batch_idx):
        outputs = self(batch["image"])
        return {
            "val_loss": self.compute_loss(batch, outputs)
        }

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
        if self.optimizer is None:
            return None
        return {
            "optimizer": self.optimizer,
            "scheduler": self.scheduler,
            "monitor": "val_loss"
        }

    def compute_loss(self, batch, outputs):
        target_availabilities = batch["target_availabilities"]
        targets = batch["target_positions"]
        preds, confidences = outputs
        loss = self.criterion(
            targets, preds, confidences, target_availabilities
        )
        return loss
