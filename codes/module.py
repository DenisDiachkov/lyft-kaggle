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

        loss = self.compute_loss(batch, outputs)
        eval_metric = self.compute_metric(batch, outputs)

        result = pl.TrainResult(loss)
        result.log("train_loss", loss, on_epoch=True, prog_bar=True)
        result.log("train_eval_metric", eval_metric, prog_bar=True)
        return result

    def validation_step(self, batch, batch_idx):
        outputs = self(batch["image"])

        loss = self.compute_loss(batch, outputs)
        eval_metric = self.compute_metric(batch, outputs)
        
        result = pl.EvalResult(checkpoint_on=loss)
        result.log("val_loss", loss, prog_bar=True)
        result.log("val_eval_metric", eval_metric, prog_bar=True)
        return result

    def test_step(self, batch, batch_idx):
        targets = batch["target_positions"]
        data = batch["image"]

        outputs = self(data).reshape(targets.shape)

        return {
            "future_coords_offsets_pd": outputs.cpu().numpy(),
            "timestamps": batch["timestamp"].cpu().numpy(),
            "agent_ids": batch["track_id"].cpu().numpy()
        }

    def test_epoch_end(self, outputs):
        write_pred_csv('submission.csv',
            timestamps=np.concatenate(
                [output["timestamps"] for output in outputs]),
            track_ids=np.concatenate(
                [output["agent_ids"] for output in outputs]),
            coords=np.concatenate(
                [output["future_coords_offsets_pd"] for output in outputs])
        )
        return {}

    def configure_optimizers(self):
        return {
            "optimizer": self.optimizer,
            "scheduler": self.scheduler,
            "monitor": "val_eval_metric"
        }

    def compute_loss(self, batch, outputs):
        target_availabilities = batch["target_availabilities"].unsqueeze(-1)
        targets = batch["target_positions"]
        outputs = outputs.reshape(targets.shape)
        loss = self.criterion(outputs, targets)
        loss = (loss * target_availabilities).mean()
        return loss

    def compute_metric(self, batch, outputs):
        target_availabilities = batch["target_availabilities"].unsqueeze(-1)
        targets = batch["target_positions"]
        outputs = outputs.reshape(targets.shape)
        eval_metric = 0
        for target, output, avail in zip(targets, outputs, target_availabilities):
            eval_metric += neg_multi_log_likelihood(
                target.cpu().numpy(),
                output.unsqueeze(0).detach().cpu().numpy(), 
                np.ones(1),
                avail.squeeze(1).cpu().numpy()
            )
        return torch.tensor(eval_metric)