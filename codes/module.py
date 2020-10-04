from pytorch_lightning import LightningModule
import torch
import pytorch_lightning as pl


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
        target_availabilities = batch["target_availabilities"].unsqueeze(-1)
        targets = batch["target_positions"]
        data = batch["image"]

        outputs = self(data).reshape(targets.shape)
        loss = self.criterion(outputs, targets)
        loss = (loss * target_availabilities).mean()

        result = pl.TrainResult(loss)
        result.log("train_loss", loss, on_epoch=True)
        return result

    def validation_step(self, batch, batch_idx):
        target_availabilities = batch["target_availabilities"].unsqueeze(-1)
        targets = batch["target_positions"]
        data = batch["image"]

        outputs = self(data).reshape(targets.shape)
        loss = self.criterion(outputs, targets)
        loss = (loss * target_availabilities).mean()
        
        result = pl.EvalResult(checkpoint_on=loss)
        result.log("val_loss", loss)
        return result

    def configure_optimizers(self):
        if self.optimizer is None:
            return None
        if not isinstance(self.optimizer, list):
            self.optimizer = [self.optimizer]
        if self.scheduler is None:
            return self.optimizer
        if not isinstance(self.scheduler, list):
            self.scheduler = [self.scheduler]
        return self.optimizer, self.scheduler
