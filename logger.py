import mlflow
import pandas as pd
from pytorch_lightning.callbacks import Callback
import os
from pytorch_lightning.callbacks import EarlyStopping

def get_early_stopper():
    return EarlyStopping(
        monitor='val_loss',
        patience=5,
        mode='min',
        verbose=True
    )
class LossHistoryLogger(Callback):
    def __init__(self, csv_path, log_to_mlflow=True):
        super().__init__()
        self.train_losses = []
        self.val_losses = []
        self.csv_path = csv_path
        self.log_to_mlflow = log_to_mlflow

    def on_train_epoch_end(self, trainer, pl_module):
        train_loss = trainer.callback_metrics.get("train_loss")
        val_loss = trainer.callback_metrics.get("val_loss")

        # Save to internal list
        self.train_losses.append(train_loss.item() if train_loss is not None else None)
        self.val_losses.append(val_loss.item() if val_loss is not None else None)

        # Log to MLflow
        epoch = len(self.train_losses) - 1
        if self.log_to_mlflow:
            if train_loss is not None:
                mlflow.log_metric("train_loss", train_loss.item(), step=epoch)
            if val_loss is not None:
                mlflow.log_metric("val_loss", val_loss.item(), step=epoch)

    def on_train_end(self, trainer, pl_module):
        # Save to CSV at end of training
        df = pd.DataFrame({
            "epoch": list(range(len(self.train_losses))),
            "train_loss": self.train_losses,
            "val_loss": self.val_losses
        })
        df.to_csv(os.path.join(self.csv_path, "log_loss.csv"), index=False)
        #df.to_csv(self.csv_path, index=False)
        print(f"Saved loss history to {self.csv_path}")
