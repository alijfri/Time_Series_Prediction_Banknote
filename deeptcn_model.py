# model_trainer.py

from darts.models import TCNModel
from darts.utils.likelihood_models.torch import GaussianLikelihood
from pytorch_lightning.callbacks import EarlyStopping
from torch.optim.lr_scheduler import ReduceLROnPlateau
from darts.utils.losses import SmapeLoss
import os
import pandas as pd
from logger import LossHistoryLogger


def get_early_stopper():
    return EarlyStopping(
        monitor='val_loss',
        min_delta=0.1,
        patience=10,
        mode='min',
        verbose=True
    )

def train_deeptcn(
    train_target_scaled,
    val_scaled_target_scaled,
    train_dynamic_covariates_scaled,
    val_dynamic_covariates_scaled,
    scalers_target,
    hyperparams,
    directory,
    model_name="deeptcn"
):
    early_stopper = get_early_stopper()
    num_samples = hyperparams.get("num_samples", 500)
    loss_logger = LossHistoryLogger(directory,log_to_mlflow=True)

    model = TCNModel(
        model_name="deeptcn",
        input_chunk_length=hyperparams.get("input_chunk", 52),
        output_chunk_length=hyperparams.get("output_chunk", 4),
        kernel_size=hyperparams.get("kernel_size", 3),
        num_filters=hyperparams.get("num_filters", 128),
        num_layers=hyperparams.get("num_layers", 8),
        dropout=hyperparams.get("dropout", 0.2),
        n_epochs=hyperparams.get("n_epochs", 150),
        likelihood=GaussianLikelihood(),
        #loss_fn=SmapeLoss(),
        batch_size=hyperparams.get("batch_size", 64),  
        dilation_base=5, 
        weight_norm=True,
        force_reset=True,
        save_checkpoints=True,
        optimizer_kwargs={
        "lr": hyperparams.get("lr", 1e-3)
    },
        lr_scheduler_cls=ReduceLROnPlateau,
    lr_scheduler_kwargs={
        "mode": "min",        # because val_loss should decrease
        "factor": 0.5,
        "patience": 2,
    },
        random_state=42,
        pl_trainer_kwargs={
            "accelerator": "gpu",
            "callbacks": [early_stopper,loss_logger],
            "max_epochs": hyperparams["n_epochs"],
            "gradient_clip_val": 1.0,
            "gradient_clip_algorithm": "norm" }
    )

    model.fit(
        series=train_target_scaled,
        past_covariates=train_dynamic_covariates_scaled,
        val_series=val_scaled_target_scaled,
        val_past_covariates=val_dynamic_covariates_scaled,
        verbose=True
    )
    return model


