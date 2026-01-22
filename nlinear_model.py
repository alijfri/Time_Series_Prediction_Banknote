
from darts.models import NLinearModel


from pytorch_lightning.callbacks import EarlyStopping
import os
import pandas as pd
from logger import LossHistoryLogger,get_early_stopper
from darts.models import NLinearModel




def train_nlinear(
    train_target_scaled,
    val_scaled_target_scaled,
    train_dynamic_covariates_scaled,
    val_dynamic_covariates_scaled,
    scalers_target,
    hyperparams,
    directory,
    model_name="nlinear"
):
    train_target_scaled = [
    ts.with_static_covariates(ts.static_covariates.drop(columns=["series_id"], errors="ignore"))
    if ts.static_covariates is not None
    else ts
    for ts in train_target_scaled
]

    val_scaled_target_scaled = [
        ts.with_static_covariates(ts.static_covariates.drop(columns=["series_id"], errors="ignore"))
        if ts.static_covariates is not None
        else ts
        for ts in val_scaled_target_scaled
    ]
    
    loss_logger = LossHistoryLogger(directory,log_to_mlflow=True)
    early_stopper = get_early_stopper()    
    model = NLinearModel(
        model_name=model_name,
        input_chunk_length=hyperparams.get("input_chunk", 52),
        output_chunk_length=hyperparams.get("output_chunk", 4),
        n_epochs=hyperparams.get("n_epochs", 150),
        optimizer_kwargs={
        "lr": hyperparams.get("lr", 1e-3),
        "weight_decay": hyperparams.get("weight_decay", 1e-5) # Add weight_decay
    },
        force_reset=True,
        save_checkpoints=True,
        random_state=42,
        pl_trainer_kwargs={
            "accelerator": "gpu",
            "callbacks": [early_stopper,loss_logger]
        }
    )

    model.fit(
        series=train_target_scaled,
        past_covariates=train_dynamic_covariates_scaled,
        val_series=val_scaled_target_scaled,
        val_past_covariates=val_dynamic_covariates_scaled,
        verbose=True
    )

    return model