from darts.models import XGBModel


from pytorch_lightning.callbacks import EarlyStopping
from torch.optim.lr_scheduler import ReduceLROnPlateau

import os
import pandas as pd
from logger import LossHistoryLogger


def get_early_stopper():
    return EarlyStopping(
        monitor='val_loss',
        patience=10,
        mode='min',
        verbose=True
    )

def train_xgboost(
    train_target_scaled,
    val_scaled_target_scaled,
    train_dynamic_covariates_scaled,
    val_dynamic_covariates_scaled,
    scalers_target,
    hyperparams,
    directory,
    model_name="xgboost"
):
    early_stopper = get_early_stopper()
    num_samples = hyperparams.get("num_samples", 500)
    loss_logger = LossHistoryLogger(directory,log_to_mlflow=True)

    model = XGBModel(
        lags=hyperparams.get("input_chunk", 52),
        lags_past_covariates=hyperparams.get("input_chunk", 52),
        output_chunk_length=hyperparams.get("output_chunk", 4),
        n_estimators=25,
        learning_rate=0.4,  # same as eta
        gamma=0.01,
        max_depth=6,
        reg_alpha=2.1875,
        reg_lambda=2.1875,
        subsample=0.6,
        colsample_bytree=0.6,
        random_state=42,
        verbosity=1  # optional: can set to 0 for silence
    )

    model.fit(
            series=train_target_scaled,
            past_covariates=train_dynamic_covariates_scaled,
            val_series=val_scaled_target_scaled,
            val_past_covariates=val_dynamic_covariates_scaled,
            verbose=True
        )
    return model