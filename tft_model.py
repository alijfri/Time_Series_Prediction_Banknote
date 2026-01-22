from darts.models import  TFTModel
from pytorch_lightning.callbacks import EarlyStopping
import os
import pandas as pd
from darts.utils.losses import SmapeLoss
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

from logger import LossHistoryLogger,get_early_stopper

def train_tft(
    train_target_scaled,
    val_target_scaled,
    train_dynamic_covariates_scaled,
    val_dynamic_covariates_scaled,
    scalers_target,
    hyperparams,
    directory,
    model_name="tft"
):
    loss_logger = LossHistoryLogger(directory, log_to_mlflow=True)
    early_stopper = get_early_stopper()  # Define this separately as in your original code
    
    model = TFTModel(
        model_name=model_name,
        input_chunk_length=hyperparams.get("input_chunk", 52),
        output_chunk_length=hyperparams.get("output_chunk", 4),
        hidden_size=hyperparams.get("hidden_size", 16),
        lstm_layers=hyperparams.get("n_lstm_layers", 2),
        num_attention_heads=hyperparams.get("num_attention_heads", 4),
        full_attention=hyperparams.get("full_attention", False),
        feed_forward=hyperparams.get("feed_forward", "GatedResidualNetwork"),
        dropout=hyperparams.get("dropout", 0.1),
        hidden_continuous_size=hyperparams.get("hidden_continuous_size", 8),
        categorical_embedding_sizes={"series_id": 133},  # If needed, pass dictionary here
        add_relative_index=True,
        loss_fn=None,                     # You can define custom loss if desired
        #likelihood=None,                  # Or use QuantileRegression or GaussianLikelihood etc.
        norm_type='LayerNorm',
        use_static_covariates=True,
        optimizer_kwargs={
            "lr": hyperparams.get("lr", 1e-3),
            "weight_decay": hyperparams.get("weight_decay", 1e-5),
        },
        force_reset=True,
        save_checkpoints=True,
        random_state=42,
        nr_epochs_val_period=hyperparams.get("nr_epochs_val_period", 1),
        pl_trainer_kwargs={
            "accelerator": "gpu",
            "callbacks": [early_stopper, loss_logger],
            "max_epochs": hyperparams.get("n_epochs", 150),
            "gradient_clip_val": 1.0,
            "gradient_clip_algorithm": "norm",
        }
    )

    print("Model Specific:", model)

    model.fit(
        series=train_target_scaled,
        past_covariates=train_dynamic_covariates_scaled,
        val_series=val_target_scaled,
        val_past_covariates=val_dynamic_covariates_scaled,
        future_covariates=None,
        val_future_covariates=None,
        verbose=True,
    )

    return model
