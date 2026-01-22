from darts.models import BlockRNNModel
from pytorch_lightning.callbacks import EarlyStopping
import os
import pandas as pd
from darts.utils.losses import SmapeLoss
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau





from logger import LossHistoryLogger
def get_early_stopper():
    return EarlyStopping(
        monitor='val_loss',
        patience=10,
        mode='min',
        verbose=True
    )
# reduce_lr = ReduceLROnPlateau(
#     factor=0.5,             # Reduce LR by half
#     patience=10          # After 10 epochs of no improvement
# )

def train_lstm(
    train_target_scaled,
    val_scaled_target_scaled,
    train_dynamic_covariates_scaled,
    val_dynamic_covariates_scaled,
    scalers_target,
    hyperparams,
    directory,
    model_name="LSTM"
):
    
    loss_logger = LossHistoryLogger(directory,log_to_mlflow=True)
    early_stopper = get_early_stopper()    
    model = BlockRNNModel(
        model="LSTM",
        model_name="LSTM",
        input_chunk_length=hyperparams.get("input_chunk", 52),
        output_chunk_length=hyperparams.get("output_chunk", 4),
        hidden_dim=128,              # hidden units per LSTM layer
        n_rnn_layers=hyperparams.get("num_layers", 8),             # ✅ stacked LSTM layers
        batch_size=hyperparams.get("batch_size"),
        dropout=hyperparams.get("dropout", 0.2),
        #num_layers=hyperparams.get("num_layers", 8),
        n_epochs=hyperparams.get("n_epochs", 150),
        loss_fn=nn.HuberLoss(),
        optimizer_kwargs={
        "lr": hyperparams.get("lr", 1e-3),
        "weight_decay": hyperparams.get("weight_decay", 1e-5)  # ⬅️ L2 regularization
    },

        lr_scheduler_cls=ReduceLROnPlateau,
    lr_scheduler_kwargs={
        "mode": "min",        # because val_loss should decrease
        "factor": 0.5,
        "patience": 5,
    },
        force_reset=True,
        save_checkpoints=True,
        random_state=42,
        pl_trainer_kwargs={
            "accelerator": "gpu",
            "callbacks": [early_stopper,loss_logger],
            "max_epochs": hyperparams["n_epochs"],
            "gradient_clip_val": 1.0,
            "gradient_clip_algorithm": "norm"
        }
    )
    print("Model Specific: ", model)
    model.fit(
        series=train_target_scaled,
        past_covariates=train_dynamic_covariates_scaled,
        val_series=val_scaled_target_scaled,
        val_past_covariates=val_dynamic_covariates_scaled,
        val_future_covariates=None,
        future_covariates=None,
        verbose=True
    )

    return model