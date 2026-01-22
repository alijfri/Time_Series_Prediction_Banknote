
import pandas as pd
from darts import TimeSeries
from darts.models import TCNModel,NBEATSModel
from darts.dataprocessing.transformers import Scaler
from darts.metrics import mae
import matplotlib.pyplot as plt
from darts.utils.likelihood_models.torch import GaussianLikelihood, QuantileRegression,NegativeBinomialLikelihood,PoissonLikelihood
from pytorch_lightning.callbacks import EarlyStopping # Import EarlyStopping
import torch
import numpy as np
import os
from metrics import compute_metrics
from deeptcn_model import train_deeptcn
from nbeats_model import train_nbeats
from nlinear_model import train_nlinear
from lstm_model import train_lstm
from tft_model import train_tft
# from rnn_model import train_rnn
# from gru_model import train_gru



model_registry = {
    "deeptcn": train_deeptcn,
     "nbeats": train_nbeats,
     "nlinear":train_nlinear,
     "LSTM":train_lstm,
     "tft":train_tft
    #  "RNN":train_rnn,
    #  "GRU":train_gru,
     
    # "lstm": train_lstm,
    # Add other models here as needed
}
def train_and_predict_models(
    model_name,
    train_target_scaled,
    train_dynamic_covariates_scaled,
    val_scaled_target_scaled,
    val_dynamic_covariates_scaled,
    scalers_target,
    hyperparams,
    directory
):
    train_fn = model_registry[model_name]
   
    model = train_fn(
        train_target_scaled,
        val_scaled_target_scaled,
        train_dynamic_covariates_scaled,
        val_dynamic_covariates_scaled,
        scalers_target,
        hyperparams,
        directory,
        model_name=model_name
    )
    
   
    

