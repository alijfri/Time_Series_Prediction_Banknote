import pandas as pd
import numpy as np
import mlflow
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error
from darts.dataprocessing.transformers import Scaler

from darts.metrics import mae, rmse, mse, smape, r2_score
import numpy as np
import torch
from darts.timeseries import TimeSeries




def compute_metrics(
    val_preds_rolled,
    val_scaled_target_scaled,
    scalers_target,
    train_target_scaled, 
    metrics=["mae", "rmse", "mse", "smape", "r2","rse"]
):
    """
    Compute evaluation metrics between predicted and true validation series.

    Parameters:
    - val_preds_rolled: list of list[TimeSeries] — predicted (per series)
    - val_scaled_target_scaled: list of TimeSeries — true values (scaled)
    - scalers_target: list of fitted scalers (for inverse transform)
    - train_target_scaled: list of TimeSeries (needed for slicing actuals aligned with predictions)
    - metrics: list of metrics to compute: "mae", "rmse", "mse", "smape", "r2", "nll"
    - probabilistic: if True, assumes preds contain Gaussian mean + std

    Returns:
    - Dictionary with average metrics across all series
    """
    results = {metric: [] for metric in metrics}
    scalers_copy = scalers_target.copy()
    for i in range(len(val_preds_rolled)):
        # Merge all prediction chunks
        pred_series_scaled = val_preds_rolled[i][0]
        for s in val_preds_rolled[i][1:]:
            pred_series_scaled = pred_series_scaled.append(s)

        # Inverse scale prediction
        pred_unscaled = scalers_copy[i].inverse_transform(pred_series_scaled)
        actual_unscaled = scalers_copy[i].inverse_transform(val_scaled_target_scaled[i])

        # Inverse scale full target
        # full_series_unscaled = scalers_copy[i].inverse_transform(
        #     train_target_scaled[i].concatenate(val_scaled_target_scaled[i],ignore_time_axis=True)
        # )

        # Slice ground truth to match prediction time index
        try:
            # Align the actual and predicted series by intersection of their time index
            pred_unscaled = pred_unscaled.slice_intersect(actual_unscaled)
            actual_unscaled = actual_unscaled.slice_intersect(pred_unscaled)
            #actual_unscaled = full_series_unscaled[pred_unscaled.time_index]
        except Exception as e:
            print(f"⚠️ Skipping series {i} due to alignment error: {e}")
            continue

        # Metrics
        # Metrics (Note: y_true comes first!)
        if "mae" in metrics:
            results["mae"].append(mae(actual_unscaled, pred_unscaled))
        if "rmse" in metrics:
            results["rmse"].append(rmse(actual_unscaled, pred_unscaled))
        if "mse" in metrics:
            results["mse"].append(mse(actual_unscaled, pred_unscaled))
        if "smape" in metrics:
            results["smape"].append(smape(actual_unscaled, pred_unscaled))
        if "r2" in metrics:
            results["r2"].append(r2_score(actual_unscaled, pred_unscaled))
        if "rse" in metrics:
            y_true = actual_unscaled.values().flatten()
            y_pred = pred_unscaled.values().flatten()
            numerator = np.sum((y_true - y_pred) ** 2)
            denominator = np.sum((y_true - np.mean(y_true)) ** 2)
            rse_value = numerator / denominator if denominator != 0 else np.nan
            results["rse"].append(rse_value)

    # Aggregate average scores
    avg_results = {f"avg_{k}": np.nanmean(v) for k, v in results.items()}
    return avg_results


