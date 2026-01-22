from darts.models import NBEATSModel, TCNModel,NLinearModel,BlockRNNModel,XGBModel,TFTModel
from darts import TimeSeries, concatenate
import os
import pandas as pd
import numpy as np

model_names_syntax = {
    "deeptcn": TCNModel,
    "nbeats": NBEATSModel,
    "nlinear": NLinearModel,
    "LSTM":BlockRNNModel,
    "xgboost":XGBModel,
    "tft": TFTModel  # Assuming TFTModel is defined elsewhere
}


def save_train_predictions_to_disk_single_csv(
    preds_unscaled,
    actuals_unscaled,
    results_dir,
    filename,
    lower_bounds=None,
    upper_bounds=None):
    """
    Save all training predictions and actuals into a single CSV with optional confidence intervals.
    Uses original series_id from static covariates.

    Parameters:
    - preds_unscaled: list of TimeSeries (predicted values)
    - actuals_unscaled: list of TimeSeries (actuals)
    - results_dir: str — output folder
    - lower_bounds: list of TimeSeries (optional)
    - upper_bounds: list of TimeSeries (optional)
    - filename: str — name of output file
    """
    filename=filename+"_predictions.csv"

    os.makedirs(results_dir, exist_ok=True)
    all_rows = []


    for i, pred_ts in enumerate(preds_unscaled):
        # Print frequencies
        

        # Get real series_id
        actual_ts = actuals_unscaled[i]
        series_id = (
            actual_ts.static_covariates_values()[0][0]
        )

        actual_ts = actuals_unscaled[i]
        print(f"Series {i} - pred freq: {pred_ts.freq}, actual freq: {actual_ts.freq}")
        # Align prediction and actuals
        aligned_pred = pred_ts.slice_intersect(actual_ts)
        aligned_actual = actual_ts.slice_intersect(aligned_pred)

        # Extract actual
        actual_df = aligned_actual.to_dataframe().rename(columns={aligned_actual.components[0]: "actual"})

        # Check if probabilistic
        is_probabilistic = aligned_pred.n_samples > 1

        if is_probabilistic:
            # Median prediction
            median_ts = aligned_pred.quantile_timeseries(0.5).slice_intersect(aligned_actual)
            median_df = median_ts.to_dataframe().rename(columns={median_ts.components[0]: "predicted_0.5"})

            # Sample matrix
            samples = aligned_pred.all_values().squeeze(axis=1)  # shape: (time, num_samples)
            sample_df = pd.DataFrame(samples, index=median_df.index)
            sample_df.columns = [f"sample_{j}" for j in range(samples.shape[1])]

            df = pd.concat([median_df, actual_df, sample_df], axis=1)

            if lower_bounds is not None and upper_bounds is not None:
                lower = lower_bounds[i].slice_intersect(aligned_pred)
                upper = upper_bounds[i].slice_intersect(aligned_pred)
                df["lower_10th"] = lower.values().flatten()
                df["upper_90th"] = upper.values().flatten()

        else:
            # Deterministic model: single forecast
            pred_df = aligned_pred.to_dataframe().rename(columns={aligned_pred.components[0]: "predicted"})
            df = pd.concat([pred_df, actual_df], axis=1)

        df["series_id"] = series_id
        df["timestamp"] = df.index

        all_rows.append(df)

    final_df = pd.concat(all_rows, axis=0)
    final_df.reset_index(drop=True, inplace=True)
    final_df.to_csv(os.path.join(results_dir, filename), index=False)




def backtest(
    all_series_scaled,
    all_dynamic_covariates_scaled,
    model_name,
    target_series,
    dynamic_covariates,
    scalers_target,
    args,
    start_offset_chunks=1,
    quantiles=(10, 90)
):
    """
    Generate rolling predictions with optional confidence intervals from a loaded model.

    Parameters:
    - model_name: str
    - model: loaded Darts model
    - target_series: list of TimeSeries (scaled)
    - dynamic_covariates: list of TimeSeries (scaled)
    - scalers_target: list of fitted scalers
    - args: argparse or dict with hyperparams
    - start_offset_chunks: int — how many input_chunks to skip for rolling start
    - quantiles: tuple — lower and upper quantiles (for CI bands)

    Returns:
    - preds_unscaled: list of TimeSeries
    - lower_bounds: list of TimeSeries or None
    - upper_bounds: list of TimeSeries or None
    """
    model_class = model_names_syntax[model_name]
    model = model_class.load_from_checkpoint(model_name=model_name, best=True)
    # Check if model is probabilistic
    is_probabilistic = model_name.lower() in ["deeptcn", "deep_ar", "transformer", "tft"]
    num_samples = args.num_samples if is_probabilistic else 1

    preds_scaled = []
    lower_bounds = [] if is_probabilistic else None
    upper_bounds = [] if is_probabilistic else None
    preds_rolled = []
    if model_name in ["nlinear"]:
        all_series_scaled = [
        ts.with_static_covariates(ts.static_covariates.drop(columns=["series_id"], errors="ignore"))
        if ts.static_covariates is not None
        else ts
        for ts in all_series_scaled
    ]
    ts = target_series[0]


            
    for i, series in enumerate(all_series_scaled):
        hist_predictions = model.historical_forecasts(
            series=series,
            past_covariates=all_dynamic_covariates_scaled[i],
            start=target_series[i].start_time() ,
            forecast_horizon=args.output_chunk,
            stride=1,
            retrain=False,
            verbose=False,
            num_samples=num_samples,
        )

        full_pred = concatenate(hist_predictions)
        preds_rolled.append(hist_predictions) ### Only for metrics
        preds_scaled.append(full_pred)

        if is_probabilistic:
            values = full_pred.all_values()  # shape: (time, 1, num_samples)
            lower = np.percentile(values, quantiles[0], axis=2).squeeze()
            upper = np.percentile(values, quantiles[1], axis=2).squeeze()
            lower_ts = full_pred.with_values(lower.reshape(-1, 1))
            upper_ts = full_pred.with_values(upper.reshape(-1, 1))
            lower_bounds.append(scalers_target[i].inverse_transform(lower_ts))
            upper_bounds.append(scalers_target[i].inverse_transform(upper_ts))

    preds_unscaled = [
        scalers_target[i].inverse_transform(preds_scaled[i]) for i in range(len(preds_scaled))
    ]

    return preds_unscaled, lower_bounds, upper_bounds,preds_rolled





    


