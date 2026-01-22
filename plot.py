import matplotlib.pyplot as plt
import numpy as np

def plot_series_forecast(series_id, results_df, fh, lower_unscaled=None, upper_unscaled=None):
    """
    Plot the forecast for a specific series_id from the results_df.

    Parameters:
        series_id (int or str): ID of the time series to plot
        results_df (pd.DataFrame): Must contain ['series_id', 'date', 'value', 'yhat']
        fh (int): Forecast horizon
        lower_unscaled (np.ndarray): Optional array of lower bounds [n_series, fh]
        upper_unscaled (np.ndarray): Optional array of upper bounds [n_series, fh]
    """
    # Filter for this series
    series_df = results_df[results_df["series_id"] == series_id].copy()
    if len(series_df) != fh:
        raise ValueError(f"Expected {fh} rows for series_id={series_id}, got {len(series_df)}")

    dates = series_df["date"]
    true_vals = series_df["value"]
    forecast = series_df["yhat"]

    plt.figure(figsize=(10, 5))
    plt.plot(dates, true_vals, label="True", marker='o')
    plt.plot(dates, forecast, label="Forecast", marker='o')

    if lower_unscaled is not None and upper_unscaled is not None:
        # We assume this series_id maps to row i in lower_unscaled
        idx = results_df["series_id"].drop_duplicates().tolist().index(series_id)
        lower_vals = lower_unscaled[idx]
        upper_vals = upper_unscaled[idx]
        plt.fill_between(dates, lower_vals, upper_vals, color='gray', alpha=0.3, label="90% Interval")

    plt.title(f"Forecast for Series {series_id}")
    plt.xlabel("Date")
    plt.ylabel("Unscaled Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
