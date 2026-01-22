# Time Series Prediction - Banknote Demand

This project trains multiple time-series models (TCN, N-BEATS, NLinear, LSTM, TFT, and XGBoost via Darts) to forecast banknote demand using static attributes and time-based covariates.

## Project layout

- `main.py`: CLI entry point for training/validation runs.
- `preprocess.py`: feature engineering, covariate generation, scaling, and data loading.
- `models.py`: model selection and orchestration.
- `*_model.py`: model-specific training helpers.
- `metrics.py`: evaluation metrics and MLflow logging.
- `saving_and_plot.py`: backtesting and prediction export helpers.

## Requirements

Install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data expectations

`main.py` expects a data directory with:

- `train_split.csv`
- `valid_split.csv`

Each CSV should include:

- `date`: timestamp column (parseable by pandas).
- `series_id`: identifier for each time series.
- `demand`: target value to forecast.
- `is_holiday`: binary indicator used for feature creation.
- One-hot encoded static features used in `main.py`:
  - `region_*`, `note_type_*`, `demand_type_*`, `denom_*` (see `main.py` for full list).

`preprocess.py` augments the data with:

- Date-based features (payday flags, seasonal sin/cos terms).
- Lag and rolling statistics (configurable; currently disabled in `main.py`).

## Usage

Run a training + validation workflow:

```bash
python main.py \
  --data_path /path/to/data/ \
  --model_type deeptcn \
  --input_chunk 52 \
  --output_chunk 8
```

Available model types include: `deeptcn`, `lstm`, `nbeats`, `nlinear`, `tft`, `xgboost`.

Results are written to `results_<model_type>/`, including `metrics.csv` and prediction exports.

## MLflow

The pipeline logs parameters and metrics to MLflow. By default it uses the local tracking URI.
Set `MLFLOW_TRACKING_URI` in the environment if you want to log to a remote server.