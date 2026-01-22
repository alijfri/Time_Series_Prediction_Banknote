from workalendar.america import Canada
from datetime import timedelta
from pandas.tseries.holiday import AbstractHolidayCalendar, Holiday, nearest_workday, GoodFriday, EasterMonday
from pandas.tseries.offsets import Week
import numpy as np
import pandas as pd
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
import torch


from darts import TimeSeries
import numpy as np

class LogTransformer:
    def __init__(self, epsilon=1e-3):
        self.epsilon = epsilon

    def fit(self, series: TimeSeries):
        return self  # stateless

    def transform(self, series: TimeSeries):
        values = series.values()
        if (values < 0).any():
            raise ValueError("Negative values found in input to LogTransformer.")
        if np.isnan(values).any() or np.isinf(values).any():
            raise ValueError("NaNs or Infs found in input before transform.")

        transformed = series.map(lambda x: np.log(x + self.epsilon))
        if np.isnan(transformed.values()).any():
            raise ValueError("NaNs produced in log transform")
        return transformed

    def fit_transform(self, series: TimeSeries):
        return self.transform(series)

    def inverse_transform(self, series: TimeSeries):
        # Use safe inverse with clipping
        values = series.values()
        if np.isnan(values).any() or np.isinf(values).any():
            raise ValueError("NaNs/Infs in input to inverse_transform")

        return series.map(lambda x: max(np.exp(x) - self.epsilon, 0.0))

    
    
def load_data(path, kind):
    df = pd.read_csv(path + kind)
    df['date'] = pd.to_datetime(df['date'])
    
    df.set_index('date', inplace=True)
    #df_filtered = df[df['week_ending'] == 5]
    return df



    

def create_feature_matrix(df, include_paydays=True, include_sin_cos=True):
    base = df.copy()

    base['year'] = base.index.year
    base['week'] = base.index.isocalendar().week
    base['month'] = base.index.month

    if include_paydays:
        all_dates = pd.date_range(start=df.index.min(), end=df.index.max(), freq='D')
        mid_month = all_dates[all_dates.day == 15]
        end_month = all_dates.to_series().groupby(all_dates.to_period("M")).max()
        paydays = pd.DatetimeIndex(mid_month).union(end_month)
        payday_year_week = set(zip(paydays.isocalendar().year, paydays.isocalendar().week))
        base['payday'] = base.apply(lambda row: 1 if (row['year'], row['week']) in payday_year_week else 0, axis=1)

        base['holiday_lag1'] = base['is_holiday'].shift(1)
        base['holiday_lag_minus_1'] = base['is_holiday'].shift(-1)
        base['payday_lag1'] = base['payday'].shift(1)
        base['payday_lag_minus_1'] = base['payday'].shift(-1)

    if include_sin_cos:
        base['sin_week'] = np.sin(2 * np.pi * base['week'] / 52)
        base['cos_week'] = np.cos(2 * np.pi * base['week'] / 52)
        base['sin_month'] = np.sin(2 * np.pi * base['month'] / 12)
        base['cos_month'] = np.cos(2 * np.pi * base['month'] / 12)
        base['sin_dayofweek'] = np.sin(2 * np.pi * base.index.dayofweek / 7)
        base['cos_dayofweek'] = np.cos(2 * np.pi * base.index.dayofweek / 7)
    del base['year']
    del base['week']
    del base['month']

    return base
def add_lags_and_rollings(df_target, df_reference, include_lags=True, include_rolling=True):
    df_reference_sorted = df_reference.sort_values(['series_id', 'date'])

    lag_list =[48,49,50,51,52,53,54,55]
    rolling_list = [3, 7, 10]

    lag_cols = [f'value_lag_{l}' for l in lag_list] if include_lags else []
    rolling_cols = (
        [f'rolling_mean_{w}' for w in rolling_list] + 
        [f'rolling_std_{w}' for w in rolling_list]
        if include_rolling else []
    )

    base = df_target.copy()
    columns_to_copy = ['series_id'] + lag_cols + rolling_cols

    for series_id in base['series_id'].unique():
        mask = base['series_id'] == series_id
        target_series = base[mask].sort_values('date').copy()
        ref_series = df_reference_sorted[df_reference_sorted['series_id'] == series_id].sort_values('date')

        N = len(target_series)
        ref_tail = ref_series[columns_to_copy].tail(N).reset_index(drop=True)
        target_series = target_series.reset_index(drop=True)

        for col in lag_cols + rolling_cols:
            if col in ref_tail.columns:
                target_series[col] = ref_tail[col].values

        # Set values back into base using the original mask
        base.loc[mask, lag_cols + rolling_cols] = target_series[lag_cols + rolling_cols].values

    return base



def add_lags_and_rollings_train(df_train, include_lags=True, include_rolling=True):
    """
    Add lag and rolling features to training dataset (self-referential).
    """
    base = df_train.copy()
    lag_list = [48,49,50,51,52,53,54,55]
    rolling_list = [3, 7, 10]

    for series_id in base['series_id'].unique():
        df_series = base[base['series_id'] == series_id].copy()
       # df_series = df_series.sort_values('date').set_index('date')

        if include_lags:
            for lag in lag_list:
                base.loc[base['series_id'] == series_id, f'value_lag_{lag}'] = (
                    df_series['demand'].shift(lag).values
                )

        if include_rolling:
            for window in rolling_list:
                base.loc[base['series_id'] == series_id, f'rolling_mean_{window}'] = (
                    df_series['demand'].shift(1).rolling(window, min_periods=1).mean().values
                )
                base.loc[base['series_id'] == series_id, f'rolling_std_{window}'] = (
                    df_series['demand'].shift(1).rolling(window, min_periods=1).std().values
                )

    return base


def removing_series_with_many_zeros(df,max_number_of_in_year_zero):
    zero_counts=zero_counts = df[df['demand'] == 0].groupby('series_id').size()
    
    series_to_remove = zero_counts[zero_counts >= max_number_of_in_year_zero].index
    
    df_filtered = df[~df['series_id'].isin(series_to_remove)]

    return df_filtered


    
    

    
def load_and_prepare_data(data_path, include_paydays=True,
                          include_sin_cos=True, include_lags=True,
                          include_rolling=True):
    # Load and process train
    # kind_train = 'trainset_split.csv'
    kind_train = 'train_split.csv'
    df_train = load_data(data_path, kind_train)
    df_features_train = create_feature_matrix(df_train, include_paydays, include_sin_cos)
    
    # Add lags and rollings to train
    df_features_train = add_lags_and_rollings_train(df_features_train, include_lags, include_rolling)
    df_features_train = df_features_train.dropna()
    #### Validation set
    kind_valid = 'valid_split.csv'
    df_valid = load_data(data_path, kind_valid)
    df_features_valid = create_feature_matrix(df_valid, include_paydays, include_sin_cos)
    df_features_valid = add_lags_and_rollings(df_features_valid, df_features_train,
                                             include_lags, include_rolling)
    
    

    df_features_valid.fillna(0, inplace=True)
    #df_features_train.fillna(0, inplace=True)

    return df_features_train, df_features_valid


def split_series(df_train,df_valid,dynamic_features,static_features):
    # Frist step we need to split the tain and test data into demand 
    
    df_train = df_train.reset_index()
    df_valid = df_valid.reset_index()
    
    #making it as type 32 
    target_column = 'demand'
    columns_to_convert_to_float32 = [target_column] + dynamic_features+static_features
    for col in columns_to_convert_to_float32:
        if col in df_train.columns:
            df_train[col] = pd.to_numeric(df_train[col], errors='coerce').astype('float32')
            df_valid[col] = pd.to_numeric(df_valid[col], errors='coerce').astype('float32')
    
    
    ### Statisc features and Creating TimeSeries objects for trianing series
    train_series = TimeSeries.from_group_dataframe(
    df_train,
    group_cols=['series_id'],
    time_col='date',
    value_cols=target_column,
    static_cols=static_features
)
    #### Dynamic features
    train_dynamic_covariates = TimeSeries.from_group_dataframe(
    df_train,
    group_cols=['series_id'],
    time_col='date',
    value_cols=dynamic_features)
    print(f"Number of series found in train set: {len(train_series)}")
    #### For Test series
        ### Statisc features and Creating TimeSeries objects for trianing series
    valid_series = TimeSeries.from_group_dataframe(
    df_valid,
    group_cols=['series_id'],
    time_col='date',
    value_cols=[target_column],
    static_cols=static_features)
    #### Dynamic features
    valid_dynamic_covariates = TimeSeries.from_group_dataframe(
    df_valid,
    group_cols=['series_id'],
    time_col='date',
    value_cols=dynamic_features)
    print(f"Number of series found in test dataset: {len(valid_series)}")
    
    ### Now all together 
 # Combine target series, ignoring time axis mismatch (use with caution)
    from darts import concatenate

    # Combine target series
    combined_series = [
        trn.concatenate(tst, ignore_time_axis=True)
        for trn, tst in zip(train_series, valid_series)
    ]
    
    combined_dynamic_covariates = [
        trn_cov.concatenate(tst_cov, ignore_time_axis=True)
        for trn_cov, tst_cov in zip(train_dynamic_covariates, valid_dynamic_covariates)
    ]


    return train_series, train_dynamic_covariates, valid_series, valid_dynamic_covariates,combined_series,combined_dynamic_covariates
def scale_series_and_covariates(train_series, train_covariates, 
                                 valid_series, valid_covariates, 
                                 threshold):
    """
    Scale target and dynamic covariate series using Darts' Scaler.
    Each TimeSeries object gets its own scaler. Scaling is applied
    only if any value in the series exceeds the threshold.
    """
    scaled_train_series = []
    scaled_valid_series = []
    series_scalers = []

    for trn, val in zip(train_series, valid_series):
        scaler = Scaler()
        trn_scaled = scaler.fit_transform(trn)
        val_scaled = scaler.transform(val)


        scaled_train_series.append(trn_scaled)
        scaled_valid_series.append(val_scaled)
        series_scalers.append(scaler)

    scaled_train_covariates = []
    scaled_valid_covariates = []
    covariate_scalers = []

    for trn_cov, val_cov in zip(train_covariates, valid_covariates):
        components = trn_cov.components
        scaled_components_train = []
        scaled_components_valid = []
        scalers = {}

        for comp in components:
            trn_comp = trn_cov[comp]
            val_comp = val_cov[comp]

            if trn_comp.all_values().max() > threshold:
                scaler = Scaler()
                trn_scaled = scaler.fit_transform(trn_comp)
                val_scaled = scaler.transform(val_comp)
                scalers[comp] = scaler
            else:
                trn_scaled = trn_comp
                val_scaled = val_comp
                scalers[comp] = None

            scaled_components_train.append(trn_scaled)
            scaled_components_valid.append(val_scaled)

        # Merge scaled components back into multivariate TimeSeries
        scaled_train_covariates.append(TimeSeries.from_times_and_values(trn_cov.time_index, 
                                                                        np.stack([comp.values() for comp in scaled_components_train], axis=-1)))
        scaled_valid_covariates.append(TimeSeries.from_times_and_values(val_cov.time_index, 
                                                                        np.stack([comp.values() for comp in scaled_components_valid], axis=-1)))

        covariate_scalers.append(scalers)
    all_series_scaled = [
        trn.concatenate(tst, ignore_time_axis=True)
        for trn, tst in zip(scaled_train_series, scaled_valid_series)
    ]
    
    all_dynamic_covariates_scaled = [
        trn_cov.concatenate(tst_cov, ignore_time_axis=True)
        for trn_cov, tst_cov in zip(scaled_train_covariates, scaled_valid_covariates)
    ]

    return (scaled_train_series, scaled_train_covariates,
            scaled_valid_series, scaled_valid_covariates,
            series_scalers, covariate_scalers,all_series_scaled,all_dynamic_covariates_scaled)