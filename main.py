# main.py

import argparse
import pandas as pd
import numpy as np
import mlflow
import os
from metrics import compute_metrics
from models import train_and_predict_models
from preprocess import load_and_prepare_data,removing_series_with_many_zeros,split_series,scale_series_and_covariates
from params import hyperparameters
from saving_and_plot import save_train_predictions_to_disk_single_csv,backtest
np.random.seed(42)





def main(args):
    mlflow.set_tracking_uri(None)

    experiment_name = f"{args.model_type}_forecasting"
    mlflow.set_experiment(experiment_name)
       # 3. Now start the run
    with mlflow.start_run(run_name=f"{args.model_type}_IC{args.input_chunk}_OC{args.output_chunk}"):
        mlflow.log_params(vars(args))
    #with mlflow.start_run():
        mlflow.set_tag("model_type", args.model_type)
        mlflow.set_tag("data_path", args.data_path)
        mlflow.set_tag("optimizer", args.optimizer)
        ### Loading and preprocessing the data
        df_train, df_valid = load_and_prepare_data(args.data_path, include_paydays=True,
                          include_sin_cos=True, include_lags=False,
                          include_rolling=False)
        
        ### --- Removing the series with too many 0s ---
        #df=removing_series_with_many_zeros(df,600)

        #static_features = ['region_idx', 'note_type_idx', 'demand_type_idx', 'denom_idx']
        static_features=['region_CGY', 'region_HFX',
       'region_MTL', 'region_NFD', 'region_QBC', 'region_REG', 'region_TOR',
       'region_VCR', 'region_WPG', 'note_type_FIT', 'note_type_New',
        'demand_type_D', 'demand_type_W','denom_10', 'denom_100', 'denom_20', 'denom_5', 'denom_50']
        
        dynamic_features = [col for col in df_train.columns if col not in static_features + ['series_id', 'date', 'demand']]

        
        ### --- Creating train valid and test series ---
        train_series, train_dynamic_covariates, valid_series, valid_dynamic_covariates,combined_series,combined_dynamic_covariates=split_series(df_train,
                                                                                                                                                df_valid,
                                                                                                                                                dynamic_features,
                                                                                                                                                static_features)
        

        ### Now calling the scalling function
        scaled_train_series, scaled_train_covariates,scaled_valid_series, scaled_valid_covariates,series_scalers, covariate_scalers,all_series_scaled,all_dynamic_covariates_scaled=scale_series_and_covariates(train_series, train_dynamic_covariates, 
                                 valid_series, valid_dynamic_covariates, 
                                 threshold=2026)
        ### Now based on the model name we get the model from the models.py file
        results_dir = f"results_{args.model_type}"
        os.makedirs(results_dir, exist_ok=True)

        hyperparams=hyperparameters(args)
        train_and_predict_models(args.model_type,scaled_train_series,scaled_train_covariates,
        scaled_valid_series,
        scaled_valid_covariates,
        series_scalers,hyperparams,results_dir)
        
        
 #=================================================================================================================
        #============== For Validation =======
        val_preds_unscaled, val_lower, val_upper,preds_rolled = backtest(all_series_scaled,all_dynamic_covariates_scaled,model_name=args.model_type,
        target_series=scaled_valid_series,
        dynamic_covariates=scaled_valid_covariates,
        scalers_target=series_scalers,
        args=args,
        start_offset_chunks=0  # start from beginning of validation  
)
        avg_scores = compute_metrics(
            preds_rolled,
            scaled_valid_series,
            series_scalers,
            scaled_train_series,
            
            metrics=["mae", "rmse", "mse", "smape", "r2", "rse"],
        )

        
        avg_scores["model"] = args.model_type
        for param_name, param_value in hyperparams.items():
            avg_scores[param_name] = param_value

        results_path = os.path.join(results_dir, "metrics.csv")
        df_scores = pd.DataFrame([avg_scores])
        if not os.path.exists(results_path):
            df_scores.to_csv(results_path, index=False)
        else:
            df_scores.to_csv(results_path, mode='a', header=False, index=False)
         #============== For Training =======
        train_preds_unscaled, train_lower, train_upper,preds_rolled = backtest(all_series_scaled,all_dynamic_covariates_scaled,
            model_name=args.model_type,
            target_series=scaled_train_series,
            dynamic_covariates=scaled_train_covariates,
            scalers_target=series_scalers,
            args=args,
            start_offset_chunks=1  # typical for training
        )
            
        

        save_train_predictions_to_disk_single_csv(
        preds_unscaled=val_preds_unscaled,
        actuals_unscaled=valid_series,
        results_dir=f"results_{args.model_type}",
        filename="valid",
        lower_bounds=val_lower,
        upper_bounds=val_upper)

        mlflow.end_run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # === General ===
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--model_type', type=str, default='deeptcn',
                        help='Type of model to use: deeptcn, lstm, nbeats, etc.')
    parser.add_argument('--input_chunk', type=int, default=52)
    parser.add_argument('--output_chunk', type=int, default=8)
    parser.add_argument('--valid_size', type=int, default=20)

    # === Training / Optimization ===
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--loss', type=str, default='mse')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--n_epochs', type=int, default=150)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--nr_epochs_val_period', type=int, default=1)

    # === DeepTCN-specific ===
    parser.add_argument('--kernel_size', type=int, default=3)
    parser.add_argument('--num_filters', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=8)

    # === LSTM-specific ===
    parser.add_argument('--lstm_units', type=int, default=64)
    parser.add_argument('--n_lstm_layers', type=int, default=1)

    # === NBEATS-specific ===
    parser.add_argument('--num_stacks', type=int, default=2)
    parser.add_argument('--num_blocks', type=int, default=2)
    parser.add_argument('--layer_widths', type=int, default=512)
    parser.add_argument('--generic_architecture', action='store_true',
                        help='If set, use generic architecture for NBEATS')

    # === Inference / Evaluation ===
    parser.add_argument('--num_samples', type=int, default=500)

    # === Misc ===
    parser.add_argument('--use_layer_norm', action='store_true',
                        help='Enable layer normalization in the model')

    args = parser.parse_args()
    main(args)