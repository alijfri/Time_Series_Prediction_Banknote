

def hyperparameters(args):
    return {
        "num_samples": args.num_samples,
        "model_type":args.model_type,
        "dropout": args.dropout,
        "n_epochs":args.n_epochs,
        "use_layer_norm": args.use_layer_norm,
        "optimizer": args.optimizer,
        "loss": args.loss,
        "lr": args.lr,
        "weight_decay": 1e-4 ,
        "lstm_units": args.lstm_units,          # optional for LSTM
        "n_lstm_layers": args.n_lstm_layers,     # optional for LSTM
        "input_chunk": args.input_chunk,
        "output_chunk": args.output_chunk,
        "kernel_size": args.kernel_size,
        "num_filters": args.num_filters,
        "num_layers": args.num_layers,
        "valid_size":args.valid_size,
        "num_stacks":args.num_stacks,
        "num_blocks":args.num_blocks,
        "layer_widths":args.layer_widths,
        "batch_size":args.batch_size,
        "generic_architecture":args.generic_architecture,
        "nr_epochs_val_period":args.nr_epochs_val_period
    }
