import os
import argparse

import optuna
import pandas as pd
import torch

# adjust the import path if needed (this matches your error traces)
from utils.chemberta_workflows_cli_loss import train_chemberta_multilabel_model  # noqa



def parse_args():
    parser = argparse.ArgumentParser(description="Optuna tuning for ChemBERTa + CIL")

    parser.add_argument("--train_csv", type=str, required=True,
                        help="Path to training CSV (used as training data).")
    parser.add_argument("--val_csv", type=str, required=True,
                        help="Path to validation CSV (used as validation/test during tuning).")

    parser.add_argument("--smiles_column", type=str, required=True,
                        help="Name of the SMILES column.")
    parser.add_argument(
        "--target_columns",
        type=str,
        nargs="+",
        required=False,
        help="List of label columns. If omitted, all columns except smiles_column are used."
    )

    parser.add_argument("--output_dir", type=str, default="optuna_runs",
                        help="Base directory where trial outputs will be stored.")
    parser.add_argument("--n_trials", type=int, default=20,
                        help="Number of Optuna trials.")
    parser.add_argument("--seed", type=int, default=123,
                        help="Random seed for reproducibility.")

    return parser.parse_args()


class ArgsForTraining:
    """
    Simple container that mimics the `args` object expected by
    train_chemberta_multilabel_model in chemberta_workflows_cli_loss.py
    """
    def __init__(self):
        # filled from CLI + Optuna suggestions
        self.smiles_column = None
        self.target_columns = None

        self.train_csv = None
        self.output_dir = None

        self.epochs = None
        self.batch_size = None
        self.lr_encoder = None
        self.lr_head = None
        self.l2_lambda = None
        self.l1_lambda = 0.0

        self.lambda1 = None
        self.lambda2 = None
        self.lambda3 = None
        self.lambda4 = None
        self.c = None

        self.dropout = None
        self.hidden_channels = None
        self.num_mlp_layers = None
        self.random_seed = None
        self.unfreeze_last_n_layers = None

def make_objective(cli_args):
    """
    Build the Optuna objective function, capturing command-line args.
    """
    # Load data once outside the trials to avoid repeated disk I/O
    df_train = pd.read_csv(cli_args.train_csv)
    df_val = pd.read_csv(cli_args.val_csv)

    def objective(trial: optuna.Trial) -> float:
        # ----- Suggest hyperparameters -----
        epochs = trial.suggest_int("epochs", 10, 40)
        batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
        lr_encoder = trial.suggest_float("lr_encoder", 1e-6, 5e-5, log=True)
        lr_head = trial.suggest_float("lr_head", 5e-5, 5e-3, log=True)
        weight_decay = trial.suggest_float("l2_lambda", 1e-6, 1e-2, log=True)

        dropout = trial.suggest_float("dropout", 0.1, 0.5)
        hidden_channels = trial.suggest_categorical("hidden_channels", [64, 128, 256, 384])
        num_mlp_layers = trial.suggest_int("num_mlp_layers", 1, 2)
        unfreeze_last_n_layers = trial.suggest_categorical(
            "unfreeze_last_n_layers", [0, 1, 2, 3]
        )

        threshold = trial.suggest_float("threshold", 0.15, 0.35)

        lambda1 = trial.suggest_float("lambda1", 0.1, 0.6)
        lambda2 = trial.suggest_float("lambda2", 0.1, 0.6)
        lambda3 = trial.suggest_float("lambda3", 0.3, 1.0)
        lambda4 = trial.suggest_float("lambda4", 0.1, 0.6)
        c = trial.suggest_float("c", 0.1, 0.3)

        # ----- Build args object for training -----
        args = ArgsForTraining()
        args.smiles_column = cli_args.smiles_column
        args.target_columns = cli_args.target_columns

        args.train_csv = cli_args.train_csv
        args.output_dir = os.path.join(cli_args.output_dir, f"trial_{trial.number}")

        args.epochs = epochs
        args.batch_size = batch_size
        args.lr_encoder = lr_encoder
        args.lr_head = lr_head
        args.l2_lambda = weight_decay
        args.l1_lambda = 0.0

        args.lambda1 = lambda1
        args.lambda2 = lambda2
        args.lambda3 = lambda3
        args.lambda4 = lambda4
        args.c = c
        args.dropout = dropout
        args.hidden_channels = hidden_channels
        args.num_mlp_layers = num_mlp_layers
        args.random_seed = cli_args.seed
        args.unfreeze_last_n_layers = unfreeze_last_n_layers

        os.makedirs(args.output_dir, exist_ok=True)

        # ----- Run a single training run -----
        # IMPORTANT: for hyperparameter tuning we treat `df_val`
        # as both the evaluation and "test" set inside the training function.
        # Your true test set should be kept separate and only used after tuning.
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        _, val_macro_f1 = train_chemberta_multilabel_model(
            args=args,
            df_train=df_train,
            df_test=df_val,   # used for evaluation + per-label metrics
            df_val=df_val,    # used as eval_dataset during training
            device=device,
            threshold=threshold,
        )

        # Optuna will try to maximize this
        return float(val_macro_f1)

    return objective


def main():
    cli_args = parse_args()
    os.makedirs(cli_args.output_dir, exist_ok=True)

    if cli_args.target_columns is None:
        df_tmp = pd.read_csv(cli_args.train_csv, nrows=1)
        all_cols = df_tmp.columns.tolist()
        if cli_args.smiles_column not in all_cols:
            raise ValueError(
                f"SMILES column '{cli_args.smiles_column}' not found in {cli_args.train_csv}. "
                f"Available columns: {all_cols}"
            )

        cli_args.target_columns = [c for c in all_cols if c != cli_args.smiles_column]
        print("Auto-detected target columns (odor descriptors):")
        print(cli_args.target_columns)
    # to make Optuna itself reproducible (doesn't guarantee full determinism in HF + CUDA)
    sampler = optuna.samplers.TPESampler(seed=cli_args.seed)

    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        study_name="chemberta_cil_tuning",
    )

    objective = make_objective(cli_args)
    study.optimize(objective, n_trials=cli_args.n_trials)

    print("\n===== Optuna finished =====")
    print(f"Best trial number: {study.best_trial.number}")
    print(f"Best value (macro F1 on val): {study.best_trial.value:.4f}")
    print("Best params:")
    for k, v in study.best_trial.params.items():
        print(f"  {k}: {v}")

    # Save study to a CSV for inspection
    df_trials = study.trials_dataframe()
    csv_path = os.path.join(cli_args.output_dir, "optuna_trials.csv")
    df_trials.to_csv(csv_path, index=False)
    print(f"\nSaved all trial results to {csv_path}")


if __name__ == "__main__":
    main()
