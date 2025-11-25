import os
import argparse

import optuna
import pandas as pd
import torch

from utils.chemberta_workflows import train_chemberta_multilabel_model


def parse_args():
    parser = argparse.ArgumentParser(description="Optuna tuning for ChemBERTa + Focal Loss on POM")

    parser.add_argument("--train_csv", type=str, required=True,
                        help="Path to training CSV (e.g. train_stratified80.csv).")
    parser.add_argument("--val_csv", type=str, required=True,
                        help="Path to validation CSV (e.g. val_stratified20.csv).")

    parser.add_argument("--smiles_column", type=str, required=True,
                        help="Name of the SMILES column, e.g. 'nonStereoSMILES'.")

    # Optional: infer target columns if not given
    parser.add_argument(
        "--target_columns",
        type=str,
        nargs="+",
        required=False,
        help="List of label columns. If omitted, all columns except smiles_column are used."
    )

    parser.add_argument("--output_dir", type=str, default="optuna_runs_focal",
                        help="Base directory where trial outputs will be stored.")
    parser.add_argument("--n_trials", type=int, default=20,
                        help="Number of Optuna trials.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility.")

    return parser.parse_args()


class ArgsForTraining:
    """
    Minimal args object to satisfy train_chemberta_multilabel_model()
    from chemberta_workflows.py
    """
    def __init__(self):
        self.smiles_column = None
        self.target_columns = None

        self.train_csv = None
        self.output_dir = None

        self.epochs = 40
        self.batch_size = None
        self.lr = None
        self.l2_lambda = None
        self.l1_lambda = 0.0

        self.dropout = None
        self.hidden_channels = None
        self.num_mlp_layers = None
        self.random_seed = None

        # NEW: focal loss + pooling hyperparameters
        self.gamma = None
        self.alpha = None
        self.pooling_strat = None   # "mean_pooling", "cls", or "attention"


def make_objective(cli_args):
    """
    Build Optuna objective for the focal-loss ChemBERTa baseline.
    """

    # Load once outside trials to avoid repeated disk I/O
    df_train = pd.read_csv(cli_args.train_csv)
    df_val = pd.read_csv(cli_args.val_csv)

    def objective(trial: optuna.Trial) -> float:
        # ----- Hyperparameter search space -----
        batch_size = trial.suggest_categorical("batch_size", [16, 32])
        lr = trial.suggest_float("lr", 0.0005, 0.003, log=True)
        weight_decay = trial.suggest_float("l2_lambda", 0.005, 0.02, log=True)

        dropout = trial.suggest_float("dropout", 0.1, 0.5)
        hidden_channels = trial.suggest_categorical("hidden_channels", [64, 128, 256, 384])
        num_mlp_layers = trial.suggest_int("num_mlp_layers", 1, 2)

        # classification threshold
        threshold = trial.suggest_float("threshold", 0.15, 0.35)

        # focal loss hyperparameters
        gamma = trial.suggest_float("gamma", 0.5, 3.0)
        alpha = trial.suggest_float("alpha", 0.0, 1.0)  # scalar alpha ∈ [0,1]

        # pooling strategy: mean, CLS, or attention
        pooling_strat = trial.suggest_categorical(
            "pooling_strat",
            ["cls_mean", "max_mean"],
        )

        # ----- Build args expected by train_chemberta_multilabel_model -----
        args = ArgsForTraining()
        args.smiles_column = cli_args.smiles_column
        args.target_columns = cli_args.target_columns

        args.train_csv = cli_args.train_csv
        args.output_dir = os.path.join(cli_args.output_dir, f"trial_{trial.number}")

        args.epochs = 40
        args.batch_size = batch_size
        args.lr = lr
        args.l2_lambda = weight_decay
        args.l1_lambda = 0.0

        args.dropout = dropout
        args.hidden_channels = hidden_channels
        args.num_mlp_layers = num_mlp_layers
        args.random_seed = cli_args.seed

        args.gamma = gamma
        args.alpha = alpha
        args.pooling_strat = pooling_strat

        os.makedirs(args.output_dir, exist_ok=True)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # We treat df_val as both eval and "test" for tuning.
        # train_chemberta_multilabel_model returns (results_dict, macro_f1).
        _, val_macro_f1 = train_chemberta_multilabel_model(
            args=args,
            df_train=df_train,
            df_test=df_val,
            df_val=df_val,
            device=device,
            threshold=threshold,
            gamma=gamma,
            alpha=alpha,
            # we'll wire pooling_strat inside chemberta_workflows.py
        )

        # Optuna will maximize this value
        return float(val_macro_f1)

    return objective


def main():
    cli_args = parse_args()
    os.makedirs(cli_args.output_dir, exist_ok=True)

    # Auto-detect target columns if not provided: everything except SMILES column
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

    sampler = optuna.samplers.TPESampler(seed=cli_args.seed)
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        study_name="chemberta_focal_tuning",
    )

    objective = make_objective(cli_args)
    study.optimize(objective, n_trials=cli_args.n_trials, catch=(ValueError,))

    print("\n===== Optuna finished (Focal Loss baseline) =====")
    print(f"Best trial number: {study.best_trial.number}")
    print(f"Best value (macro F1 on val): {study.best_trial.value:.4f}")
    print("Best params:")
    for k, v in study.best_trial.params.items():
        print(f"  {k}: {v}")

    # Save all trial results to CSV
    df_trials = study.trials_dataframe()
    csv_path = os.path.join(cli_args.output_dir, "optuna_trials_focal.csv")
    df_trials.to_csv(csv_path, index=False)
    print(f"\nSaved all trial results to {csv_path}")


if __name__ == "__main__":
    main()
