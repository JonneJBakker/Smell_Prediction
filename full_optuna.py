# optuna_tune_chemberta.py
import os
import json
from types import SimpleNamespace

import optuna
import pandas as pd

from utils.chemberta_workflows import train_chemberta_multilabel_model


RANDOM_SEED = 123124

TRAIN_CSV = "Data/splits/train_stratified80.csv"
VAL_CSV   = "Data/splits/val_stratified10.csv"
TEST_CSV  = "Data/splits/test_stratified10.csv"


def load_splits():
    df_train = pd.read_csv(TRAIN_CSV)
    df_val = pd.read_csv(VAL_CSV)
    df_test = pd.read_csv(TEST_CSV)
    return df_train, df_val, df_test


def infer_smiles_and_targets(df: pd.DataFrame):
    smiles_col = "nonStereoSMILES" if "nonStereoSMILES" in df.columns else df.columns[0]
    target_cols = [
        c for c in df.columns
        if c != smiles_col and pd.api.types.is_numeric_dtype(df[c])
    ]
    if not target_cols:
        raise ValueError("No numeric target columns found. Check your dataset format.")
    return smiles_col, target_cols


def build_args(trial: optuna.Trial, smiles_col: str, target_cols: list[str]) -> SimpleNamespace:

    pooling_strat = trial.suggest_categorical(
        "pooling_strat", ["cls", "mean", "max", "cls_mean", "mean_max", "cls_max"]
    )

    loss_type = trial.suggest_categorical("loss_type", ["bce", "focal"])
    gamma = 0.0
    alpha = None
    if loss_type == "focal":
        gamma = trial.suggest_float("gamma", 1.75, 2.5)
        # keep alpha scalar; if you later use per-label alpha, extend this
        alpha = trial.suggest_float("alpha", 0.1, 0.5)

    # Optim / training
    lr = 0.0005
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
    weight_decay = trial.suggest_float("weight_decay", 0.05, 0.15)
    epochs = 40

    # Head
    dropout = trial.suggest_float("dropout", 0.0, 0.5)
    hidden_channels = trial.suggest_categorical("hidden_channels", [64, 128, 256, 512])
    num_mlp_layers = trial.suggest_int("num_mlp_layers", 1, 2)

    # Metric threshold
    threshold = trial.suggest_float("threshold", 0.25, 0.4)

    # Lora
    lora_r = trial.suggest_categorical("lora_r", [8, 16, 32])
    lora_alpha = trial.suggest_categorical("lora_alpha", [8, 16, 32, 64])
    lora_dropout = trial.suggest_float("lora_dropout", 0.0, 0.15)

    # Output directory per trial
    out_dir = os.path.join("optuna_runs", f"trial_{trial.number}")

    return SimpleNamespace(
        # required by workflow
        smiles_column=smiles_col,
        target_columns=target_cols,
        output_dir=out_dir,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        dropout=dropout,
        hidden_channels=hidden_channels,
        num_mlp_layers=num_mlp_layers,
        pooling_strat=pooling_strat,
        threshold=threshold,
        random_seed=RANDOM_SEED,

        # loss-related
        loss_type=loss_type,
        gamma=gamma,
        alpha=alpha,

        # Lora
        use_lora=True,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
    )


def objective(trial: optuna.Trial, df_train: pd.DataFrame, df_val: pd.DataFrame, smiles_col: str, target_cols: list[str]) -> float:
    args = build_args(trial, smiles_col, target_cols)

    # Ensure trial output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Run training; evaluate on validation
    # train_chemberta_multilabel_model returns (metrics_dict, test_macro_f1)
    metrics, val_macro_f1 = train_chemberta_multilabel_model(
        args=args,
        df_train=df_train,
        df_val=df_val,
        df_test=df_val,
        device=None,
    )

    score = float(val_macro_f1)
    return score


def main():
    os.makedirs("optuna_runs", exist_ok=True)

    df_train, df_val, df_test = load_splits()
    smiles_col, target_cols = infer_smiles_and_targets(df_train)

    storage = None

    study = optuna.create_study(
        study_name="chemberta_lora_tuning",
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
        storage=storage,
        load_if_exists=bool(storage),
    )

    study.optimize(
        lambda t: objective(t, df_train=df_train, df_val=df_val, smiles_col=smiles_col, target_cols=target_cols),
        n_trials=50,
        show_progress_bar=True,
    )

    print("\nBest validation macro-F1:", study.best_value)
    print("Best params:\n", study.best_params)

    with open("optuna_best_params.json", "w") as f:
        json.dump({"best_value": study.best_value, "best_params": study.best_params}, f, indent=2)



if __name__ == "__main__":
    main()
