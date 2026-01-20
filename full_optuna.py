# optuna_tune_chemberta.py
import os
import json
from types import SimpleNamespace

import optuna
import pandas as pd

from utils.chemberta_workflows import train_chemberta_multilabel_model


RANDOM_SEED = 123124

# Update these if your paths differ
TRAIN_CSV = "Data/splits/int_train_stratified80.csv"
VAL_CSV   = "Data/splits/int_val_stratified10.csv"
TEST_CSV  = "Data/splits/int_test_stratified10.csv"


def load_splits():
    df_train = pd.read_csv(TRAIN_CSV)
    df_val = pd.read_csv(VAL_CSV)
    df_test = pd.read_csv(TEST_CSV)
    return df_train, df_val, df_test


def infer_smiles_and_targets(df: pd.DataFrame):
    # Most likely your column is "smiles"
    smiles_col = "smiles" if "smiles" in df.columns else df.columns[0]
    target_cols = [
        c for c in df.columns
        if c != smiles_col and pd.api.types.is_numeric_dtype(df[c])
    ]
    if not target_cols:
        raise ValueError("No numeric target columns found. Check your dataset format.")
    return smiles_col, target_cols


def build_args(trial: optuna.Trial, smiles_col: str, target_cols: list[str]) -> SimpleNamespace:
    # Pooling strategies supported by your workflow:
    # "cls", "mean", "max", "cls_mean", "mean_max", "attention"
    pooling_strat = trial.suggest_categorical(
        "pooling_strat", ["cls", "mean", "max", "cls_mean", "mean_max", "cls_max"]
    )

    # Loss choice
    loss_type = trial.suggest_categorical("loss_type", ["bce", "focal"])
    gamma = 0.0
    alpha = None
    if loss_type == "focal":
        gamma = trial.suggest_float("gamma", 1.75, 2.5)
        # keep alpha scalar; if you later use per-label alpha, extend this
        alpha = trial.suggest_float("alpha", 0.1, 0.5)

    # Optim / training
    lr = 0.001
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
    weight_decay = trial.suggest_float("weight_decay", 0.05, 0.15)
    epochs = 20

    # Head
    dropout = trial.suggest_float("dropout", 0.0, 0.5)
    hidden_channels = trial.suggest_categorical("hidden_channels", [64, 128, 256, 512])
    num_mlp_layers = trial.suggest_int("num_mlp_layers", 1, 2)

    # Metric threshold (used by compute_metrics in workflow)
    threshold = trial.suggest_float("threshold", 0.25, 0.4)

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
        l2_lambda=weight_decay,
        dropout=dropout,
        hidden_channels=hidden_channels,
        num_mlp_layers=num_mlp_layers,
        pooling_strat=pooling_strat,
        threshold=threshold,
        random_seed=RANDOM_SEED,

        # loss-related (workflow reads these via getattr)
        loss_type=loss_type,
        gamma=gamma,
        alpha=alpha,

        # ASL args exist in workflow; keep defaults unused
        gamma_pos=0.0,
        gamma_neg=4.0,
        asl_clip=0.0,
    )


def objective(trial: optuna.Trial, df_train: pd.DataFrame, df_val: pd.DataFrame, smiles_col: str, target_cols: list[str]) -> float:
    """
    IMPORTANT:
    Your workflow evaluates on df_test at the end.
    During Optuna, we pass df_val as df_test to avoid touching the real test split.
    """
    args = build_args(trial, smiles_col, target_cols)

    # Ensure trial output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Run training; evaluate on validation (passed as df_test)
    # train_chemberta_multilabel_model returns (metrics_dict, test_macro_f1)
    metrics, val_macro_f1 = train_chemberta_multilabel_model(
        args=args,
        df_train=df_train,
        df_val=df_val,
        df_test=df_val,   # <-- validation-as-test during tuning
        device=None,
    )

    # Prefer the explicit macro-F1 returned by evaluate_per_label_metrics
    score = float(val_macro_f1)

    # Optional: report to Optuna for pruning
    # (we only get one score per trial here, but this still works for some pruners)
    trial.report(score, step=0)
    if trial.should_prune():
        raise optuna.TrialPruned()

    return score


def main():
    os.makedirs("optuna_runs", exist_ok=True)

    df_train, df_val, df_test = load_splits()
    smiles_col, target_cols = infer_smiles_and_targets(df_train)

    # You can persist Optuna results with SQLite if you want:
    # storage = "sqlite:///optuna_chemberta.db"
    storage = None

    study = optuna.create_study(
        study_name="chemberta_frozen_mlp_tuning",
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

    # ---- Final retrain once on real test set (recommended) ----
    best = study.best_params
    final_args = SimpleNamespace(
        smiles_column=smiles_col,
        target_columns=target_cols,
        output_dir=os.path.join("final_runs", "chemberta_best"),
        epochs=50,
        batch_size=int(best["batch_size"]),
        lr=float(best["lr"]),
        l2_lambda=float(best["weight_decay"]),
        dropout=float(best["dropout"]),
        hidden_channels=int(best["hidden_channels"]),
        num_mlp_layers=int(best["num_mlp_layers"]),
        pooling_strat=best["pooling_strat"],
        threshold=float(best["threshold"]),
        random_seed=RANDOM_SEED,

        loss_type=best["loss_type"],
        gamma=float(best.get("gamma", 0.0)),
        alpha=(float(best["alpha"]) if best["loss_type"] == "focal" else None),

        gamma_pos=0.0,
        gamma_neg=4.0,
        asl_clip=0.0,
    )

    os.makedirs(final_args.output_dir, exist_ok=True)
    os.makedirs("final_runs", exist_ok=True)

    _, test_macro_f1 = train_chemberta_multilabel_model(
        args=final_args,
        df_train=df_train,
        df_val=df_val,
        df_test=df_test,  # <-- real test split only once here
        device=None,
    )
    print("\nFinal TEST macro-F1:", test_macro_f1)


if __name__ == "__main__":
    main()
