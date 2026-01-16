import numpy as np
import pandas as pd
from numpy.f2py.cfuncs import commonhooks
from sklearn.model_selection import train_test_split
from rdkit import Chem
from rdkit import RDLogger
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

RANDOM_SEED = 123124



def stratified_train_val_test_split(
    df: pd.DataFrame,
    smiles_col: str,
    target_cols=None,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = RANDOM_SEED,
):
    """
    Stratified multi-label train/val/test split on a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset with SMILES and label columns.
    smiles_col : str
        Name of the column containing SMILES strings.
    target_cols : list or None
        List of label column names. If None, will infer as all numeric columns
        except the SMILES column.
    test_size : float
        Fraction of data to use for the test set.
    val_size : float
        Fraction of data to use for the validation set (of the whole dataset).
    random_state : int
        Random seed.

    Returns
    -------
    df_train, df_val, df_test : pd.DataFrame
    """
    df = df.drop(columns=[column for column in df.columns.to_list() if
                          column in ["descriptors"]])

    # 1) Decide which columns are labels
    if target_cols is None:
        # infer: numeric columns except smiles_col
        numeric_cols = [
            c for c in df.columns
            if c != smiles_col and pd.api.types.is_numeric_dtype(df[c])
        ]
        target_cols = numeric_cols

    if len(target_cols) == 0:
        raise ValueError("No target columns found. Please pass target_cols explicitly.")

    # 2) Multi-label matrix for stratification
    Y = df[target_cols].values

    # 3) First split: train vs test
    msss = MultilabelStratifiedShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=random_state,
    )
    train_idx, test_idx = next(msss.split(df, Y))

    df_train_full = df.iloc[train_idx].reset_index(drop=True)
    df_test = df.iloc[test_idx].reset_index(drop=True)

    # 4) Second split: train vs val (inside train_full)
    Y_train_full = df_train_full[target_cols].values
    val_relative_size = val_size / (1.0 - test_size)

    msss = MultilabelStratifiedShuffleSplit(
        n_splits=1,
        test_size=val_relative_size,
        random_state=random_state,
    )
    train_idx, val_idx = next(msss.split(df_train_full, Y_train_full))

    df_train = df_train_full.iloc[train_idx].reset_index(drop=True)
    df_val = df_train_full.iloc[val_idx].reset_index(drop=True)

    print("Label means per split (per label):")
    print("  train:", df_train[target_cols].mean(numeric_only=True).to_dict())
    print("  val  :", df_val[target_cols].mean(numeric_only=True).to_dict())
    print("  test :", df_test[target_cols].mean(numeric_only=True).to_dict())

    df_train.to_csv("Data/splits/int_train_stratified80.csv", index=False)
    df_val.to_csv("Data/splits/int_val_stratified10.csv", index=False)
    df_test.to_csv("Data/splits/int_test_stratified10.csv", index=False)
