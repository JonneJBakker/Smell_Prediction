import numpy as np
import pandas as pd
from numpy.f2py.cfuncs import commonhooks
from sklearn.model_selection import train_test_split
from rdkit import Chem
from rdkit import RDLogger
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

RANDOM_SEED = 123124



def split_data(df, smiles_col):

    df = df.drop(columns=[column for column in df.columns.to_list() if
                              column in ["descriptors"]])
    df = df.rename(columns={smiles_col: "smiles"})

    not_chosen_columns = ['smiles']

    label_cols = [col for col in df.columns if col not in not_chosen_columns]

    train, test= train_test_split(df, test_size=0.2, random_state=RANDOM_SEED)

    train.to_csv("Data/splits/train.csv", index=False)
    test.to_csv("Data/splits/test.csv", index=False)


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

    # 5) Optional: quick sanity check of label frequencies
    print("Label means per split (per label):")
    print("  train:", df_train[target_cols].mean(numeric_only=True).to_dict())
    print("  val  :", df_val[target_cols].mean(numeric_only=True).to_dict())
    print("  test :", df_test[target_cols].mean(numeric_only=True).to_dict())

    df_train.to_csv("Data/splits/train_stratified.csv", index=False)
    df_val.to_csv("Data/splits/val_stratified.csv", index=False)
    df_test.to_csv("Data/splits/test_stratified.csv", index=False)

def valid_smiles(s: str) -> bool:
    if not isinstance(s, str): return False
    s = s.strip()
    if not s or s.lower() in {"nan", "none", "null", "na", "n/a"}:
        return False
    return Chem.MolFromSmiles(s) is not None

def randomize_smiles(smi, max_tries=20):
    """Return ONE randomized SMILES string for a valid molecule."""
    if not valid_smiles(smi):
        return None

    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    for _ in range(max_tries):
        try:
            try:
                rs = Chem.MolToSmiles(mol, canonical=False, doRandom=True, randomSeed=RANDOM_SEED)
            except TypeError:
                rs = Chem.MolToSmiles(mol, canonical=False, doRandom=True)
            if rs and rs != "":
                return rs
        except Exception:
            continue
    return None


def augment_train_csv(train_path, out_csv_path, smiles_col="smiles", k=5, percentile=80.0, keep_original=True):

    df = pd.read_csv(train_path)

    label_cols = [c for c in df.columns if c != smiles_col]

    #check for rare labels
    prevalences = df[label_cols].mean(axis=0)
    cutoff = np.percentile(prevalences.values, percentile)
    rare_labels = [c for c, p in prevalences.items() if p < cutoff]
    print(rare_labels)

    rows = []
    for idx, row in df.iterrows():
        base = row.to_dict()
        base_smi = base[smiles_col]


        #  keep the original canonical form
        if keep_original:
            rows.append(base)

        # check if row has rare label
        is_rare_row = any((int(base[lbl]) == 1) for lbl in rare_labels)

        if not is_rare_row:
            # collect unique randomized variants for this molecule
            seen = set()
            tries = 0
            while len(seen) < k and tries < k * 10:
                tries += 1
                rs = randomize_smiles(base_smi)
                if rs and rs not in seen and rs != base_smi:
                    seen.add(rs)
                    aug = {**{lc: row[lc] for lc in label_cols}, smiles_col: rs}
                    rows.append(aug)

    out = pd.DataFrame(rows, columns=[smiles_col] + label_cols)
    out.to_csv(out_csv_path, index=False)
    return out