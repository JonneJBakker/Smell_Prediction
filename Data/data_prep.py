import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from rdkit import Chem
from rdkit import RDLogger

RANDOM_SEED = 123124



def split_data(df, smiles_col):

    df = df.drop(columns=[column for column in df.columns.to_list() if
                              column in ["descriptors"]])
    df = df.rename(columns={smiles_col: "smiles"})

    not_chosen_columns = ['smiles']

    label_cols = [col for col in df.columns if col not in not_chosen_columns]

    smiles_train, smiles_test, labels_train, labels_test = train_test_split(df['smiles'], df[label_cols], test_size=0.2, random_state=RANDOM_SEED)
    smiles_val, smiles_test, labels_val, labels_test = train_test_split(smiles_test, labels_test, test_size=0.2, random_state=RANDOM_SEED)

    smiles_train.to_csv("Data/splits/smiles_train.csv", index=False)
    smiles_test.to_csv("Data/splits/smiles_test.csv", index=False)
    smiles_val.to_csv("Data/splits/smiles_val.csv", index=False)
    labels_train.to_csv("Data/splits/labels_train.csv", index=False)
    labels_test.to_csv("Data/splits/labels_test.csv", index=False)
    labels_val.to_csv("Data/splits/labels_val.csv", index=False)

    #return smiles_train, smiles_val, smiles_test, labels_train, labels_val, labels_test,  label_cols


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


def augment_train_csv(smiles_train_path, labels_train_csv_path, out_csv_path, smiles_col="smiles", k=5, percentile=80.0, keep_original=True):

    smiles_train = pd.read_csv(smiles_train_path)
    labels_train = pd.read_csv(labels_train_csv_path)

    df = pd.concat([smiles_train, labels_train], axis=1)

    print(df)

    label_cols = [c for c in df.columns if c != smiles_col]

    #check for rare labels
    prevalences = labels_train.mean(axis=0)
    cutoff = np.percentile(prevalences.values, percentile)
    rare_labels = [c for c, p in prevalences.items() if p >= cutoff]
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