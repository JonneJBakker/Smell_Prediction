import pandas as pd
from sklearn.model_selection import train_test_split

RANDOM_SEED = 123124

def split_data(df, smiles_col):

    df = df.drop(columns=[column for column in df.columns.to_list() if
                              column in ["descriptors"]])
    df = df.rename(columns={smiles_col: "smiles"})

    not_chosen_columns = ['smiles']

    label_cols = [col for col in df.columns if col not in not_chosen_columns]

    x_train, x_test, y_train, y_test = train_test_split(df['smiles'], df[label_cols], test_size=0.2, random_state=RANDOM_SEED)
    x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=0.2, random_state=RANDOM_SEED)
    return x_train, x_val, x_test, y_train, y_val, y_test,  label_cols


