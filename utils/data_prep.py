import pandas as pd
from sklearn.model_selection import train_test_split

RANDOM_SEED = 123124

def split_data(df):

    df = df.drop(columns=[column for column in df.columns.to_list() if
                              column in ["descriptors"]])
    len_df = len(df)
    train_scent, test_scent = train_test_split(df, test_size=0.2, random_state=RANDOM_SEED)
    train_scent.to_csv("Data/train_Scent.csv", index=False)
    test_scent.to_csv("Data/test_Scent.csv", index=False)


