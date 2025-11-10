import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def normalize_csv(
    df: pd.DataFrame,
    target_col: str = "pce_1",
    scaler: StandardScaler = None,
    fit_scaler: bool = True,
):
    # Make a copy to avoid modifying the original dataframe
    df_copy = df.copy()
    
    if fit_scaler and not scaler:
        scaler = StandardScaler()
        # Reshape to 2D array for sklearn
        scaler.fit(df_copy[target_col].values.reshape(-1, 1))
    
    df_copy[target_col] = df_copy[target_col].astype(np.float64)
    # Transform also needs 2D array
    df_copy.loc[:, target_col] = scaler.transform(df_copy[target_col].values.reshape(-1, 1)).flatten()
    return df_copy, scaler

def inverse_transform(
    y, 
    scaler,
):
    return y * scaler.scale_ + scaler.mean_


