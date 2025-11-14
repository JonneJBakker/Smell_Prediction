# %%
#import sys
#import os
#sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import argparse
from utils.normalizing import normalize_csv
from utils.chemberta_workflows import train_chemberta_multilabel_model
# %%
RANDOM_SEED = 19237
# %%

# %%
def train_mlc():
    train = pd.read_csv("Data/splits/train_stratified80.csv")
    test = pd.read_csv("Data/splits/test_stratified10.csv")
    val = pd.read_csv("Data/splits/val_stratified10.csv")

    target_cols = [col for col in train.columns if col not in ['nonStereoSMILES']]
    # Make an args parser
    smell_mlc_defaults = {
        'train_csv': '../Data/splits/train_stratified80.csv',
        'test_csv': '../Data/splits/test_stratified10.csv',
        'target_columns': target_cols,
        'smiles_column': 'nonStereoSMILES',
        'output_dir': '../trained_models',
        'epochs': 120,
        'batch_size': 16,
        'lr': 0.001,
        'l1_lambda': 0.0,
        'l2_lambda': 0.01,
        'dropout': 0.3,
        'hidden_channels': 128,
        'num_mlp_layers': 1,
        'random_seed': RANDOM_SEED,
    }

    smell_mlc_parser = argparse.Namespace(**smell_mlc_defaults)

    # %%
    smell_mlc_results = train_chemberta_multilabel_model(smell_mlc_parser, train, test, val)
    print(smell_mlc_results)
