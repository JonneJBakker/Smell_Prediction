# %%
#import sys
#import os
#sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import copy
from types import SimpleNamespace

import pandas as pd
import argparse
#from utils.normalizing import normalize_csv
from utils.chemberta_workflows import train_chemberta_multilabel_model
#from utils.molformer_workflows import train_molformer_multilabel_model
#from utils.make_pom import plot_pca
#from utils.contrastie_loss import train_chemberta_multilabel_model
#from utils.chemberta_workflows_copy import grid_search_gamma_alpha, get_val_probs_and_labels, \
    #find_best_global_threshold, get_test_probs_and_labels, find_best_thresholds_per_label
#from utils.chemberta_workflows_cli_loss import train_chemberta_multilabel_model
from sklearn.metrics import f1_score


# %%
RANDOM_SEED = 42
# %%

# %%
def train_mlc():
    train = pd.read_csv("Data/splits/train_stratified80.csv")
    test = pd.read_csv("Data/splits/test_stratified10.csv")
    val = pd.read_csv("Data/splits/val_stratified10.csv")
    target_cols = [col for col in train.columns if col not in ['nonStereoSMILES']]


    final_args_frozen = SimpleNamespace(
        train_csv='../Data/splits/train_stratified80.csv',
        test_csv='../Data/splits/test_stratified10.csv',
        smiles_column="nonStereoSMILES",
        target_columns=target_cols,
        output_dir=f'../trained_models/',

        epochs=40,
        batch_size=8,
        lr=0.0005,
        l2_lambda=0.06710588932518757,
        pooling_strat="mean",
        loss_type="focal",
        gamma=2.0984128550461207,
        alpha=0.1915039083259983,
        weight_decay=0.06710588932518757,
        dropout=0.08076239608280959,
        hidden_channels=256,
        num_mlp_layers=2,
        threshold=0.35,

        use_lora = False,
        lora_r=16,
        lora_alpha=16,
        lora_dropout=0.1,
        random_seed=42,
    )

    lora_params = SimpleNamespace(
        train_csv='../Data/splits/train_stratified80.csv',
        test_csv='../Data/splits/test_stratified10.csv',
        smiles_column="nonStereoSMILES",
        target_columns=target_cols,
        output_dir=f'../trained_models/',
        epochs=40,
        lr=0.0005,
        pooling_strat="mean",
        loss_type="focal",
        gamma=1.8183167530341804,
        alpha=0.2781964441223481,
        batch_size=8,
        weight_decay=0.10108776189661574,
        dropout=0.17426042486882132,
        hidden_channels=256,
        num_mlp_layers=2,
        threshold=0.35,
        use_lora = True,
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.0614101620783747,
        random_seed=42,
    )


    # LORA NOW GOOD OPTUNA
    best_params = SimpleNamespace(
        train_csv='../Data/splits/train_stratified80.csv',
        test_csv='../Data/splits/test_stratified10.csv',
        smiles_column="nonStereoSMILES",
        target_columns=target_cols,
        output_dir=f'../trained_models/',
        epochs=50,
        lr=0.0005,
        pooling_strat="cls_mean",
        loss_type="focal",
        gamma=1.9167984440970527,
        alpha=0.1612449232980698,
        batch_size=16,
        weight_decay=0.056757176895569576,
        dropout=0.29820017391677023,
        hidden_channels=512,
        num_mlp_layers=1,
        threshold=0.3636917584461052,
        use_lora = True,
        lora_r=8,
        lora_alpha=64,
        lora_dropout=0.08509500422040107,
        random_seed=42,
    )

    #smell_mlc_parser = argparse.Namespace(**asym_loss_best)
    pooling_strats = ["mean", "max", "cls", "cls_mean", "cls_max", "mean_max"]

    for p in pooling_strats:
        args = copy.deepcopy(best_params)
        args.pooling_strat = p


        print(f"\n=== Training with pooling_strat={p} ===")
        result = train_chemberta_multilabel_model(args, df_train=train, df_test=test, df_val=val)

        print(p)

