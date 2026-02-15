
import copy
from types import SimpleNamespace

import pandas as pd

from utils.chemberta_final_train_plus_saving import train_and_save_chemberta_multilabel_model
from utils.chemberta_workflows import train_chemberta_multilabel_model

RANDOM_SEED = 42



def train_mlc():
    train = pd.read_csv("Data/splits/train_stratified80.csv")
    test = pd.read_csv("Data/splits/test_stratified10.csv")
    val = pd.read_csv("Data/splits/val_stratified10.csv")
    target_cols = [col for col in train.columns if col not in ['nonStereoSMILES']]


    frozen_params = SimpleNamespace(
        train_csv='../Data/splits/train_stratified80.csv',
        test_csv='../Data/splits/test_stratified10.csv',
        smiles_column="nonStereoSMILES",
        target_columns=target_cols,
        output_dir=f'../trained_models/',
        epochs=70,
        lr=0.00025,
        pooling_strat='cls_mean',
        loss_type='focal',
        gamma=2.1915795902136392,
        alpha=0.3521229430603956,
        batch_size=8,
        weight_decay=0.113858788969263,
        dropout=0.10269007352639603,
        hidden_channels=256,
        num_mlp_layers=2,
        threshold=0.3579330101884744,
        use_lora = False,
        lora_r = 8,
        lora_alpha = 64,
        lora_dropout = 0.08509500422040107,
        random_seed = 42,
    )

    # LORA NOW GOOD OPTUNA
    lora_params = SimpleNamespace(
        train_csv='../Data/splits/train_stratified80.csv',
        test_csv='../Data/splits/test_stratified10.csv',
        smiles_column="nonStereoSMILES",
        target_columns=target_cols,
        output_dir=f'../trained_models/',
        epochs=70,
        lr=0.00035,
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

    pooling_strats = ["mean", "max", "cls", "cls_mean", "cls_max", "mean_max"] #"mean", "max", "cls", "cls_mean", "cls_max", "mean_max"

    for p in pooling_strats:
        args = copy.deepcopy(frozen_params)
        args.pooling_strat = p


        print(f"\n=== Training with pooling_strat={p} ===")
        result = train_chemberta_multilabel_model(args, df_train=train, df_test=test, df_val=val)

        print(p)

    for p in pooling_strats:
        args = copy.deepcopy(lora_params)
        args.pooling_strat = p

        print(f"\n=== Training with pooling_strat={p} ===")
        result = train_chemberta_multilabel_model(args, df_train=train, df_test=test, df_val=val)

        print(p)


    train_and_save_chemberta_multilabel_model(frozen_params, df_train=train, df_test=test, df_val=val)
    train_and_save_chemberta_multilabel_model(lora_params, df_train=train, df_test=test, df_val=val)
