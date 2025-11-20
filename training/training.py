# %%
#import sys
#import os
#sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import argparse
#from utils.normalizing import normalize_csv
from utils.chemberta_workflows import train_chemberta_multilabel_model
from utils.chemberta_workflows_copy import grid_search_gamma_alpha, get_val_probs_and_labels, \
    find_best_global_threshold, get_test_probs_and_labels, find_best_thresholds_per_label

from sklearn.metrics import f1_score

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
        'output_dir': f'../trained_models/',
        'epochs': 60,
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
   #smell_mlc_results, f1_macro = train_chemberta_multilabel_model(smell_mlc_parser, train, test, val, threshold=threshold, gamma = gamma, alpha = alpha)
    #print(smell_mlc_results)
    #return f1_macro

    # 1) First: grid search gamma and alpha with a fixed threshold (e.g. 0.25)
    results, best_output = grid_search_gamma_alpha(
        args=smell_mlc_parser,
        df_train=train,
        df_test=test,  # used only for reporting inside that function
        df_val=val,
        gammas=[0.75],
        alphas=[None, 0.25],
        threshold=0.25,  # fixed during grid search
    )

    # 2) Get validation predictions for the best (gamma, alpha) model
    val_probs, val_labels = get_val_probs_and_labels(smell_mlc_parser, val, best_output)

    # 3A) Find best global threshold on validation
    #best_t, best_val_f1 = find_best_global_threshold(val_probs, val_labels, metric_average="macro")

    # (or 3B) For per-label thresholds:
    best_thresholds = find_best_thresholds_per_label(val_probs, val_labels)

    # 4) Final evaluation on the test set using the chosen threshold(s)
    test_probs, test_labels = get_test_probs_and_labels(smell_mlc_parser, test, best_output)

    # 4A) Using global threshold:
    #test_preds = (test_probs >= best_t).astype(int)

    # 4B) Using per-label thresholds:
    test_preds = (test_probs >= best_thresholds).astype(int)

    # 5) Compute final test metrics
    test_macro_f1 = f1_score(test_labels, test_preds, average="macro", zero_division=0)
    test_micro_f1 = f1_score(test_labels, test_preds, average="micro", zero_division=0)

    print(f"Final TEST macro-F1 (with tuned gamma, alpha, threshold): {test_macro_f1:.4f}")
    print(f"Final TEST micro-F1: {test_micro_f1:.4f}")

