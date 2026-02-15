# %% [markdown]
# # ChemBERTa × TransformerLens: mechanistic interpretability notebook
#
# **Goal** – Load a fine‑tuned ChemBERTa checkpoint, port its encoder into
# [TransformerLens](https://github.com/neelnanda‑io/TransformerLens) (TL),
# validate it is functionally the same as the original model, and run a
# round of mechanistic interpretability analyses and visualizations
# (neuron ablations and regression-lens probes).  
# This notebook is used for development. After a technique works, it is moved to
# an independent Python file in utils/ and imported to ensure modularity.
#
# %%
import copy
import hashlib
from pathlib import Path
import torch
import pandas as pd
import numpy as np
import os
import sys

import matplotlib.colors as mcolors
from mechanistic_interpretability.utils.chemberta_workflows_mlc import ChembertaMultiLabelClassifier
from mechanistic_interpretability.utils.lora_diff import rank_heads_by_attn_change_encoder
from mechanistic_interpretability.utils.tl_conversion import load_chemberta_models
from mechanistic_interpretability.utils.tl_validation import validate_conversion, test_prediction_equivalence
from mechanistic_interpretability.utils.tl_ablation import run_ablation_analysis_with_metrics, plot_ablation_metrics
from mechanistic_interpretability.utils.tl_regression import  run_classification_lens, \
    compare_molecule_groups_classification_lens, plot_top_bottom_labels_f1, \
    compute_f1_per_label_per_layer
from mechanistic_interpretability.utils.tl_regression import plot_cluster_layer_f1_lines

from mechanistic_interpretability.utils.tl_lora_ablation import (
    scale_lora_contribution,
)

# %%
# For Frozen Model
MODEL_PATH = "mechanistic_interpretability/trained_models/FROZEN_FINAL/chemberta_multilabel_model_final.bin"
ADAPTER_DIR = "mechanistic_interpretability/trained_models/FROZEN_FINAL/"
MERGED_MODEL_PATH = "mechanistic_interpretability/trained_models/FROZEN_FINAL/chemberta_multilabel_model_final.bin"
FULL_PATH = "mechanistic_interpretability/clustered_data/pom/Multi-Labelled_Smiles_Odors_dataset.csv"
TEST_PATH = "mechanistic_interpretability/clustered_data/pom/test_stratified10.csv"
VAL_PATH = "mechanistic_interpretability/clustered_data/pom/val_stratified10.csv"
TRAIN_PATH = "mechanistic_interpretability/clustered_data/pom/train_stratified80.csv"
TOKENIZER_NAME = "DeepChem/ChemBERTa-77M-MLM"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
hyperparams_path = None
SCALER_PATH = None
TARGET_COLUMN = [
'alcoholic', 'aldehydic', 'alliaceous', 'almond', 'amber', 'animal',
'anisic', 'apple', 'apricot', 'aromatic', 'balsamic', 'banana', 'beefy',
'bergamot', 'berry', 'bitter', 'black currant', 'brandy', 'burnt',
'buttery', 'cabbage', 'camphoreous', 'caramellic', 'cedar', 'celery',
'chamomile', 'cheesy', 'cherry', 'chocolate', 'cinnamon', 'citrus', 'clean',
'clove', 'cocoa', 'coconut', 'coffee', 'cognac', 'cooked', 'cooling',
'cortex', 'coumarinic', 'creamy', 'cucumber', 'dairy', 'dry', 'earthy',
'ethereal', 'fatty', 'fermented', 'fishy', 'floral', 'fresh', 'fruit skin',
'fruity', 'garlic', 'gassy', 'geranium', 'grape', 'grapefruit', 'grassy',
'green', 'hawthorn', 'hay', 'hazelnut', 'herbal', 'honey', 'hyacinth',
'jasmin', 'juicy', 'ketonic', 'lactonic', 'lavender', 'leafy', 'leathery',
'lemon', 'lily', 'malty', 'meaty', 'medicinal', 'melon', 'metallic',
'milky', 'mint', 'muguet', 'mushroom', 'musk', 'musty', 'natural', 'nutty',
'odorless', 'oily', 'onion', 'orange', 'orangeflower', 'orris', 'ozone',
'peach', 'pear', 'phenolic', 'pine', 'pineapple', 'plum', 'popcorn',
'potato', 'powdery', 'pungent', 'radish', 'raspberry', 'ripe', 'roasted',
'rose', 'rummy', 'sandalwood', 'savory', 'sharp', 'smoky', 'soapy',
'solvent', 'sour', 'spicy', 'strawberry', 'sulfurous', 'sweaty', 'sweet',
'tea', 'terpenic', 'tobacco', 'tomato', 'tropical', 'vanilla', 'vegetable',
'vetiver', 'violet', 'warm', 'waxy', 'weedy', 'winey', 'woody'
]
print(DEVICE)

def label_to_color(label):
    # Generate stable hash from label name
    h = int(hashlib.md5(label.encode()).hexdigest(), 16)

    # Map hash to hue in HSV space
    hue = (h % 360) / 360.0

    # Fixed saturation and value for readability
    return mcolors.hsv_to_rgb((hue, 0.65, 0.85))

LABEL_COLORS = {label: label_to_color(label) for label in TARGET_COLUMN}

# %%
full_data = pd.read_csv(FULL_PATH)
train_data = pd.read_csv(TRAIN_PATH)
hf_encoder, tl_encoder, tokenizer, hf_regressor, tl_regressor, scaler = load_chemberta_models(
    MERGED_MODEL_PATH, TOKENIZER_NAME, DEVICE, SCALER_PATH, train_data=train_data
)
print(hf_encoder, tl_encoder, tokenizer, hf_regressor, tl_regressor, scaler)
# %% [markdown]
# Validating conversation (check whether the two models have the same internals and output, extremely important!)
# First check internal and then output
# %%
test_smiles = "CCO" # arbitrary
inputs = tokenizer(test_smiles, return_tensors="pt").to(DEVICE)

conversion_results = validate_conversion(hf_encoder, tl_encoder, inputs["input_ids"], inputs["attention_mask"])
print(f"The difference between the final embeddings are less than 0.001: {conversion_results["final_output"] < 0.001}")

prediction_results = test_prediction_equivalence(hf_regressor, tl_regressor, [test_smiles], tokenizer, DEVICE)
print(f"The predictions are equivalent: {prediction_results["is_equivalent"]}")
# %% [markdown]
# Let's run ablation studies to see the effect of missing components
test_data = pd.read_csv(TEST_PATH)
test_molecules = test_data['nonStereoSMILES'].to_list()
targets = test_data[TARGET_COLUMN].values

print(f"Testing ablation on {len(test_molecules)} molecules")
#print(f"Target range: {min(targets):.3f} to {max(targets):.3f}")

smell_results = run_ablation_analysis_with_metrics(tl_encoder, tl_regressor, tokenizer, test_data, smiles_column="nonStereoSMILES", target_column=TARGET_COLUMN, output_dir=Path("results/smell/final_frozen"), n_seeds=10, scaler=scaler)
plot_ablation_metrics(smell_results, Path("results/smell/final_frozen"), title="Ablation on Frozen model")

# %% [markdown]
## assume TARGET_COLUMNS is a list of all label column names
# e.g. TARGET_COLUMNS = smell_labels_list
test_data = pd.read_csv(TEST_PATH)
smiles_list = test_data["nonStereoSMILES"].tolist()
targets = test_data[TARGET_COLUMN].values.astype(int)

results = run_classification_lens(
    tl_model=tl_encoder,
    classifier=tl_regressor,
    smiles=smiles_list,
    tokenizer=tokenizer,
    device=DEVICE,
    batch_size=8,
    threshold=0.35,
)

df_f1 = compute_f1_per_label_per_layer(
    results,
    smiles_list=smiles_list,
    targets=targets,
)
df_f1.to_csv("mechanistic_interpretability/results/smell/f1_per_label_per_layer_frozen.csv")

top_labels, bottom_labels = plot_top_bottom_labels_f1(
    df_f1_layer_label=df_f1,
    label_names=TARGET_COLUMN,
    k=5,
    title="Per-label F1 across layers (Frozen Model)",
    save_path="mechanistic_interpretability/results/smell/final_frozen/classification_lens/top_bottom_f1_frozen.pdf",
    LABEL_COLORS=LABEL_COLORS,
)


# %% [markdown]
# Now classification lens ons clusters
clustered_data = pd.read_csv("mechanistic_interpretability/clustered_data/smell_prediction/pom_test_clustered.csv")
molecule_groups = {}
ordered_targets = []
for cluster, group in clustered_data.groupby('cluster'):
    cluster_name = f"Cluster {cluster + 1}"
    molecule_groups[cluster_name] = group['nonStereoSMILES'].tolist()
    # group[TARGET_COLUMN] is (n_group, num_labels)
    ordered_targets.extend(group[TARGET_COLUMN].values.astype(int))

ordered_targets = np.array(ordered_targets)

group_results = compare_molecule_groups_classification_lens(
    tl_model=tl_encoder,
    classifier=tl_regressor,
    group_smiles=molecule_groups,
    tokenizer=tokenizer,
    targets=ordered_targets,
    threshold=0.35,
    results_dir="mechanistic_interpretability/results/smell/final_frozen/classification_lens/",
    device=DEVICE,
)

plot_cluster_layer_f1_lines(group_results, metric="macro_f1_present", results_dir=Path("mechanistic_interpretability/results/smell/final_frozen/classification_lens"), title="")

# %%
# Now for LoRA Model
MODEL_PATH = "mechanistic_interpretability/trained_models/LORA_FINAL/chemberta_multilabel_model_final.bin"
ADAPTER_DIR = "mechanistic_interpretability/trained_models/LORA_FINAL/"
MERGED_MODEL_PATH = "mechanistic_interpretability/trained_models/LORA_FINAL/chemberta_multilabel_model_final_merged_plain.bin"
FULL_PATH = "mechanistic_interpretability/clustered_data/pom/Multi-Labelled_Smiles_Odors_dataset.csv"
TEST_PATH = "mechanistic_interpretability/clustered_data/pom/test_stratified10.csv"
VAL_PATH = "mechanistic_interpretability/clustered_data/pom/val_stratified10.csv"
TRAIN_PATH = "mechanistic_interpretability/clustered_data/pom/train_stratified80.csv"
TOKENIZER_NAME = "DeepChem/ChemBERTa-77M-MLM"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
hyperparams_path = None
SCALER_PATH = None
TARGET_COLUMN = [
'alcoholic', 'aldehydic', 'alliaceous', 'almond', 'amber', 'animal',
'anisic', 'apple', 'apricot', 'aromatic', 'balsamic', 'banana', 'beefy',
'bergamot', 'berry', 'bitter', 'black currant', 'brandy', 'burnt',
'buttery', 'cabbage', 'camphoreous', 'caramellic', 'cedar', 'celery',
'chamomile', 'cheesy', 'cherry', 'chocolate', 'cinnamon', 'citrus', 'clean',
'clove', 'cocoa', 'coconut', 'coffee', 'cognac', 'cooked', 'cooling',
'cortex', 'coumarinic', 'creamy', 'cucumber', 'dairy', 'dry', 'earthy',
'ethereal', 'fatty', 'fermented', 'fishy', 'floral', 'fresh', 'fruit skin',
'fruity', 'garlic', 'gassy', 'geranium', 'grape', 'grapefruit', 'grassy',
'green', 'hawthorn', 'hay', 'hazelnut', 'herbal', 'honey', 'hyacinth',
'jasmin', 'juicy', 'ketonic', 'lactonic', 'lavender', 'leafy', 'leathery',
'lemon', 'lily', 'malty', 'meaty', 'medicinal', 'melon', 'metallic',
'milky', 'mint', 'muguet', 'mushroom', 'musk', 'musty', 'natural', 'nutty',
'odorless', 'oily', 'onion', 'orange', 'orangeflower', 'orris', 'ozone',
'peach', 'pear', 'phenolic', 'pine', 'pineapple', 'plum', 'popcorn',
'potato', 'powdery', 'pungent', 'radish', 'raspberry', 'ripe', 'roasted',
'rose', 'rummy', 'sandalwood', 'savory', 'sharp', 'smoky', 'soapy',
'solvent', 'sour', 'spicy', 'strawberry', 'sulfurous', 'sweaty', 'sweet',
'tea', 'terpenic', 'tobacco', 'tomato', 'tropical', 'vanilla', 'vegetable',
'vetiver', 'violet', 'warm', 'waxy', 'weedy', 'winey', 'woody'
]
print(DEVICE)

def label_to_color(label):
    # Generate stable hash from label name
    h = int(hashlib.md5(label.encode()).hexdigest(), 16)

    # Map hash to hue in HSV space
    hue = (h % 360) / 360.0

    # Fixed saturation and value for readability
    return mcolors.hsv_to_rgb((hue, 0.65, 0.85))

LABEL_COLORS = {label: label_to_color(label) for label in TARGET_COLUMN}

# %%
full_data = pd.read_csv(FULL_PATH)
train_data = pd.read_csv(TRAIN_PATH)
hf_encoder, tl_encoder, tokenizer, hf_regressor, tl_regressor, scaler = load_chemberta_models(
    MERGED_MODEL_PATH, TOKENIZER_NAME, DEVICE, SCALER_PATH, train_data=train_data
)
print(hf_encoder, tl_encoder, tokenizer, hf_regressor, tl_regressor, scaler)
# %% [markdown]
# Validating conversation (check whether the two models have the same internals and output, extremely important!)
# First check internal and then output
# %%
test_smiles = "CCO" # arbitrary
inputs = tokenizer(test_smiles, return_tensors="pt").to(DEVICE)

conversion_results = validate_conversion(hf_encoder, tl_encoder, inputs["input_ids"], inputs["attention_mask"])
print(f"The difference between the final embeddings are less than 0.001: {conversion_results["final_output"] < 0.001}")

prediction_results = test_prediction_equivalence(hf_regressor, tl_regressor, [test_smiles], tokenizer, DEVICE)
print(f"The predictions are equivalent: {prediction_results["is_equivalent"]}")
# %% [markdown]
# Let's run ablation studies to see the effect of missing components
test_data = pd.read_csv(TEST_PATH)
test_molecules = test_data['nonStereoSMILES'].to_list()
targets = test_data[TARGET_COLUMN].values

print(f"Testing ablation on {len(test_molecules)} molecules")
#print(f"Target range: {min(targets):.3f} to {max(targets):.3f}")

smell_results = run_ablation_analysis_with_metrics(tl_encoder, tl_regressor, tokenizer, test_data, smiles_column="nonStereoSMILES", target_column=TARGET_COLUMN, output_dir=Path("results/smell/final_lora"), n_seeds=10, scaler=scaler)
plot_ablation_metrics(smell_results, Path("mechanistic_interpretability/results/smell/final_lora"), title="Ablation on LoRA fine-tuned model")

# %% [markdown]
## assume TARGET_COLUMNS is a list of all label column names
# e.g. TARGET_COLUMNS = smell_labels_list
test_data = pd.read_csv(TEST_PATH)
smiles_list = test_data["nonStereoSMILES"].tolist()
targets = test_data[TARGET_COLUMN].values.astype(int)

results = run_classification_lens(
    tl_model=tl_encoder,
    classifier=tl_regressor,
    smiles=smiles_list,
    tokenizer=tokenizer,
    device=DEVICE,
    batch_size=8,
    threshold=0.35,
)

df_f1 = compute_f1_per_label_per_layer(
    results,
    smiles_list=smiles_list,
    targets=targets,
)
df_f1.to_csv("mechanistic_interpretability/results/smell/f1_per_label_per_layer_lora.csv")

top_labels, bottom_labels = plot_top_bottom_labels_f1(
    df_f1_layer_label=df_f1,
    label_names=TARGET_COLUMN,
    k=5,
    title="Per-label F1 across layers (LoRA)",
    save_path="mechanistic_interpretability/results/smell/final_lora/classification_lens/top_bottom_f1_frozen.pdf",
    LABEL_COLORS=LABEL_COLORS,
)


# %% [markdown]
# Now classification lens ons clusters
clustered_data = pd.read_csv("mechanistic_interpretability/clustered_data/smell_prediction/pom_test_clustered.csv")
molecule_groups = {}
ordered_targets = []
for cluster, group in clustered_data.groupby('cluster'):
    cluster_name = f"Cluster {cluster + 1}"
    molecule_groups[cluster_name] = group['nonStereoSMILES'].tolist()
    # group[TARGET_COLUMN] is (n_group, num_labels)
    ordered_targets.extend(group[TARGET_COLUMN].values.astype(int))

ordered_targets = np.array(ordered_targets)

group_results = compare_molecule_groups_classification_lens(
    tl_model=tl_encoder,
    classifier=tl_regressor,
    group_smiles=molecule_groups,
    tokenizer=tokenizer,
    targets=ordered_targets,
    threshold=0.35,
    results_dir="mechanistic_interpretability/results/smell/final_lora/classification_lens",
    device=DEVICE,
)


plot_cluster_layer_f1_lines(group_results, metric="macro_f1_present", results_dir=Path("mechanistic_interpretability/results/smell/final_lora/classification_lens"), title="Per-cluster Macro F1 across layers (LoRA)")




# %%
#lora attention studies
device = DEVICE

if hyperparams_path is None:
    # Try to find hyperparameters.json in the same directory as the model
    model_dir = Path(MODEL_PATH).parent
    hyperparams_path = model_dir / "hyperparameters.json"

# Load hyperparameters with fallback to defaults
ARGS = {
    "dropout": 0.34,
    "hidden_channels": 384,
    "num_mlp_layers": 1,
}

if Path(hyperparams_path).exists():
    try:
        import json

        with open(hyperparams_path, 'r') as f:
            loaded_hyperparams = json.load(f)
        ARGS.update(loaded_hyperparams)
        print(f"Loaded hyperparameters from {hyperparams_path}")
        print(f"Hyperparameters: {ARGS}")
    except Exception as e:
        print(f"Could not load hyperparameters from {hyperparams_path}: {e}")
        print(f"Using defaults: {ARGS}")
else:
    print(f"Hyperparameters file not found at {hyperparams_path}")
    print(f"Using defaults: {ARGS}")

hf_model = ChembertaMultiLabelClassifier(
    pretrained=TOKENIZER_NAME,
    num_labels=len(TARGET_COLUMN),
    pooling_strat=ARGS["pooling_strat"],
    dropout=ARGS["dropout"],
    hidden_channels=ARGS["hidden_channels"],
    num_mlp_layers=ARGS["num_mlp_layers"],
    gamma=ARGS["gamma"],
    alpha=ARGS["alpha"],
    lora_r=ARGS["lora_r"],
    lora_alpha=ARGS["lora_alpha"],
    lora_dropout=ARGS["lora_dropout"]
)

state = torch.load(MODEL_PATH, map_location="cpu")
hf_model.load_state_dict(state, strict=True)
hf_model.to(device)
hf_model.eval()

test_data = pd.read_csv(TEST_PATH)
smiles_list = test_data["nonStereoSMILES"].tolist()

hf_off = scale_lora_contribution(hf_model, scale=0.0).to(device)
hf_on  = scale_lora_contribution(hf_model, scale=1.0).to(device)

df = rank_heads_by_attn_change_encoder(
    hf_off, hf_on,
    tokenizer_name=TOKENIZER_NAME,
    smiles_list=smiles_list,
    device=device,
    batch_size=8,
    max_batches=100,
)

print(df.head(36))


num_layers = int(df["layer"].max()) + 1
num_heads  = int(df["head"].max()) + 1

heatmap = np.zeros((num_layers, num_heads), dtype=float)

for r in df.itertuples(index=False):
    # r.layer, r.head, r.mean_abs_attn_diff come from column names
    heatmap[int(r.layer), int(r.head)] = float(r.mean_abs_attn_diff)


import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.imshow(heatmap, aspect="auto")
plt.colorbar(label="Mean |Δ attention| (LoRA on − off)")

plt.xlabel("Head")
plt.ylabel("Layer")

plt.xticks(
    ticks=np.arange(num_heads),
    labels=np.arange(1, num_heads + 1)
)
plt.yticks(
    ticks=np.arange(num_layers),
    labels=np.arange(1, num_layers + 1)
)

plt.title("LoRA-induced attention change per head")
plt.tight_layout()
plt.savefig("mechanistic_interpretability/results/smell/lora_diff_heatmap.pdf")
plt.show()