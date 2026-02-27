# Mechanistic Interpretability of ChemBERTa for Multi-Label Odour Prediction

This repository contains the code for a project on multi-label odour prediction from molecular
structure using SMILES representations.

The project evaluates ChemBERTa (a Transformer pretrained on SMILES)
under different fine-tuning strategies and investigates its internal
mechanisms using mechanistic interpretability (MI) techniques.

------------------------------------------------------------------------

## Overview

The main research goals are:

1.  Evaluate ChemBERTa for multi-label odour prediction.
2.  Compare:
    -   Frozen encoder + trained classification head
    -   LoRA fine-tuning (parameter-efficient fine-tuning)
    -   MPNN graph-based baseline
3.  Analyze internal mechanisms using:
    -   Ablation studies (attention vs FFN)
    -   Classification lens probing
    -   LoRA attention-difference analysis

The dataset consists of molecules represented as SMILES strings with
multiple odour labels per molecule.

------------------------------------------------------------------------

## Repository Structure

    
    ├── Data/
    │   ├── Multi-Labelled_Smiles_Odors_dataset.csv
    │   ├── splits/
    │   │   ├── train_stratified80.csv
    │   │   ├── val_stratified10.csv
    │   │   └── test_stratified10.csv
    │   ├── data_prep.py
    │   ├── analysis.py
    │   └── functional_group_detector.py
    │
    ├── models/
    │   ├── mpnn.py
    │   └── simple_mlp.py
    │
    ├── training/
    │   └── training.py
    │
    ├── utils/
    │   ├── chemberta_workflows.py
    │   ├── chemberta_final_train_plus_saving.py
    │   ├── chemberta_dataset.py
    │   └── plotting.py
    │
    ├── mechanistic_interpretability/
    │   ├── TL_chem.py
    │   ├── models/
    │   │   └── chemberta_regressor.py
    │   ├── utils/
    │   │   ├── tl_conversion.py
    │   │   ├── ablation.py
    │   │   ├── lens.py
    │   │   └── ...
    │   ├── requirements.txt
    │   └── LICENSE
    │
    ├── trained_models/
    │   ├── FROZEN_FINAL/
    │   └── LORA_FINAL/
    │
    ├── figures/
    ├── optuna_runs/
    ├── full_optuna.py
    ├── main.py
    ├── requirements.txt
    └── README.md

------------------------------------------------------------------------

## Installation

### 1. Create a virtual environment

Recommended Python version: 3.10+

    python -m venv .venv
    source .venv/bin/activate  # Windows: .venv\Scripts\activate
    pip install --upgrade pip

### 2. Install dependencies

Main project:

    pip install -r requirements.txt

Mechanistic interpretability dependencies:

    pip install -r mechanistic_interpretability/requirements.txt

------------------------------------------------------------------------

## Dataset

Main dataset:

    Data/Multi-Labelled_Smiles_Odors_dataset.csv

Pre-generated multi-label stratified splits (80/10/10):

    Data/splits/train_stratified80.csv
    Data/splits/val_stratified10.csv
    Data/splits/test_stratified10.csv

To regenerate splits, see:

    Data/data_prep.py

------------------------------------------------------------------------

## Training ChemBERTa

Run from the project root:

    python main.py

This will:

-   Load stratified train/validation/test splits
-   Train frozen ChemBERTa variants
-   Train LoRA fine-tuned variants
-   Save final frozen and LoRA checkpoints
-   Export evaluation metrics

------------------------------------------------------------------------

## Hyperparameters

Hyperparameters are defined inside:

    training/training.py

Key parameters include:

-   pooling_strat: mean \| max \| cls \| cls_mean \| cls_max \| mean_max
-   loss_type: focal \| bce
-   gamma, alpha: Focal loss parameters
-   threshold: Probability threshold for converting predictions to
    binary labels
-   LoRA-specific: lora_r, lora_alpha, lora_dropout

------------------------------------------------------------------------

## Hyperparameter Search (Optuna)

Run:

    python full_optuna.py

Results are stored in:

    optuna_runs/

------------------------------------------------------------------------

## MPNN Baseline

Graph-based baseline implementation:

    models/mpnn.py

------------------------------------------------------------------------

## Mechanistic Interpretability (MI)

MI experiments are located in:

    mechanistic_interpretability/

Main entrypoint:

    python mechanistic_interpretability/TL_chem.py

Ensure trained model checkpoints exist in:

    trained_models/FROZEN_FINAL/
    trained_models/LORA_FINAL/

------------------------------------------------------------------------

## Outputs

Saved artifacts include:

-   Trained model checkpoints
-   Per-label metrics (per_label_metrics.csv)
-   Training curves
-   PDF figures in /figures
-   Optuna search logs

------------------------------------------------------------------------

## Reproducibility

-   Fixed random seed
-   Precomputed stratified splits included
-   All evaluation uses fixed train/val/test splits


------------------------------------------------------------------------

## Acknowledgements


- mechanistic_interpretability/ module is adapted from: **MoLe** (Molecular Lens)
Pollice Research Group – Data and Intelligence
https://git.lwp.rug.nl/pollice-research-group/data-and-intelligence/mole

- **ChemBERTa**: Pre-trained model from [DeepChem](https://github.com/deepchem/deepchem)

- This project uses the odour dataset originally published on [Kaggle](https://www.kaggle.com/code/aryanamitbarsainyan/replicate-principal-odor-map)