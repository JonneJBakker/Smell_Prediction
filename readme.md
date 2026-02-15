# ChemBERTa for Multi-Label Odor Prediction

## Overview

This project investigates **multi-label odor prediction from molecular SMILES representations** using deep learning approaches. The primary objective is to model the relationship between molecular structure and perceptual odor descriptors.

We compare transformer-based language models for chemistry (ChemBERTa) with graph-based neural networks (MPNN), and explore parameter-efficient fine-tuning strategies. The study emphasizes:

- Representation learning from SMILES strings
- Multi-label classification under class imbalance
- Parameter-efficient adaptation (LoRA)
- Per-label performance analysis
- Functional group and fragment-level interpretability

---

## Dataset

**Input:** SMILES strings  
**Output:** Multiple odor descriptors per molecule

Main dataset file:

```
Data/Multi-Labelled_Smiles_Odors_dataset.csv
```

Stratified splits (80/10/10):

```
Data/splits/train_stratified80.csv
Data/splits/val_stratified10.csv
Data/splits/test_stratified10.csv
```

Splitting is performed using stratification techniques to preserve label distribution across subsets.

---

## Model Architectures

### 1. ChemBERTa (Frozen Backbone)

A pretrained ChemBERTa transformer encoder is used as a fixed feature extractor. Only the classification head is trained.

- Backbone parameters frozen
- Linear classification head
- Sigmoid outputs for multi-label prediction

Saved under:

```
trained_models/FROZEN_FINAL/
```

---

### 2. ChemBERTa with LoRA (Parameter-Efficient Fine-Tuning)

Low-Rank Adaptation (LoRA) is applied to selected attention layers to enable parameter-efficient fine-tuning.

Saved under:

```
trained_models/LORA_FINAL/
```

Merged model artifact:

```
chemberta_multilabel_model_final_merged_plain.bin
```

---

### 3. Message Passing Neural Network (MPNN)

A graph-based baseline model implemented in:

```
models/mpnn.py
```

This model operates on molecular graph representations rather than SMILES token sequences.

---

## Training Pipeline

Primary entry point:

```
python main.py
```

Alternative scripts:

- Hyperparameter optimization: `python full_optuna.py`

### Hyperparameter Optimization

Optuna is used for automated hyperparameter search.


---

## Evaluation Metrics

Model performance is evaluated using:

- Micro F1-score
- Macro F1-score
- AUROC

Per-label metrics are saved in:

```
trained_models/*/per_label_metrics.csv
```

---

## Functional Group & Fragment Analysis

Implemented in:

```
Data/functional_group_detector.py
Data/analysis.py
```

This module supports:

- Functional group detection
- Fragment–odor association analysis
- Co-occurrence matrix construction
- Exploratory chemical interpretability studies


---

## Installation

### Requirements

- Python 3.10+
- torch
- transformers
- peft
- scikit-learn
- pandas
- numpy
- matplotlib
- optuna
- RDKit (for functional group analysis)

Example installation:

```bash
pip install -r requirements.txt
```




## Author

Jonne Bakker
University of Groningen
2026

