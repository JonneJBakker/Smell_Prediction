"""
Utility module to handle ChemBERTa training workflows.

This module contains functions for training ChemBERTa models with preprocessing,
training, and evaluation.
"""

import os
import json
import time
import pickle
from datetime import datetime
import argparse

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, f1_score
)
from torch import nn
from torch.utils.data import Dataset
from transformers import (
    EarlyStoppingCallback,
    RobertaModel,
    RobertaTokenizerFast,
    Trainer,
    TrainingArguments,
    AutoConfig
)
from transformers.modeling_outputs import SequenceClassifierOutput

from models.simple_mlp import SimpleMLP

DEFAULT_PRETRAINED_NAME = "DeepChem/ChemBERTa-77M-MLM"


class L1Trainer(Trainer):
    """
    Custom trainer that adds L1 regularization to the loss function.
    """

    def __init__(self, l1_lambda=0.0, **kwargs):
        super().__init__(**kwargs)
        self.l1_lambda = l1_lambda
        self.regularized_losses = []  # Track regularized losses for plotting

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        raw_mse_loss = outputs.loss  # This is the raw MSE loss from the model

        # For training, add L1 regularization
        if self.training:
            if self.l1_lambda > 0:
                l1_norm = sum(p.abs().sum() for p in model.parameters())
                regularized_loss = raw_mse_loss + self.l1_lambda * l1_norm

                self.regularized_losses.append(regularized_loss.item())
                loss = regularized_loss
            else:
                loss = raw_mse_loss
                self.regularized_losses.append(raw_mse_loss.item())
        else:
            loss = raw_mse_loss

        return (loss, outputs) if return_outputs else loss

    def training(self):
        """Check if the model is in training mode"""
        return self.model.training if hasattr(self.model, "training") else True

    def log(self, logs, *args, **kwargs):
        """Override log method to include regularized loss in history"""
        if self.model.training and self.regularized_losses:
            logs["regularized_loss"] = self.regularized_losses[-1]
        super().log(logs, *args, **kwargs)



class ChembertaMultiLabelClassifier(nn.Module):
    """
    ChemBERTa multi-label classification model.
    """

    def __init__(
        self,
        pretrained,
        num_labels,
        num_features=0,
        dropout=0.3,
        hidden_channels=100,
        num_mlp_layers=1,
        pos_weight=None,
    ):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained(pretrained, add_pooling_layer=False)

        self.dropout = nn.Dropout(dropout)
        num_input_features = self.roberta.config.hidden_size

        # Output dimension = num_labels, one logit per label
        self.classifier = SimpleMLP(
            num_input_features,
            hidden_channels,
            num_mlp_layers,
            num_labels,
            dropout,
        )

        if pos_weight is not None:
            self.loss_fct = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            self.loss_fct = nn.BCEWithLogitsLoss()

    def forward(self, input_ids=None, attention_mask=None, labels=None, features=None):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        cls_emb = outputs.last_hidden_state[:, 0, :]  # CLS token

        x = self.dropout(cls_emb)

        # (batch_size, num_labels)
        logits = self.classifier(x)

        loss = None
        if labels is not None:
            # labels expected shape: (batch_size, num_labels), dtype=float
            loss = self.loss_fct(logits, labels)

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
        )

class ChembertaDataset(Dataset):
    """
    Dataset for ChemBERTa model that handles tokenized SMILES strings and numerical features.
    """

    def __init__(self, texts, targets, tokenizer, features=None):
        # Pre-tokenize all texts at initialization time
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors="pt",
        )
        self.targets = torch.tensor(targets, dtype=torch.float32)
        self.features = None
        if features is not None:
            self.features = torch.tensor(features, dtype=torch.float32)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        item = {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": self.targets[idx],
        }

        if self.features is not None and hasattr(self, "features"):
            item["features"] = self.features[idx]

        return item


def get_multilabel_compute_metrics_fn(threshold=0.5):
    """
    Returns a function to compute metrics for multi-label classification.

    Assumes:
        - eval_pred.predictions is (N, num_labels) logits
        - eval_pred.label_ids is (N, num_labels) in {0,1}
    """

    def compute_metrics(eval_pred):
        logits = eval_pred.predictions
        labels = eval_pred.label_ids

        # Convert logits to probabilities then to binary predictions
        probs = 1 / (1 + np.exp(-logits))
        preds = (probs >= threshold).astype(int)

        # Simple overall metrics
        # Flatten for micro metrics
        micro_accuracy = accuracy_score(labels.flatten(), preds.flatten())
        micro_f1 = f1_score(labels.flatten(), preds.flatten(), average="micro")
        macro_f1 = f1_score(labels, preds, average="macro")

        return {
            "micro_accuracy": micro_accuracy,
            "micro_f1": micro_f1,
            "macro_f1": macro_f1,
        }

    return compute_metrics


def train_chemberta_multilabel_model(
    args, df_train, df_test, df_val, device=None
):
    """
    Train a ChemBERTa model for multi-label classification on SMILES data.

    Expected args:
        - smiles_column: name of column with SMILES strings
        - target_columns: list of column names for labels (multi-label)
        - train_csv, output_dir, epochs, batch_size, lr, l2_lambda, l1_lambda,
          dropout, hidden_channels, num_mlp_layers, random_seed
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = RobertaTokenizerFast.from_pretrained(DEFAULT_PRETRAINED_NAME)

    smiles_col = args.smiles_column
    target_cols = args.target_columns  # <-- list of label columns

    # Create datasets
    texts_train = df_train[smiles_col].tolist()
    targets_train = df_train[target_cols].values.astype(np.float32)

    texts_test = df_test[smiles_col].tolist()
    targets_test = df_test[target_cols].values.astype(np.float32)

    texts_val = df_val[smiles_col].tolist()
    targets_val = df_val[target_cols].values.astype(np.float32)

    num_labels = targets_train.shape[1]

    train_dataset = ChembertaDataset(texts_train, targets_train, tokenizer)
    test_dataset = ChembertaDataset(texts_test, targets_test, tokenizer)
    val_dataset = ChembertaDataset(texts_val, targets_val, tokenizer)

    # Calculate pos_weight
    pos_targets = df_train[target_cols].values
    num_pos = (pos_targets == 1).sum(axis=0)
    num_neg = (pos_targets == 0).sum(axis=0)

    # Avoid division by zero
    pos_weight = torch.tensor(num_neg / np.maximum(num_pos, 1), dtype=torch.float32)

    # Create model
    model = ChembertaMultiLabelClassifier(
        pretrained=DEFAULT_PRETRAINED_NAME,
        num_labels=num_labels,
        dropout=args.dropout,
        hidden_channels=args.hidden_channels,
        num_mlp_layers=args.num_mlp_layers,
        pos_weight=pos_weight,
    )

    # Setup training arguments
    dataset_name = os.path.splitext(os.path.basename(args.train_csv))[0]
    output_dir = os.path.join(args.output_dir, dataset_name, "chemberta_multilabel")
    os.makedirs(output_dir, exist_ok=True)

    evaluation_strategy = "epoch"
    save_strategy = "epoch"
    load_best_model_at_end = True

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        eval_strategy=evaluation_strategy,
        save_strategy=save_strategy,
        save_total_limit=1,
        learning_rate=args.lr,
        weight_decay=args.l2_lambda,
        load_best_model_at_end=load_best_model_at_end,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_strategy="epoch",
        logging_first_step=True,
        seed=args.random_seed,
        report_to=["tensorboard"],
    )

    compute_metrics = get_multilabel_compute_metrics_fn(threshold=0.5)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
        #l1_lambda=args.l1_lambda,
    )

    print("\nTraining ChemBERTa multi-label model...")
    start_time = datetime.now()
    train_result = trainer.train()
    training_time = (datetime.now() - start_time).total_seconds()
    print(f"Training completed in {training_time:.2f} seconds")

    print("\nEvaluating model on test set...")
    metrics = trainer.evaluate(eval_dataset=val_dataset)

    predictions_output = trainer.predict(val_dataset)
    logits = predictions_output.predictions
    probs = 1 / (1 + np.exp(-logits))
    preds = (probs >= 0.5).astype(int)
    labels = predictions_output.label_ids

    print(f"\nModel parameters:")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    # Save model, tokenizer, config
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    AutoConfig.from_pretrained(DEFAULT_PRETRAINED_NAME).save_pretrained(output_dir)

    model_path = os.path.join(output_dir, "chemberta_multilabel_model_final.bin")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    hyperparams = {
        "hidden_channels": args.hidden_channels,
        "lr": args.lr,
        "l2_lambda": args.l2_lambda,
        "l1_lambda": args.l1_lambda,
        "batch_size": args.batch_size,
        "dropout": args.dropout,
        "num_mlp_layers": args.num_mlp_layers,
        "epochs": args.epochs,
        "num_labels": num_labels,
    }
    hyperparams_path = os.path.join(output_dir, "hyperparameters.json")
    with open(hyperparams_path, "w") as f:
        json.dump(hyperparams, f, indent=2)

    complete_results = {
        "model": model,
        "trainer": trainer,
        "metrics": metrics,
        "probs": probs,
        "preds": preds,
        "targets": labels,
        "history": trainer.state.log_history,
        "training_time": training_time,
        "output_dir": output_dir,
        "hyperparams": hyperparams,
    }

    json_serializable_results = {
        "metrics": metrics,
        "probs": probs.tolist(),
        "preds": preds.tolist(),
        "targets": labels.tolist(),
        "history": trainer.state.log_history,
        "training_time": training_time,
        "output_dir": output_dir,
        "hyperparams": hyperparams,
    }

    complete_results_path = os.path.join(output_dir, "all_results.json")
    with open(complete_results_path, "w") as f:
        json.dump(json_serializable_results, f, indent=2)

    return complete_results
