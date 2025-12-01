"""
Utility module to handle ChemBERTa training workflows.

This module contains functions for training ChemBERTa models with preprocessing,
training, and evaluation, plus optional contrastive learning on SMILES.

Public API (same as original):
    - FocalLoss
    - ChembertaMultiLabelClassifier
    - ChembertaDataset
    - get_multilabel_compute_metrics_fn
    - evaluate_per_label_metrics
    - train_chemberta_multilabel_model
"""

import os
import json
from datetime import datetime
import argparse
import time
import pickle

import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.utils.data import Dataset
import torch.nn.functional as F

from transformers import (
    EarlyStoppingCallback,
    RobertaModel,
    RobertaTokenizerFast,
    Trainer,
    TrainingArguments,
    AutoConfig,
)
from transformers.modeling_outputs import SequenceClassifierOutput

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

from models.simple_mlp import SimpleMLP

from rdkit import Chem
DEFAULT_PRETRAINED_NAME = "DeepChem/ChemBERTa-77M-MLM"


# ============================================================
#  Losses / pooling / heads
# ============================================================

class FocalLoss(nn.Module):
    """
    Multi-label focal loss.
    Args:
        alpha: balancing factor (float or tensor of shape [num_labels])
        gamma: focusing parameter
        reduction: 'mean' or 'sum'
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        # logits: (B, L), targets: (B, L)
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, targets, reduction='none'
        )  # (B, L)

        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma  # (B, L)

        if self.alpha is not None:
            if isinstance(self.alpha, torch.Tensor):
                alpha = self.alpha.to(logits.device)
                focal_weight = alpha * focal_weight
            else:
                focal_weight = self.alpha * focal_weight

        loss = focal_weight * bce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


def mean_pool(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Mean pool over non-padding tokens.

    hidden_states: (batch, seq_len, hidden)
    attention_mask: (batch, seq_len) with 1 for tokens, 0 for padding
    """
    mask = attention_mask.unsqueeze(-1).type_as(hidden_states)  # (B, L, 1)
    summed = (hidden_states * mask).sum(dim=1)                  # (B, H)
    denom = mask.sum(dim=1).clamp(min=1e-9)
    return summed / denom

# ============================================================
#  ChemBERTa model with optional contrastive term
# ============================================================

class ChembertaMultiLabelClassifier(nn.Module):
    """
    ChemBERTa multi-label classification model, extended with optional
    contrastive learning on the pooled text embedding.

    The constructor signature mirrors the original, plus two new kwargs:
        contrastive_weight (lambda) and contrastive_temperature.
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
        gamma=0.75,
        alpha=None,
        pooling_strat="max_mean",  # "cls", "mean_pooling", "cls_mean", "max_mean", "attention"
        contrastive_weight: float = 0.0,
        contrastive_temperature: float = 0.1,
    ):
        super().__init__()

        self.pooling_strat = pooling_strat
        self.num_labels = num_labels
        self.num_features = num_features

        self.roberta = RobertaModel.from_pretrained(pretrained, add_pooling_layer=False)
        self.pooling_strat = pooling_strat

        # freeze language model
        for param in self.roberta.parameters():
            param.requires_grad = False

        hidden_size = self.roberta.config.hidden_size

        # Pooling-specific dimension
        if pooling_strat in ("cls_mean", "max_mean"):
            pooled_dim = 2 * hidden_size
        else:
            pooled_dim = hidden_size

        #self.query_vector = nn.Parameter(torch.randn(hidden_size))

        # Include extra numerical features
        if num_features > 0:
            num_input_features = pooled_dim + num_features
        else:
            num_input_features = pooled_dim

        # Classifier head (same as original)
        self.dropout = nn.Dropout(dropout)
        self.classifier = SimpleMLP(
            num_input_features,
            hidden_channels,
            num_mlp_layers,
            num_labels,
            dropout,
        )

        # Focal loss; pos_weight kept for backward compatibility (if you want)
        if pos_weight is not None:
            # you can integrate pos_weight into alpha if desired
            pass

        self.loss_fct = FocalLoss(
            alpha=alpha,
            gamma=gamma,
            reduction="mean",
        )

        # Contrastive head and hyperparams
        self.contrastive_head = nn.Sequential(
            nn.Linear(pooled_dim, pooled_dim),
            nn.ReLU(),
            nn.Linear(pooled_dim, pooled_dim),
        )
        self.contrastive_weight = contrastive_weight
        self.contrastive_temperature = contrastive_temperature

    # ---------- Pooling helpers ----------

    def _pool(
        self,
        token_embs: torch.Tensor,
        attention_mask: torch.Tensor,
        strat: str = None,
    ) -> torch.Tensor:
        """
        Return pooled sequence embedding according to pooling_strat.
        """
        if strat is None:
            strat = self.pooling_strat

        if strat == "cls_mean":
            cls_emb = token_embs[:, 0, :]                       # [B, H]
            mean_emb = mean_pool(token_embs, attention_mask)    # [B, H]
            pooled = torch.cat([mean_emb, cls_emb], dim=1)      # [B, 2H]

        elif strat == "max_mean":
            mean_emb = mean_pool(token_embs, attention_mask)    # [B, H]
            mask = attention_mask.unsqueeze(-1).bool()          # [B, L, 1]
            masked = token_embs.masked_fill(~mask, float("-inf"))
            max_emb, _ = masked.max(dim=1)                      # [B, H]
            pooled = torch.cat([mean_emb, max_emb], dim=1)      # [B, 2H]

        elif strat == "mean_pooling":
            pooled = mean_pool(token_embs, attention_mask)      # [B, H]

        elif strat == "cls":
            pooled = token_embs[:, 0, :]                        # [B, H]

        elif strat == "attention":
            # token_embs: [B, L, H]
            attn_scores = torch.matmul(token_embs, self.query_vector)  # [B, L]
            attn_weights = torch.softmax(attn_scores, dim=1).unsqueeze(-1)  # [B, L, 1]
            pooled = torch.sum(token_embs * attn_weights, dim=1)  # [B, H]

        else:
            raise ValueError(f"Unknown pooling strategy: {str}")

        return pooled

    def _nt_xent_loss(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        NT-Xent loss: one-direction InfoNCE for a batch of positive pairs.

        z1, z2 are normalized embeddings of shape [B, D].
        """
        if self.contrastive_temperature is None:
            temperature = 0.1
        else:
            temperature = self.contrastive_temperature

        logits = (z1 @ z2.t()) / temperature  # [B, B]
        labels = torch.arange(z1.size(0), device=z1.device)
        return F.cross_entropy(logits, labels)

    # ---------- Public encode helper ----------

    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        strat: str = None,
        normalize: bool = True,
    ) -> torch.Tensor:
        """
        Return pooled sequence embedding (no classifier, no contrastive head).
        """
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        token_embs = outputs.last_hidden_state
        pooled = self._pool(token_embs, attention_mask, strat=strat)
        if normalize:
            pooled = F.normalize(pooled, dim=-1)
        return pooled

    # ---------- Forward: classification + optional contrastive ----------

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        features=None,
        strat=None,
        # NEW: second view for contrastive learning
        input_ids_pair=None,
        attention_mask_pair=None,
    ) -> SequenceClassifierOutput:

        if strat is None:
            strat = self.pooling_strat

        # Encode main view
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        token_embs = outputs.last_hidden_state
        pooled_text = self._pool(token_embs, attention_mask, strat=strat)  # [B, *]

        # Include extra features ONLY for classification
        if features is not None:
            pooled_for_cls = torch.cat([pooled_text, features], dim=1)
        else:
            pooled_for_cls = pooled_text

        x = self.dropout(pooled_for_cls)
        logits = self.classifier(x)

        # Classification loss
        cls_loss = None
        if labels is not None:
            cls_loss = self.loss_fct(logits, labels)

        # Contrastive loss (optional)
        cl_loss = None
        if (
            self.contrastive_weight is not None
            and self.contrastive_weight > 0.0
            and input_ids_pair is not None
            and attention_mask_pair is not None
        ):
            outputs_pair = self.roberta(
                input_ids=input_ids_pair,
                attention_mask=attention_mask_pair,
            )
            token_embs_pair = outputs_pair.last_hidden_state
            pooled_text_pair = self._pool(
                token_embs_pair,
                attention_mask_pair,
                strat=strat,
            )

            z1 = F.normalize(self.contrastive_head(pooled_text), dim=-1)
            z2 = F.normalize(self.contrastive_head(pooled_text_pair), dim=-1)
            cl_loss = self._nt_xent_loss(z1, z2)

        # Combine
        if cls_loss is not None and cl_loss is not None:
            loss = cls_loss + self.contrastive_weight * cl_loss
        elif cls_loss is not None:
            loss = cls_loss
        elif cl_loss is not None:
            loss = cl_loss
        else:
            loss = None

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
        )


# ============================================================
#  Datasets
# ============================================================

class ChembertaDataset(Dataset):
    """
    Plain classification dataset (no contrastive pairs).
    Mirrors original class.
    """

    def __init__(self, texts, targets, tokenizer, features=None, max_length=512):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
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
        if self.features is not None:
            item["features"] = self.features[idx]
        return item


class ChembertaContrastiveDataset(Dataset):
    """
    Dataset that returns a second augmented SMILES view per sample
    for contrastive learning.
    """

    def __init__(
        self,
        texts,
        targets,
        tokenizer,
        augment_fn,
        features=None,
        max_length=512,
    ):
        self.texts = texts
        self.augment_fn = augment_fn

        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )

        texts_aug = [augment_fn(t) for t in texts]
        self.encodings_pair = tokenizer(
            texts_aug,
            truncation=True,
            padding="max_length",
            max_length=max_length,
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
            "input_ids_pair": self.encodings_pair["input_ids"][idx],
            "attention_mask_pair": self.encodings_pair["attention_mask"][idx],
            "labels": self.targets[idx],
        }
        if self.features is not None:
            item["features"] = self.features[idx]
        return item


# ============================================================
#  SMILES augmentation for contrastive learning
# ============================================================

def augment_random_smiles(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles
    return Chem.MolToSmiles(mol, doRandom=True)


def mask_smiles(smiles: str, p: float = 0.03) -> str:
    chars = list(smiles)
    for i in range(len(chars)):
        if chars[i].isalnum() and np.random.rand() < p:
            chars[i] = "*"
    return "".join(chars)


def token_dropout(smiles: str, p: float = 0.02) -> str:
    out = []
    for c in smiles:
        if np.random.rand() < p:
            continue
        out.append(c)
    return "".join(out)


def default_augment_fn(smiles: str) -> str:
    aug = augment_random_smiles(smiles)
    if np.random.rand() < 0.0:
        aug = mask_smiles(aug, p=0.03)
    if np.random.rand() < 0.0:
        aug = token_dropout(aug, p=0.02)
    return aug


# ============================================================
#  Metrics
# ============================================================

def get_multilabel_compute_metrics_fn(threshold):
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

        accuracy = accuracy_score(labels, preds)
        precision_macro = precision_score(labels, preds, average="macro", zero_division=0)
        recall_macro = recall_score(labels, preds, average="macro", zero_division=0)
        f1_macro = f1_score(labels, preds, average="macro", zero_division=0)
        f1_micro = f1_score(labels, preds, average="micro", zero_division=0)

        try:
            auroc_macro = roc_auc_score(labels, probs, average="macro")
        except Exception:
            auroc_macro = float("nan")

        return {
            "accuracy": accuracy,
            "precision_macro": precision_macro,
            "recall_macro": recall_macro,
            "macro_f1": f1_macro,
            "micro_f1": f1_micro,
            "auroc_macro": auroc_macro,
        }

    return compute_metrics


def evaluate_per_label_metrics(trainer, dataset, target_cols, threshold):
    """
    Evaluate per-label precision, recall, F1, and frequency for a multi-label model.

    Args:
        trainer: Hugging Face Trainer object after training.
        dataset: Dataset object (train/val/test) to evaluate on.
        target_cols: List of label column names (order matters!).
        threshold: Float probability threshold.
    """
    pred_output = trainer.predict(dataset)
    logits = pred_output.predictions
    labels = pred_output.label_ids

    probs = 1 / (1 + np.exp(-logits))
    preds = (probs >= threshold).astype(int)

    # Per-label metrics
    precisions = precision_score(labels, preds, average=None, zero_division=0)
    recalls = recall_score(labels, preds, average=None, zero_division=0)
    f1s = f1_score(labels, preds, average=None, zero_division=0)
    supports = labels.sum(axis=0)
    freqs = supports / labels.shape[0]

    try:
        auroc = roc_auc_score(labels, probs, average="macro")
    except Exception:
        auroc = float("nan")

    df_metrics = pd.DataFrame({
        "label": target_cols,
        "precision": precisions,
        "recall": recalls,
        "f1": f1s,
        "support": supports,
        "frequency": freqs,
    }).sort_values("f1", ascending=False).reset_index(drop=True)

    print(f"AUROC: {auroc}")
    print("\nMacro F1:", np.round(f1_score(labels, preds, average="macro"), 3))
    print("Micro F1:", np.round(f1_score(labels, preds, average="micro"), 3))
    print("\nTop and bottom 5 labels by F1:")
    print(df_metrics.head(5)[["label", "f1"]])
    print(df_metrics.tail(5)[["label", "f1"]])

    csv_path = f"{trainer.args.output_dir}/per_label_metrics.csv"
    df_metrics.to_csv(csv_path, index=False)
    print(f" Saved per-label metrics to {csv_path}")

    return np.round(f1_score(labels, preds, average="macro"), 3)


# ============================================================
#  Training wrapper (mirrors original)
# ============================================================

def train_chemberta_multilabel_model(
    args,
    df_train,
    df_test,
    df_val,
    device=None,
    threshold=0.25,
    gamma=1.0,
    alpha=None,
):
    """
    Train a ChemBERTa model for multi-label classification on SMILES data.

    Expected args (same as original, plus a few optional contrastive args):
        - smiles_column
        - target_columns
        - train_csv
        - output_dir
        - epochs
        - batch_size
        - lr
        - l2_lambda
        - l1_lambda (if you used it)
        - dropout
        - hidden_channels
        - num_mlp_layers
        - random_seed
        - pooling_strat
        - OPTIONAL: use_contrastive (bool)
        - OPTIONAL: contrastive_weight (float)
        - OPTIONAL: contrastive_temperature (float)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = RobertaTokenizerFast.from_pretrained(DEFAULT_PRETRAINED_NAME)

    smiles_col = args.smiles_column
    target_cols = args.target_columns

    texts_train = df_train[smiles_col].tolist()
    targets_train = df_train[target_cols].values.astype(np.float32)

    texts_test = df_test[smiles_col].tolist()
    targets_test = df_test[target_cols].values.astype(np.float32)

    texts_val = df_val[smiles_col].tolist()
    targets_val = df_val[target_cols].values.astype(np.float32)

    num_labels = len(target_cols)

    # Compute class-wise alpha if requested
    alpha_tensor = None
    if alpha is not None:
        pos_counts = targets_train.sum(axis=0)
        neg_counts = targets_train.shape[0] - pos_counts
        alpha_np = alpha * (neg_counts / (pos_counts + 1e-8))
        alpha_tensor = torch.tensor(alpha_np, dtype=torch.float32, device=device)
        print("Using alpha:", alpha_np)

    # Whether to use contrastive training
    use_contrastive = getattr(args, "use_contrastive", False)
    contrastive_weight = float(getattr(args, "contrastive_weight", 0.0))
    contrastive_temperature = float(getattr(args, "contrastive_temperature", 0.1))

    # Build datasets
    if use_contrastive and contrastive_weight > 0.0:
        print("Using contrastive dataset + loss.")
        train_dataset = ChembertaContrastiveDataset(
            texts=texts_train,
            targets=targets_train,
            tokenizer=tokenizer,
            augment_fn=default_augment_fn,
            max_length=getattr(args, "max_length", 512),
        )
    else:
        print("Using standard classification dataset (no contrastive pairs).")
        train_dataset = ChembertaDataset(
            texts=texts_train,
            targets=targets_train,
            tokenizer=tokenizer,
            max_length=getattr(args, "max_length", 512),
        )

    val_dataset = ChembertaDataset(
        texts=texts_val,
        targets=targets_val,
        tokenizer=tokenizer,
        max_length=getattr(args, "max_length", 512),
    )

    test_dataset = ChembertaDataset(
        texts=texts_test,
        targets=targets_test,
        tokenizer=tokenizer,
        max_length=getattr(args, "max_length", 512),
    )

    # Create model
    model = ChembertaMultiLabelClassifier(
        pretrained=DEFAULT_PRETRAINED_NAME,
        num_labels=num_labels,
        dropout=args.dropout,
        hidden_channels=args.hidden_channels,
        num_mlp_layers=args.num_mlp_layers,
        pos_weight=None,
        gamma=gamma,
        alpha=alpha_tensor,
        pooling_strat=args.pooling_strat,
        contrastive_weight=contrastive_weight if use_contrastive else 0.0,
        contrastive_temperature=contrastive_temperature,
    )

    # Training args
    output_dir = os.path.join(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        learning_rate=args.lr,
        weight_decay=args.l2_lambda,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_strategy="epoch",
        logging_first_step=True,
        seed=args.random_seed,
        report_to=["tensorboard"],
        remove_unused_columns=False,  # needed for contrastive extra keys
    )

    compute_metrics = get_multilabel_compute_metrics_fn(threshold=threshold)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    print("\nTraining ChemBERTa multi-label model...")
    start_time = datetime.now()
    train_result = trainer.train()
    training_time = (datetime.now() - start_time).total_seconds()
    print(f"Training completed in {training_time:.2f} seconds")

    print("\nEvaluating model on test set...")
    metrics = trainer.evaluate(eval_dataset=val_dataset)

    f1_macro = evaluate_per_label_metrics(trainer, test_dataset, target_cols, threshold=threshold)

    predictions_output = trainer.predict(test_dataset)
    logits = predictions_output.predictions
    probs = 1 / (1 + np.exp(-logits))
    preds = (probs >= threshold).astype(int)
    labels = predictions_output.label_ids

    # Save results
    hyperparams = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "l2_lambda": args.l2_lambda,
        "dropout": args.dropout,
        "hidden_channels": args.hidden_channels,
        "num_mlp_layers": args.num_mlp_layers,
        "random_seed": args.random_seed,
        "pooling_strat": args.pooling_strat,
        "gamma": gamma,
        "alpha": alpha,
        "use_contrastive": use_contrastive,
        "contrastive_weight": contrastive_weight,
        "contrastive_temperature": contrastive_temperature,
    }

    complete_results = {
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

    return complete_results, f1_macro
