"""
Utility module to handle MolFormer training workflows.

This module mirrors `chemberta_workflows.py` but uses a MolFormer-style
Transformer as the backbone instead of ChemBERTa/Roberta.

It supports multi-label classification on SMILES with:
- Focal loss
- Multiple pooling strategies (cls, mean, max_mean, attention, mean_pooling)
- HuggingFace Trainer integration
"""

import os
import json
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset

from sklearn.metrics import (
    precision_score,
    recall_score,
    classification_report,
    accuracy_score,
    f1_score,
    hamming_loss,
    jaccard_score,
    roc_auc_score,
)

from transformers import (
    AutoModel,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    AutoConfig,
)
from transformers.modeling_outputs import SequenceClassifierOutput

from models.simple_mlp import SimpleMLP


# ---------------------------------------------------------------------------
# CONFIG: change this to the MolFormer checkpoint you want to use
# ---------------------------------------------------------------------------
DEFAULT_MOLFORMER_NAME = "ibm-research/MoLFormer-XL-both-10pct"
# ^ Replace this string with the actual HF model id you want to use.
#   It should be something you can load via:
#   AutoModel.from_pretrained(DEFAULT_MOLFORMER_NAME)
#   AutoTokenizer.from_pretrained(DEFAULT_MOLFORMER_NAME)


class FocalLoss(nn.Module):
    """
    Multi-label focal loss.

    Args:
        alpha: balancing factor (float or tensor of shape [num_labels])
        gamma: focusing parameter
        reduction: 'mean' or 'sum'
    """

    def __init__(self, alpha=None, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        # BCE with logits
        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            logits,
            targets,
            reduction="none",
        )

        # p_t = sigmoid(logit) if target=1, else 1-sigmoid(logit)
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)

        # focal weight
        focal_weight = (1 - p_t) ** self.gamma

        if self.alpha is not None:
            focal_weight = self.alpha * focal_weight

        loss = focal_weight * bce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


def mean_pool(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Masked mean pooling over the sequence dimension.

    hidden_states: (batch, seq_len, hidden)
    attention_mask: (batch, seq_len)
    """
    mask = attention_mask.unsqueeze(-1).type_as(hidden_states)  # (batch, seq_len, 1)
    summed = (hidden_states * mask).sum(dim=1)  # (batch, hidden)
    denom = mask.sum(dim=1).clamp(min=1e-9)
    return summed / denom


class MolformerMultiLabelClassifier(nn.Module):
    """
    MolFormer multi-label classification model.

    Architecturally analogous to ChembertaMultiLabelClassifier, but uses
    an AutoModel-loaded MolFormer backbone and matching tokenizer.

    Pooling strategies:
        - 'cls': use first token embedding
        - 'mean_pooling': masked mean over tokens
        - 'cls_mean': concat [mean, cls]
        - 'max_mean': concat [mean, max]
        - 'attention': learned query-vector attention over tokens
    """

    def __init__(
        self,
        pretrained: str,
        num_labels: int,
        num_features: int = 0,
        dropout: float = 0.3,
        hidden_channels: int = 100,
        num_mlp_layers: int = 1,
        pos_weight=None,
        gamma: float = 0.75,
        alpha=None,
        pooling_strat: str = "cls_mean",
        freeze_backbone: bool = True,
    ):
        super().__init__()

        # Load MolFormer-style backbone
        self.backbone = AutoModel.from_pretrained(
            pretrained,
            trust_remote_code=True,
        )
        self.pooling_strat = pooling_strat

        # Optionally freeze backbone
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        hidden_size = self.backbone.config.hidden_size

        # Optional attention pooling
        if self.pooling_strat == "attention":
            self.query_vector = nn.Parameter(torch.randn(hidden_size))

        self.dropout = nn.Dropout(dropout)

        if self.pooling_strat in ("cls_mean", "max_mean"):
            num_input_features = 2 * hidden_size
        else:
            num_input_features = hidden_size

        # Include optional extra numeric features if you want:
        # num_input_features += num_features

        # MLP head → num_labels logits
        self.mlp = SimpleMLP(
            num_input_features,
            hidden_channels,
            num_mlp_layers,
            num_labels,
            dropout,
        )

        self.loss_fct = FocalLoss(
            alpha=alpha,
            gamma=gamma,
            reduction="mean",
        )

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        features=None,
        strat: str = None,
    ):
        if strat is None:
            strat = self.pooling_strat

        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        # Many HF models expose last hidden state as .last_hidden_state
        token_embs = outputs.last_hidden_state  # (batch, seq_len, hidden)

        if strat == "cls_mean":
            cls_emb = token_embs[:, 0, :]  # CLS-like token
            mean_pooled = mean_pool(token_embs, attention_mask)  # (batch, hidden)
            pooled = torch.cat([mean_pooled, cls_emb], dim=1)
            x = self.dropout(pooled)

        elif strat == "max_mean":
            mean_pooled = mean_pool(token_embs, attention_mask)  # [B, H]

            mask = attention_mask.unsqueeze(-1).bool()  # [B, N, 1]
            # set padded positions to -inf so they don't affect max
            masked_token_embs = token_embs.masked_fill(~mask, float("-inf"))
            max_pooled, _ = masked_token_embs.max(dim=1)  # [B, H]

            pooled = torch.cat([mean_pooled, max_pooled], dim=1)  # [B, 2H]
            x = self.dropout(pooled)

        elif strat == "mean_pooling":
            pooled = mean_pool(token_embs, attention_mask)  # (batch, hidden)
            x = self.dropout(pooled)

        elif strat == "cls":
            cls_emb = token_embs[:, 0, :]  # CLS token
            x = self.dropout(cls_emb)

        elif strat == "attention":
            # [B, N, D]
            # attn_scores: [B, N]
            attn_scores = torch.matmul(token_embs, self.query_vector)
            # [B, N, 1]
            attn_weights = torch.softmax(attn_scores, dim=1).unsqueeze(-1)
            # weighted sum → [B, D]
            pooled = torch.sum(token_embs * attn_weights, dim=1)
            x = self.dropout(pooled)

        else:
            # Fallback to CLS
            cls_emb = token_embs[:, 0, :]
            x = self.dropout(cls_emb)

        # (batch_size, num_labels)
        logits = self.mlp(x)

        loss = None
        if labels is not None:
            # labels expected shape: (batch_size, num_labels), dtype=float
            loss = self.loss_fct(logits, labels)

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
        )


class MolformerDataset(Dataset):
    """
    Dataset for MolFormer model that handles tokenized SMILES strings and numerical features.
    """

    def __init__(self, texts, targets, tokenizer, features=None, max_length: int = 512):
        # Pre-tokenize all texts at initialization time
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

        if self.features is not None and hasattr(self, "features"):
            item["features"] = self.features[idx]

        return item


def get_multilabel_compute_metrics_fn(threshold: float = 0.3):
    """
    Returns a function to compute metrics for multi-label classification.

    Assumes:
        - eval_pred.predictions is (N, num_labels) logits
        - eval_pred.label_ids is (N, num_labels) in {0,1}
    """

    def compute_metrics(eval_pred):
        logits = eval_pred.predictions
        labels = eval_pred.label_ids

        probs = 1 / (1 + np.exp(-logits))
        preds = (probs >= threshold).astype(int)

        micro_accuracy = accuracy_score(labels, preds)
        auroc = roc_auc_score(labels, probs, multi_class="ovr", average="macro")
        micro_f1 = f1_score(labels.astype(int), preds, average="micro", zero_division=0)
        macro_f1 = f1_score(labels.astype(int), preds, average="macro", zero_division=0)
        samples_f1 = f1_score(labels, preds, average="samples", zero_division=0)
        hamming = hamming_loss(labels, preds)
        jaccard_samples = jaccard_score(labels, preds, average="samples", zero_division=0)

        return {
            "micro_accuracy": micro_accuracy,
            "auroc": auroc,
            "micro_f1": micro_f1,
            "macro_f1": macro_f1,
            "samples_f1": samples_f1,
            "hamming_loss": hamming,
            "jaccard_samples": jaccard_samples,
        }

    return compute_metrics


def evaluate_per_label_metrics(
    trainer,
    dataset,
    target_cols: List[str],
    threshold: float = 0.3,
):
    """
    Evaluate per-label precision, recall, F1, and frequency for a multi-label MolFormer model.

    Returns pd.DataFrame with columns:
        ['label', 'precision', 'recall', 'f1', 'support', 'frequency']
    """
    pred_output = trainer.predict(dataset)
    logits = pred_output.predictions
    labels = pred_output.label_ids

    probs = 1 / (1 + np.exp(-logits))
    preds = (probs >= threshold).astype(int)

    preds_per_label = preds.sum(axis=0)
    true_per_label = labels.sum(axis=0)

    print("Predicted positives per label:", preds_per_label)
    print("True positives per label:", true_per_label)

    precisions = precision_score(labels, preds, average=None, zero_division=0)
    recalls = recall_score(labels, preds, average=None, zero_division=0)
    f1s = f1_score(labels, preds, average=None, zero_division=0)
    supports = labels.sum(axis=0)
    freqs = labels.mean(axis=0)
    auroc = roc_auc_score(labels, probs, multi_class="ovr", average="macro")

    df_metrics = pd.DataFrame(
        {
            "label": target_cols,
            "precision": precisions,
            "recall": recalls,
            "f1": f1s,
            "support": supports,
            "frequency": freqs,
        }
    ).sort_values("f1", ascending=False).reset_index(drop=True)

    print(f"AUROC: {auroc}")
    print("\nMacro F1:", np.round(f1_score(labels, preds, average="macro"), 3))
    print("Micro F1:", np.round(f1_score(labels, preds, average="micro"), 3))
    print("\nTop and bottom 5 labels by F1:")
    print(df_metrics.head(5)[["label", "f1"]])
    print(df_metrics.tail(5)[["label", "f1"]])

    csv_path = f"{trainer.args.output_dir}/per_label_metrics_molformer.csv"
    df_metrics.to_csv(csv_path, index=False)
    print(f"Saved per-label metrics to {csv_path}")

    return np.round(f1_score(labels, preds, average="macro"), 3)


def train_molformer_multilabel_model(
    args,
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    df_val: pd.DataFrame,
    device=None,
):
    """
    Train a MolFormer model for multi-label classification on SMILES data.

    Expected args (mirroring chemberta_workflows):
        - smiles_column: name of column with SMILES strings
        - target_columns: list of column names for labels (multi-label)
        - train_csv, output_dir, epochs, batch_size, lr, l2_lambda, l1_lambda,
          dropout, hidden_channels, num_mlp_layers, random_seed, threshold,
          gamma, alpha, pooling_strat
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(
        DEFAULT_MOLFORMER_NAME,
        trust_remote_code=True,
    )

    smiles_col = args.smiles_column
    target_cols = args.target_columns

    texts_train = df_train[smiles_col].tolist()
    targets_train = df_train[target_cols].values.astype(np.float32)

    texts_test = df_test[smiles_col].tolist()
    targets_test = df_test[target_cols].values.astype(np.float32)

    texts_val = df_val[smiles_col].tolist()
    targets_val = df_val[target_cols].values.astype(np.float32)

    num_labels = targets_train.shape[1]

    train_dataset = MolformerDataset(texts_train, targets_train, tokenizer)
    test_dataset = MolformerDataset(texts_test, targets_test, tokenizer)
    val_dataset = MolformerDataset(texts_val, targets_val, tokenizer)

    model = MolformerMultiLabelClassifier(
        pretrained=DEFAULT_MOLFORMER_NAME,
        num_labels=num_labels,
        dropout=args.dropout,
        hidden_channels=args.hidden_channels,
        num_mlp_layers=args.num_mlp_layers,
        pos_weight=None,
        gamma=args.gamma,
        alpha=args.alpha,
        pooling_strat=args.pooling_strat,
        freeze_backbone=getattr(args, "freeze_backbone", True),
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"molformer_multilabel_{timestamp}")
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

    compute_metrics = get_multilabel_compute_metrics_fn(threshold=args.threshold)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    print("\nTraining MolFormer multi-label model.")
    start_time = datetime.now()
    train_result = trainer.train()
    training_time = (datetime.now() - start_time).total_seconds()
    print(f"Training completed in {training_time:.2f} seconds")

    print("\nEvaluating model on test set.")
    metrics = trainer.evaluate(eval_dataset=test_dataset)

    f1_macro = evaluate_per_label_metrics(trainer, test_dataset, target_cols, threshold=args.threshold)

    predictions_output = trainer.predict(test_dataset)
    logits = predictions_output.predictions
    probs = 1 / (1 + np.exp(-logits))
    preds = (probs >= args.threshold).astype(int)
    labels = predictions_output.label_ids

    print(f"\nModel parameters:")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    AutoConfig.from_pretrained(
        DEFAULT_MOLFORMER_NAME,
        trust_remote_code=True,
    ).save_pretrained(output_dir)

    model_path = os.path.join(output_dir, "molformer_multilabel_model_final.bin")
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
        "threshold": args.threshold,
        "gamma": args.gamma,
        "alpha": args.alpha,
        "pooling_strat": args.pooling_strat,
        "freeze_backbone": getattr(args, "freeze_backbone", True),
    }
    hyperparams_path = os.path.join(output_dir, "hyperparameters_molformer.json")
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

    complete_results_path = os.path.join(output_dir, "all_results_molformer.json")
    with open(complete_results_path, "w") as f:
        json.dump(json_serializable_results, f, indent=2)

    return complete_results, f1_macro
