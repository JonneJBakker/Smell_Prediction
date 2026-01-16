"""
chemberta_workflows.py (NO LoRA)

- Loads DeepChem/ChemBERTa-77M-MLM via Hugging Face
- Freezes base encoder, trains an MLP head
- Pooling strategies: cls, mean, max, cls_mean, mean_max, attention
- Losses: bce, focal, (optional) asl
"""

import os
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    hamming_loss,
    jaccard_score,
    roc_auc_score,
)

from transformers import (
    RobertaModel,
    RobertaTokenizerFast,
    Trainer,
    TrainingArguments,
)
from transformers.modeling_outputs import SequenceClassifierOutput

from models.simple_mlp import SimpleMLP

DEFAULT_PRETRAINED_NAME = "DeepChem/ChemBERTa-77M-MLM"


# -------------------------
# Loss functions
# -------------------------
class FocalLoss(nn.Module):
    """Multi-label focal loss on logits + {0,1} targets."""
    def __init__(self, alpha=None, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
        weight = (1.0 - p_t) ** self.gamma

        if self.alpha is not None:
            # alpha can be scalar or tensor broadcastable to targets
            weight = self.alpha * weight

        loss = weight * bce
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class AsymmetricFocalLoss(nn.Module):
    """
    Asymmetric focal loss for multi-label classification.
    Useful under heavy imbalance.
    """
    def __init__(
        self,
        gamma_pos=0.0,
        gamma_neg=4.0,
        clip=0.0,
        eps=1e-8,
        reduction="mean",
        pos_weight=None,
    ):
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip = clip
        self.eps = eps
        self.reduction = reduction
        self.pos_weight = pos_weight

    def forward(self, logits, targets):
        prob = torch.sigmoid(logits)
        prob_pos = prob
        prob_neg = 1.0 - prob

        if self.clip and self.clip > 0:
            prob_neg = torch.clamp(prob_neg + self.clip, max=1.0)

        log_pos = torch.log(torch.clamp(prob_pos, min=self.eps))
        log_neg = torch.log(torch.clamp(prob_neg, min=self.eps))

        loss = targets * log_pos + (1.0 - targets) * log_neg

        if self.pos_weight is not None:
            loss = loss * (targets * self.pos_weight + (1.0 - targets))

        if self.gamma_pos != 0.0 or self.gamma_neg != 0.0:
            pt = prob_pos * targets + prob_neg * (1.0 - targets)
            gamma = self.gamma_pos * targets + self.gamma_neg * (1.0 - targets)
            loss = loss * torch.pow(1.0 - pt, gamma)

        loss = -loss
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


# -------------------------
# Pooling utilities
# -------------------------
def mean_pool(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).type_as(hidden_states)  # (B, T, 1)
    summed = (hidden_states * mask).sum(dim=1)                  # (B, H)
    denom = mask.sum(dim=1).clamp(min=1e-9)                     # (B, 1)
    return summed / denom


def max_pool(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).bool()                  # (B, T, 1)
    masked = hidden_states.masked_fill(~mask, float("-inf"))    # padded -> -inf
    pooled, _ = masked.max(dim=1)                               # (B, H)
    # if a sequence were all padding (shouldn't happen), replace -inf with 0
    pooled = torch.where(torch.isfinite(pooled), pooled, torch.zeros_like(pooled))
    return pooled

# -------------------------
# Model
# -------------------------
class ChembertaMultiLabelClassifier(nn.Module):
    """
    ChemBERTa encoder (frozen) + pooling + MLP head.
    """
    def __init__(
        self,
        pretrained: str,
        num_labels: int,
        dropout: float = 0.3,
        hidden_channels: int = 256,
        num_mlp_layers: int = 2,
        pooling_strat: str = "mean_max",
        loss_type: str = "bce",  # "bce" | "focal" | "asl"
        # focal params
        gamma: float = 2.0,
        alpha=None,
        # asl params
        gamma_pos: float = 0.0,
        gamma_neg: float = 4.0,
        asl_clip: float = 0.0,
        pos_weight=None,
        freeze_encoder: bool = True,
    ):
        super().__init__()
        self.pooling_strat = pooling_strat

        self.roberta = RobertaModel.from_pretrained(pretrained, add_pooling_layer=False)

        if freeze_encoder:
            for p in self.roberta.parameters():
                p.requires_grad = False

        if pooling_strat == "attention":
            self.query_vector = nn.Parameter(torch.randn(self.roberta.config.hidden_size))

        self.dropout = nn.Dropout(dropout)

        base_dim = self.roberta.config.hidden_size
        if pooling_strat in ("cls_mean", "mean_max", "cls_max"):
            head_in_dim = 2 * base_dim
        else:
            head_in_dim = base_dim

        self.mlp = SimpleMLP(
            input_dim=head_in_dim,
            hidden_channels=hidden_channels,
            num_layers=num_mlp_layers,
            output_dim=num_labels,
            dropout=dropout,
        )

        loss_type = (loss_type or "bce").lower()
        self.loss_type = loss_type

        if loss_type == "bce":
            self.loss_fct = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        elif loss_type == "focal":
            self.loss_fct = FocalLoss(alpha=alpha, gamma=gamma, reduction="mean")
        elif loss_type == "asl":
            self.loss_fct = AsymmetricFocalLoss(
                gamma_pos=gamma_pos,
                gamma_neg=gamma_neg,
                clip=asl_clip,
                reduction="mean",
                pos_weight=pos_weight,
            )
        else:
            raise ValueError(f"Unknown loss_type='{loss_type}'. Use 'bce', 'focal', or 'asl'.")

    def _pool(self, token_embs: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        strat = self.pooling_strat

        if strat == "cls":
            return token_embs[:, 0, :]

        if strat == "mean":
            return mean_pool(token_embs, attention_mask)

        if strat == "max":
            return max_pool(token_embs, attention_mask)

        if strat == "cls_mean":
            cls = token_embs[:, 0, :]
            mean = mean_pool(token_embs, attention_mask)
            return torch.cat([cls, mean], dim=1)

        if strat == "mean_max":
            mean = mean_pool(token_embs, attention_mask)
            mx = max_pool(token_embs, attention_mask)
            return torch.cat([mean, mx], dim=1)

        if strat == "cls_max":
            cls = token_embs[:, 0, :]
            max = max_pool(token_embs, attention_mask)
            return torch.cat([cls, max], dim=1)

        if strat == "attention":
            # masked attention pooling with a learned query vector
            # scores: (B, T)
            scores = torch.matmul(token_embs, self.query_vector)
            # mask out padding positions
            scores = scores.masked_fill(attention_mask == 0, float("-inf"))
            weights = torch.softmax(scores, dim=1).unsqueeze(-1)   # (B, T, 1)
            return torch.sum(token_embs * weights, dim=1)           # (B, H)

        raise ValueError(f"Unknown pooling_strat='{strat}'.")

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        token_embs = outputs.last_hidden_state  # (B, T, H)

        pooled = self._pool(token_embs, attention_mask)
        x = self.dropout(pooled)
        logits = self.mlp(x)

        loss = None
        if labels is not None:
            loss = self.loss_fct(logits, labels)

        return SequenceClassifierOutput(loss=loss, logits=logits)


# -------------------------
# Dataset
# -------------------------
class ChembertaDataset(Dataset):
    def __init__(self, texts, targets, tokenizer, max_length=512):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": self.targets[idx],
        }


# -------------------------
# Metrics
# -------------------------
def get_multilabel_compute_metrics_fn(threshold=0.3):
    def compute_metrics(eval_pred):
        logits = eval_pred.predictions
        labels = eval_pred.label_ids

        probs = 1.0 / (1.0 + np.exp(-logits))
        preds = (probs >= threshold).astype(int)

        micro_accuracy = accuracy_score(labels, preds)
        auroc = roc_auc_score(labels, probs, multi_class="ovr", average="macro")
        micro_f1 = f1_score(labels.astype(int), preds, average="micro", zero_division=0)
        macro_f1 = f1_score(labels.astype(int), preds, average="macro", zero_division=0)
        samples_f1 = f1_score(labels.astype(int), preds, average="samples", zero_division=0)
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


def evaluate_per_label_metrics(trainer, dataset, label_names, threshold=0.3):
    """
    Optional helper if you want per-label metrics; returns macro-F1 for convenience.
    """
    out = trainer.predict(dataset)
    logits = out.predictions
    probs = 1.0 / (1.0 + np.exp(-logits))
    preds = (probs >= threshold).astype(int)
    y_true = out.label_ids.astype(int)

    macro_f1 = f1_score(y_true, preds, average="macro", zero_division=0)
    return macro_f1


# -------------------------
# Training entrypoint
# -------------------------
def train_chemberta_multilabel_model(args, df_train, df_test, df_val, device=None):
    """
    Train ChemBERTa (frozen encoder) + MLP head.

    Expected args fields (minimal):
      - smiles_column
      - target_columns
      - train_csv (for naming)
      - output_dir
      - epochs, batch_size, lr
      - l2_lambda (weight decay), dropout
      - hidden_channels, num_mlp_layers
      - pooling_strat
      - loss_type (bce|focal|asl), gamma, alpha, gamma_pos, gamma_neg, asl_clip
      - threshold
      - random_seed
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = RobertaTokenizerFast.from_pretrained(DEFAULT_PRETRAINED_NAME)

    smiles_col = args.smiles_column
    target_cols = args.target_columns

    texts_train = df_train[smiles_col].tolist()
    y_train = df_train[target_cols].values.astype(np.float32)

    texts_val = df_val[smiles_col].tolist()
    y_val = df_val[target_cols].values.astype(np.float32)

    texts_test = df_test[smiles_col].tolist()
    y_test = df_test[target_cols].values.astype(np.float32)

    num_labels = y_train.shape[1]

    train_ds = ChembertaDataset(texts_train, y_train, tokenizer)
    val_ds = ChembertaDataset(texts_val, y_val, tokenizer)
    test_ds = ChembertaDataset(texts_test, y_test, tokenizer)

    model = ChembertaMultiLabelClassifier(
        pretrained=DEFAULT_PRETRAINED_NAME,
        num_labels=num_labels,
        dropout=args.dropout,
        hidden_channels=args.hidden_channels,
        num_mlp_layers=args.num_mlp_layers,
        pooling_strat=args.pooling_strat,
        loss_type=getattr(args, "loss_type", "bce"),
        gamma=getattr(args, "gamma", 2.0),
        alpha=getattr(args, "alpha", None),
        gamma_pos=getattr(args, "gamma_pos", 0.0),
        gamma_neg=getattr(args, "gamma_neg", 4.0),
        asl_clip=getattr(args, "asl_clip", 0.0),
        freeze_encoder=True,
    )

    output_dir = args.output_dir
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
    )

    compute_metrics = get_multilabel_compute_metrics_fn(threshold=args.threshold)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )

    print("\nTraining ChemBERTa (frozen encoder) multi-label model...")
    start_time = datetime.now()
    trainer.train()
    training_time = (datetime.now() - start_time).total_seconds()
    print(f"Training completed in {training_time:.2f} seconds")

    print("\nEvaluating model on test set...")
    metrics = trainer.evaluate(eval_dataset=test_ds)
    test_macro_f1 = evaluate_per_label_metrics(trainer, test_ds, target_cols, threshold=args.threshold)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("\nModel parameters:")
    print(f"  Total parameters:     {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    return metrics, test_macro_f1
