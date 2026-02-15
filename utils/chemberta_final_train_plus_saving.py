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
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
import torch
from sklearn.metrics import (
    precision_score, recall_score, classification_report,
    accuracy_score, f1_score, hamming_loss,
    jaccard_score, roc_auc_score
)
from torch import nn
from torch.utils.data import Dataset

from transformers import (
    EarlyStoppingCallback,
    RobertaModel,
    RobertaTokenizerFast,
    Trainer,
    TrainingArguments,
    AutoConfig, AutoModel
)
from transformers.modeling_outputs import SequenceClassifierOutput

from models.simple_mlp import SimpleMLP

DEFAULT_PRETRAINED_NAME = "DeepChem/ChemBERTa-77M-MLM"


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
    # hidden_states: (batch, seq_len, hidden)
    # attention_mask: (batch, seq_len)
    mask = attention_mask.unsqueeze(-1).type_as(hidden_states)  # (batch, seq_len, 1)
    summed = (hidden_states * mask).sum(dim=1)                  # (batch, hidden)
    denom = mask.sum(dim=1).clamp(min=1e-9)
    return summed / denom

class ChembertaMultiLabelClassifier(nn.Module):
    """
    ChemBERTa multi-label classification model.
    """

    def __init__(
        self,
        pretrained,
        num_labels,
        dropout=0.3,
        hidden_channels=100,
        num_mlp_layers=1,
        gamma = 0.75,
        alpha = None,
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        pooling_strat = "max_mean",
        use_lora = False,
    ):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained(pretrained, add_pooling_layer=False)
        self.pooling_strat = pooling_strat



        for param in self.roberta.parameters():
             param.requires_grad = False

        if use_lora:
            lora_cfg = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                bias="none",
                target_modules=["query", "key", "value"],
            )

            self.roberta = get_peft_model(self.roberta, lora_cfg)

            self.roberta.print_trainable_parameters()



        self.dropout = nn.Dropout(dropout)
        if self.pooling_strat == "cls_mean" or self.pooling_strat == "max_mean":
            num_input_features = 2*self.roberta.config.hidden_size
        else:
            num_input_features = self.roberta.config.hidden_size

        # Output dimension = num_labels, one logit per label
        self.mlp = SimpleMLP(
            num_input_features,
            hidden_channels,
            num_mlp_layers,
            num_labels,
            dropout,
        )

        self.loss_fct = FocalLoss(
            gamma=gamma,
            alpha=alpha,
            reduction="mean",
        )

    def forward(self, input_ids=None, attention_mask=None, labels=None, features=None, strat=None):
        if strat is None:
            strat = self.pooling_strat
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)

        if strat == "cls_mean":
            cls_emb = outputs.last_hidden_state[:, 0, :]  # CLS token
            token_embs = outputs.last_hidden_state  # (batch, seq_len, hidden)
            mean_pooled = mean_pool(token_embs, attention_mask)  # (batch, hidden)
            pooled = torch.cat([mean_pooled, cls_emb], dim=1)
            x = self.dropout(pooled)

        if strat == "max_mean":
            token_embs = outputs.last_hidden_state
            mean_pooled = mean_pool(token_embs, attention_mask)  # [B, H]

            mask = attention_mask.unsqueeze(-1).bool()  # [B, N, 1]

            # set padded positions to -inf so they don't affect max
            masked_token_embs = token_embs.masked_fill(~mask, float("-inf"))

            max_pooled, _ = masked_token_embs.max(dim=1)  # [B, H]

            pooled = torch.cat([mean_pooled, max_pooled], dim=1)  # [B, 2H]
            x = self.dropout(pooled)

        if strat == "mean":
            token_embs = outputs.last_hidden_state  # (batch, seq_len, hidden)
            pooled = mean_pool(token_embs, attention_mask)  # (batch, hidden)
            x = self.dropout(pooled)

        elif strat == "cls":
            cls_emb = outputs.last_hidden_state[:, 0, :]  # CLS token
            x = self.dropout(cls_emb)


        logits = self.mlp(x)

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


def get_multilabel_compute_metrics_fn(threshold=0.3):
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
        micro_accuracy = accuracy_score(labels, preds)
        auroc = roc_auc_score(labels, probs, multi_class="ovr", average="macro")
        micro_f1 = f1_score(labels.astype(int), preds, average="micro", zero_division=0)
        macro_f1 = f1_score(labels.astype(int), preds, average="macro", zero_division=0)
        samples_f1 = f1_score(labels, preds, average="samples", zero_division=0)
        hamming = hamming_loss(labels, preds)
        jaccard_samples = jaccard_score(labels, preds, average="samples", zero_division = 0)

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

def find_peft_dir(root_dir: str) -> str:
    """Return a directory that contains adapter_config.json (root or latest checkpoint-*)."""
    root_dir = os.path.abspath(root_dir)

    # root itself?
    if os.path.exists(os.path.join(root_dir, "adapter_config.json")):
        return root_dir

    # search checkpoint-* subdirs
    ckpts = []
    if os.path.isdir(root_dir):
        for name in os.listdir(root_dir):
            p = os.path.join(root_dir, name)
            if os.path.isdir(p) and name.startswith("checkpoint-"):
                if os.path.exists(os.path.join(p, "adapter_config.json")):
                    ckpts.append(p)

    if not ckpts:
        raise FileNotFoundError(
            f"Can't find adapter_config.json in '{root_dir}' or any checkpoint-* subdir. "
            f"Files in root: {os.listdir(root_dir) if os.path.isdir(root_dir) else 'N/A'}"
        )

    ckpts.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return ckpts[0]


def train_and_save_chemberta_multilabel_model(
    args, df_train, df_test, df_val, device=None,
):
    """
    Train a ChemBERTa model for multi-label classification on SMILES data.

    Expected args:
        - smiles_column: name of column with SMILES strings
        - target_columns: list of column names for labels (multi-label)
        - train_csv, output_dir, epochs, batch_size, lr, weight_decay, l1_lambda,
          dropout, hidden_channels, num_mlp_layers, random_seed
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = RobertaTokenizerFast.from_pretrained(DEFAULT_PRETRAINED_NAME)

    smiles_col = args.smiles_column
    target_cols = args.target_columns

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


    # Create model
    model = ChembertaMultiLabelClassifier(
        pretrained=DEFAULT_PRETRAINED_NAME,
        num_labels=num_labels,
        dropout=args.dropout,
        hidden_channels=args.hidden_channels,
        num_mlp_layers=args.num_mlp_layers,
        gamma = args.gamma,
        alpha = args.alpha,
        pooling_strat=args.pooling_strat,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        use_lora=args.use_lora,
    )

    # Setup training arguments
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, timestamp)
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
        weight_decay=args.weight_decay,
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

    print("\nTraining ChemBERTa multi-label model...")
    start_time = datetime.now()
    train_result = trainer.train()
    training_time = (datetime.now() - start_time).total_seconds()
    print(f"Training completed in {training_time:.2f} seconds")

    print("\nEvaluating model on test set...")
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

    # Save model, tokenizer, config
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    AutoConfig.from_pretrained(DEFAULT_PRETRAINED_NAME).save_pretrained(output_dir)

    # IMPORTANT: ensure adapter files exist at output_dir (or at least somewhere we can find)
    # (This is safe even if trainer.save_model already did it.)
    model.roberta.save_pretrained(output_dir)

    model_path = os.path.join(output_dir, "chemberta_multilabel_model_final.bin")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    if args.use_lora:
        #  MERGE LORA INTO BASE MODEL FOR TL COMPATIBILITY
        print("Output dir:", os.path.abspath(output_dir))
        print("Root files:", os.listdir(output_dir))

        peft_dir = find_peft_dir(output_dir)
        print("Using PEFT dir:", peft_dir)
        print("PEFT dir files:", os.listdir(peft_dir))

        # 1) Load base Roberta
        base_roberta = AutoModel.from_pretrained(
            DEFAULT_PRETRAINED_NAME,
            add_pooling_layer=False,
        )

        # 2) Load trained adapter and merge
        peft_roberta = PeftModel.from_pretrained(base_roberta, peft_dir)
        merged_roberta = peft_roberta.merge_and_unload()

        # 3) Swap merged backbone into your classifier
        model.roberta = merged_roberta

        # 4) Save a *plain* state_dict (NO peft / lora keys)
        plain_model_path = os.path.join(
            output_dir, "chemberta_multilabel_model_final_merged_plain.bin"
        )
        torch.save(model.state_dict(), plain_model_path)
        print(f"Merged plain model saved to {plain_model_path}")

    hyperparams = {
        "hidden_channels": args.hidden_channels,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "dropout": args.dropout,
        "num_mlp_layers": args.num_mlp_layers,
        "pooling_strat": args.pooling_strat,
        "random_seed": args.random_seed,
        "loss_type": args.loss_type,
        "epochs": args.epochs,
        "num_labels": num_labels,
        "threshold": args.threshold,
        "gamma": args.gamma,
        "alpha": args.alpha,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
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

    return complete_results, f1_macro


def evaluate_per_label_metrics(trainer, dataset, target_cols, threshold=0.3):
    """
    Evaluate per-label precision, recall, F1, and frequency for a multi-label model.

    Args:
        trainer: Hugging Face Trainer object after training.
        dataset: Dataset object (train/val/test) to evaluate on.
        target_cols: List of label column names (order matters!).
        threshold: Float, probability threshold for converting logits → binary predictions.

    Returns:
        pd.DataFrame with columns:
        ['label', 'precision', 'recall', 'f1', 'support', 'frequency']
    """

    # Run prediction ----
    pred_output = trainer.predict(dataset)
    logits = pred_output.predictions
    labels = pred_output.label_ids

    # Convert logits → probabilities → binary predictions ----
    probs = 1 / (1 + np.exp(-logits))
    preds = (probs >= threshold).astype(int)

    preds_per_label = preds.sum(axis=0)  # number of predicted positives per label
    true_per_label = labels.sum(axis=0)  # number of actual positives per label

    print("Predicted positives per label:", preds_per_label)
    print("True positives per label:", true_per_label)

    # Compute metrics per label ----
    precisions = precision_score(labels, preds, average=None, zero_division=0)
    recalls = recall_score(labels, preds, average=None, zero_division=0)
    f1s = f1_score(labels, preds, average=None, zero_division=0)
    supports = labels.sum(axis=0)
    freqs = labels.mean(axis=0)
    auroc = roc_auc_score(labels, probs, multi_class="ovr", average="macro")

    # Create dataframe
    df_metrics = pd.DataFrame({
        "label": target_cols,
        "precision": precisions,
        "recall": recalls,
        "f1": f1s,
        "support": supports,
        "frequency": freqs,
    }).sort_values("f1", ascending=False).reset_index(drop=True)


    # Print summary
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


def sweep_thresholds_from_saved_results(
    results_path,
    metric="macro_f1",
    thresholds=None,
):
    """
    Sweep over thresholds for a trained ChemBERTa model using saved probs/targets.

    Args:
        results_path: path to all_results.json saved by train_chemberta_multilabel_model
        metric: which metric to use to select the best threshold.
                One of: "macro_f1", "micro_f1", "samples_f1",
                        "jaccard_samples", "micro_accuracy"
        thresholds: iterable of thresholds to try. If None, uses np.linspace(0.05, 0.95, 19)

    Returns:
        df: DataFrame with metrics for each threshold
        best_row: row of df with the best value for `metric`
    """
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 19)

    with open(results_path, "r") as f:
        results = json.load(f)

    probs = np.array(results["probs"])    # shape (N, num_labels)
    labels = np.array(results["targets"]) # shape (N, num_labels)

    rows = []
    for t in thresholds:
        preds = (probs >= t).astype(int)

        micro_acc = accuracy_score(labels, preds)
        macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
        micro_f1 = f1_score(labels, preds, average="micro", zero_division=0)
        samples_f1 = f1_score(labels, preds, average="samples", zero_division=0)
        jaccard_samples = jaccard_score(labels, preds, average="samples", zero_division=0)
        hamm = hamming_loss(labels, preds)

        rows.append({
            "threshold": float(t),
            "micro_accuracy": micro_acc,
            "macro_f1": macro_f1,
            "micro_f1": micro_f1,
            "samples_f1": samples_f1,
            "jaccard_samples": jaccard_samples,
            "hamming_loss": hamm,
        })

    df = pd.DataFrame(rows)
    best_idx = df[metric].idxmax()
    best_row = df.loc[best_idx]

    print("\nBest threshold based on", metric)
    print(best_row)

    return df, best_row
