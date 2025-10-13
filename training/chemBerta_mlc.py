import pandas as pd
import numpy as np
import torch

from torch.utils.data.dataset import Dataset
from transformers import TrainingArguments, Trainer, AutoTokenizer, AutoModelForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

MODEL_NAME = "DeepChem/ChemBERTa-77M-MLM"

from Data.data_prep import split_data


def train_chemBerta(df, smiles_col):
    smiles_train, smiles_val, smiles_test, labels_train, labels_val, labels_test, label_cols = split_data(df, smiles_col=smiles_col)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

    class ChemDataset(torch.utils.data.Dataset):
        def __init__(self, smiles, labels):
            self.encodings = tokenizer(list(smiles), truncation=True, padding=True, max_length=512, return_tensors="pt")
            self.labels = torch.tensor(labels.values, dtype=torch.float32)

        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item['labels'] = self.labels[idx]
            return item

        def __len__(self):
            return len(self.labels)

    train_dataset = ChemDataset(smiles_train, labels_train)
    val_dataset = ChemDataset(smiles_val, labels_val)


    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        labels = labels.astype(np.float32)  # (N, L)
        probs = 1 / (1 + np.exp(-logits))  # sigmoid

        metrics = {}
        # ROC-AUC (if each label has both classes in the eval set)
        try:
            metrics["roc_auc_micro"] = roc_auc_score(labels, probs, average="micro")
            metrics["roc_auc_macro"] = roc_auc_score(labels, probs, average="macro")
        except ValueError:
            # happens if a label has only one class present in y_true
            pass

        # PR-AUC (micro)
        try:
            metrics["pr_auc_micro"] = average_precision_score(labels, probs, average="micro")
        except ValueError:
            pass

        # Example thresholding at 0.5 for f1-ish proxy (not exact F1, just quick stats)
        preds = (probs >= 0.5).astype(np.float32)
        tp = (preds * labels).sum()
        fp = (preds * (1 - labels)).sum()
        fn = ((1 - preds) * labels).sum()
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        metrics.update(dict(precision=precision, recall=recall, f1=f1))
        return metrics

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(label_cols),
        problem_type="multi_label_classification"
    )

    training_args = TrainingArguments(
        output_dir="./chemberta_mlc",
        eval_strategy="epoch",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=50,
        learning_rate=5e-5,
        load_best_model_at_end=False,
        report_to = ["tensorboard"]
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )
    trainer.train()
