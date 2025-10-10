import pandas as pd
import numpy as np
import torch

from torch.utils.data.dataset import Dataset
from transformers import TrainingArguments, Trainer, AutoTokenizer, AutoModelForSequenceClassification
from sklearn.model_selection import train_test_split

MODEL_NAME = "DeepChem/ChemBERTa-77M-MLM"

from Data.data_prep import split_data


def train_chemBerta(df, smiles_col):
    x_train, x_test, y_train, y_test, label_cols = split_data(df, smiles_col=smiles_col)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

    train_encodings = tokenizer(list(x_train), truncation=True, padding=True, max_length=512, return_tensors="pt")
    test_encodings = tokenizer(list(x_test), truncation=True, padding=True, max_length=512, return_tensors="pt")

    class ChemDataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = torch.tensor(labels.values, dtype=torch.float32)

        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item['labels'] = self.labels[idx]
            return item

        def __len__(self):
            return len(self.labels)

    train_dataset = ChemDataset(train_encodings, y_train)
    test_dataset = ChemDataset(test_encodings, y_test)

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
        num_train_epochs=5,
        learning_rate=5e-5,
        load_best_model_at_end=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )
    trainer.train()