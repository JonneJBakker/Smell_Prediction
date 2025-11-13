#!/usr/bin/env python3
"""
visualize_attention.py

Load a trained attention-pooling ChemBERTa model and visualize
a bar chart of token-level attention weights for a SMILES string.

Usage:
  python visualize_attention.py \
    --model_dir /path/to/.../chemberta_multilabel \
    --smiles "CC1=CC(=O)NC(=O)N1" \
    [--max_len 512] [--no_show] [--out_png attention.png]
"""
import os
import json
import argparse
import math

import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F

import os
os.environ.setdefault("MPLBACKEND", "Agg")  # force non-GUI backend
import matplotlib.pyplot as plt
from transformers import RobertaTokenizerFast
from utils.chemberta_workflows import ChembertaMultiLabelClassifier, DEFAULT_PRETRAINED_NAME  # make sure this is on PYTHONPATH

SPECIAL_TOKENS = {"<s>", "</s>", "<pad>"}

def _device_of(model):
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")

def _load_hparams(model_dir):
    hp_path = os.path.join(model_dir, "hyperparameters.json")
    with open(hp_path, "r") as f:
        hp = json.load(f)
    # we need num_labels; optionally reuse hidden_channels/num_mlp_layers/dropout if your class expects them
    return {
        "num_labels": hp["num_labels"],
        "hidden_channels": hp.get("hidden_channels", 100),
        "num_mlp_layers": hp.get("num_mlp_layers", 1),
        "dropout": hp.get("dropout", 0.3),
    }

def _load_model(model_dir, device):
    # tokenizer was saved here during training
    tokenizer = RobertaTokenizerFast.from_pretrained(model_dir)

    # model hyperparams
    hp = _load_hparams(model_dir)

    # instantiate classifier (must be the ATTENTION-POOLING version)
    model = ChembertaMultiLabelClassifier(
        pretrained=DEFAULT_PRETRAINED_NAME,
        num_labels=hp["num_labels"],
        dropout=hp["dropout"],
        hidden_channels=hp["hidden_channels"],
        num_mlp_layers=hp["num_mlp_layers"],
        pos_weight=None,
    ).to(device)

    # load weights
    # prefer your explicit file; else fall back to HF-style file if you saved via trainer
    state_candidates = [
        os.path.join(model_dir, "chemberta_multilabel_model_final.bin"),
        os.path.join(model_dir, "pytorch_model.bin"),
    ]
    state_path = next((p for p in state_candidates if os.path.exists(p)), None)
    if state_path is None:
        raise FileNotFoundError(
            f"Could not find model weights in: {state_candidates}. "
            "Make sure you passed the correct --model_dir."
        )

    state = torch.load(state_path, map_location=device)
    model.load_state_dict(state, strict=False)
    model.eval()

    return model, tokenizer

def _masked_softmax(scores, mask, dim=-1):
    scores = scores.masked_fill(mask == 0, float("-inf"))
    return torch.softmax(scores, dim=dim)

def _tokens_and_attention(model, tokenizer, smiles, device, max_len=512):
    enc = tokenizer(
        smiles,
        return_tensors="pt",
        padding=False,
        truncation=True,
        max_length=max_len,
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    with torch.no_grad():
        # Forward once through your model.
        # If your model caches attention as `last_attn_weights`, we can read it directly.
        _ = model(input_ids=input_ids, attention_mask=attention_mask)

        if getattr(model, "last_attn_weights", None) is not None:
            alpha = model.last_attn_weights[0]  # (L,)
        else:
            # Recompute using the model's attention block (works if your class defines `self.attention`).
            outputs = model.roberta(input_ids=input_ids, attention_mask=attention_mask)
            token_embs = outputs.last_hidden_state  # (1, L, H)
            scores = model.attention(token_embs).squeeze(-1)  # (1, L)
            alpha = _masked_softmax(scores, attention_mask, dim=1)[0]

    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
    mask_row = attention_mask[0].tolist()

    # filter padding/specials and renormalize
    kept_tokens, kept_weights = [], []
    for t, w, m in zip(tokens, alpha.tolist(), mask_row):
        if m == 0:  # pad
            continue
        if t in SPECIAL_TOKENS:
            continue
        if not math.isfinite(float(w)):
            continue
        kept_tokens.append(t)
        kept_weights.append(float(w))

    # normalize to sum=1 over kept tokens
    s = sum(kept_weights) or 1.0
    kept_weights = [w / s for w in kept_weights]

    return kept_tokens, kept_weights
def _predict_labels(model, tokenizer, smiles, device, threshold=0.5, max_len=512):
    """
    Run the model to get probabilities and binary predictions for one SMILES string.
    Returns (probs: np.ndarray, preds: np.ndarray)
    """
    enc = tokenizer(
        smiles,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_len,
    ).to(device)

    with torch.no_grad():
        outputs = model(**enc)
        logits = outputs.logits.detach().cpu().numpy()
        probs = 1 / (1 + np.exp(-logits))  # sigmoid
        preds = (probs >= threshold).astype(int)

    return probs[0], preds[0]


def make_attention_table(model, tokenizer, smiles, target_cols=None,
                         threshold=0.5, device=None, max_len=512,
                         out_csv=None):
    """
    Creates a pandas DataFrame with:
      token, weight, rank_desc, and *only* the labels predicted as 1.
    Optionally writes it to CSV.
    """
    if device is None:
        device = next(model.parameters()).device

    # attention weights
    tokens, weights = _tokens_and_attention(model, tokenizer, smiles, device, max_len)

    # predictions
    probs, preds = _predict_labels(model, tokenizer, smiles, device, threshold, max_len)

    # build tidy DataFrame
    df = pd.DataFrame({
        "token": tokens,
        "weight": weights,
        "rank_desc": pd.Series(weights).rank(ascending=False, method="dense").astype(int),
    })

    # ----- Extract only predicted labels with value 1 -----
    if target_cols is not None and len(target_cols) == len(preds):
        positive_labels = [lbl for lbl, p in zip(target_cols, preds) if p == 1]
    else:
        # fallback: show indices if no label names
        positive_labels = [str(i) for i, p in enumerate(preds) if p == 1]

    df["predicted_labels"] = ", ".join(positive_labels) if positive_labels else "(none)"
    # optionally include probabilities only for predicted=1 labels
    if target_cols is not None and len(target_cols) == len(preds):
        pos_probs = {lbl: round(float(probs[i]), 3)
                     for i, lbl in enumerate(target_cols) if preds[i] == 1}
        df["predicted_probs"] = str(pos_probs)
    else:
        df["predicted_probs"] = str([])

    if out_csv:
        df.to_csv(out_csv, index=False)
        print(f"Saved attention+predictions table to: {out_csv}")

    return df


def _plot_bar(tokens, weights, title=None, out_png="plots/attention_plot.png", show=True):
    plt.figure(figsize=(min(18, 0.6 * max(len(tokens), 5)), 3.5))
    plt.bar(range(len(tokens)), weights)
    plt.xticks(range(len(tokens)), tokens, rotation=90)
    plt.ylabel("Attention weight")
    if title:
        plt.title(title)
    plt.tight_layout()
    if out_png:
        plt.savefig(out_png, dpi=200)
        print(f"Saved: {out_png}")
    if show:
        plt.show()
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True, help="Path to trained model folder (where hyperparameters.json lives)")
    ap.add_argument("--smiles", required=True, help="SMILES string to visualize")
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--out_png", default=None, help="Optional path to save the bar chart")
    ap.add_argument("--no_show", action="store_true", help="Do not open a window; useful on headless servers")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = _load_model(args.model_dir, device)

    tokens, weights = _tokens_and_attention(model, tokenizer, args.smiles, device, max_len=args.max_len)

    # inside main(), after computing tokens, weights
    table_path = "attention_with_preds.csv"

    val = pd.read_csv("Data/splits/val_stratified.csv")

    target_cols = [col for col in val.columns if col not in ['nonStereoSMILES']]
    df = make_attention_table(model, tokenizer, args.smiles,
                              target_cols=target_cols,  # or pass your label names list
                              out_csv=table_path)
    print(df.head())

    title = f"Attention over tokens for: {args.smiles}"
    _plot_bar(tokens, weights, title=title, out_png=args.out_png, show=not args.no_show)

if __name__ == "__main__":
    main()
