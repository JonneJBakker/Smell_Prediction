import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

f = 1.2
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 14*f,
    "axes.titlesize": 16*f,
    "axes.labelsize": 14*f,
    "xtick.labelsize": 12*f,
    "ytick.labelsize": 12*f,
    "legend.fontsize": 11*f,
    "figure.dpi": 300,
})

def main():
    # Load
    frozen = pd.read_csv("../trained_models/FROZEN_FINAL/per_label_metrics.csv")
    lora   = pd.read_csv("../trained_models/LORA_FINAL/per_label_metrics.csv")
    mpnn   = pd.read_csv("../Data/mpnn_per_label_metrics.csv")

    # Keep only what we need
    frozen = frozen[["label", "f1", "frequency"]].rename(columns={"f1": "f1_frozen"})
    lora   = lora[["label", "f1"]].rename(columns={"f1": "f1_lora"})
    mpnn   = mpnn[["label", "f1"]].rename(columns={"f1": "f1_mpnn"})

    # Align by label
    merged = (
        frozen
        .merge(lora, on="label", how="inner")
        .merge(mpnn, on="label", how="inner")
    )

   # sanity checks
    assert merged["label"].is_unique
    assert not merged.isna().any().any()

    # Sort for visualization
    K = 30

    topk = (
        merged
        .sort_values("frequency", ascending=False)
        .head(K)
        .reset_index(drop=True)
    )

    print(merged.head())
    # Plot
    x = np.arange(len(topk))
    width = 0.25

    plt.figure(figsize=(12, 5))

    plt.bar(x - width, topk["f1_mpnn"], width, label="MPNN")
    plt.bar(x, topk["f1_frozen"], width, label="ChemBERTa (Frozen)")
    plt.bar(x + width, topk["f1_lora"], width, label="ChemBERTa + LoRA")

    plt.xticks(x, topk["label"], rotation=45, ha="right")
    plt.ylabel("Per-label F1-score")
    plt.xlabel("Most frequent labels")
    plt.title(f"Per-label F1 Comparison of Most Frequent Labels")
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"Top_{K}_per_label_f1_bar_chart.pdf", bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()

