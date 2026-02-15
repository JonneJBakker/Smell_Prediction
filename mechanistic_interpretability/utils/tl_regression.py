"""Regression lens analysis module for ChemBERTa interpretability.

This module implements "regression lens" analysis by applying the final readout layer
after each transformer block to see what predictions the model would make at
each intermediate layer. This reveals how the model's "thinking" evolves through
the layers.

Key functions:
- run_regression_lens(): Apply readout after each layer for one or more molecules
- plot_regression_lens_results(): Visualize prediction changes through layers
- compare_molecules_regression_lens(): Compare regression lens patterns across molecules
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, roc_auc_score, f1_score, accuracy_score, hamming_loss, jaccard_score
import torch
import transformer_lens as tl
from transformers import RobertaTokenizerFast

from .tl_conversion import FaithfulTLRegressor

plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
})

def run_classification_lens(
        tl_model: tl.HookedEncoder,
        classifier: FaithfulTLRegressor,
        smiles: List[str],
        tokenizer: RobertaTokenizerFast,
        device: Optional[str] = None,
        batch_size: int = 32,
        threshold: float = 0.3,
) -> Dict:
    """
    Classification lens for multi-label ChemBERTa.

    For each layer (including embeddings) we:
      * take the full sequence hidden states from the TL cache
      * apply the SAME pooling strategy as the classifier (max_mean/cls_mean/etc.)
      * run the pooled vector through classifier.mlp
      * return per-layer probabilities (or logits if you prefer)
    """
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    tl_model = tl_model.to(device).eval()
    classifier = classifier.to(device).eval()

    pooling_strat = getattr(classifier, "pooling_strat", "cls")

    def mean_pool_token_embs(token_embs: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # token_embs: [seq_len, hidden]
        # attention_mask: [seq_len]
        mask = attention_mask.unsqueeze(-1).type_as(token_embs)   # [seq_len, 1]
        summed = (token_embs * mask).sum(dim=0)                   # [hidden]
        denom = mask.sum(dim=0).clamp(min=1e-9)
        return summed / denom                                     # [hidden]

    results = {}

    for batch_start in range(0, len(smiles), batch_size):
        batch_smiles = smiles[batch_start:batch_start + batch_size]

        inputs = tokenizer(
            batch_smiles,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(device)

        with torch.no_grad():
            _, cache = tl_model.run_with_cache(
                inputs["input_ids"],
                one_zero_attention_mask=inputs["attention_mask"],
            )

            for batch_idx, smile in enumerate(batch_smiles):
                smile_results = {}
                attn_mask = inputs["attention_mask"][batch_idx]  # [seq_len]

                for layer in range(tl_model.cfg.n_layers + 1):
                    if layer == 0:
                        # embedding layer outputs: [batch, seq, d_model]
                        token_embs = cache["hook_full_embed"][batch_idx]  # [seq_len, d_model]
                        layer_name = "Embedding"
                    else:
                        token_embs = cache[f"blocks.{layer-1}.hook_normalized_resid_post"][batch_idx]  # [seq_len, d_model]
                        layer_name = f"{layer}"

                    # --- replicate ChembertaMultiLabelClassifier pooling ---
                    strat = pooling_strat

                    if strat == "cls_mean":
                        cls_emb = token_embs[0, :]  # [d_model]
                        mean_pooled = mean_pool_token_embs(token_embs, attn_mask)  # [d_model]
                        pooled = torch.cat([mean_pooled, cls_emb], dim=-1)        # [2*d_model]

                    elif strat == "max_mean":
                        mean_pooled = mean_pool_token_embs(token_embs, attn_mask)  # [d_model]

                        mask_bool = attn_mask.unsqueeze(-1).bool()                 # [seq_len, 1]
                        masked_token_embs = token_embs.masked_fill(~mask_bool, float("-inf"))
                        max_pooled, _ = masked_token_embs.max(dim=0)               # [d_model]

                        pooled = torch.cat([mean_pooled, max_pooled], dim=-1)      # [2*d_model]

                    elif strat == "mean":
                        pooled = mean_pool_token_embs(token_embs, attn_mask)       # [d_model]

                    elif strat == "cls":
                        pooled = token_embs[0, :]                                   # [d_model]

                    elif strat == "attention":
                        # query_vector: [d_model]
                        q = classifier.query_vector
                        # attn_scores: [seq_len]
                        attn_scores = torch.matmul(token_embs, q)                   # [seq_len]
                        attn_weights = torch.softmax(attn_scores, dim=0).unsqueeze(-1)  # [seq_len, 1]
                        pooled = torch.sum(token_embs * attn_weights, dim=0)        # [d_model]

                    else:
                        raise ValueError(f"Unknown pooling_strat '{strat}'")

                    # Apply same dropout + MLP as classifier
                    x = classifier.dropout(pooled)               # [feat_dim]
                    logits = classifier.mlp_head(x.unsqueeze(0))      # [1, num_labels]
                    probs = torch.sigmoid(logits).squeeze(0)     # [num_labels]

                    # store probabilities (or logits if you prefer)
                    smile_results[layer_name] = {
                        "logits": logits.squeeze(0).cpu().numpy(),
                        "probs": probs.cpu().numpy(),
                        "binary": (probs >= threshold).cpu().numpy(),
                    }

                results[smile] = smile_results

    return results


def run_regression_lens(
        tl_model: tl.HookedEncoder,
        regressor: FaithfulTLRegressor,
        scaler,
        smiles: List[str],
        tokenizer: RobertaTokenizerFast,
        device: Optional[str] = None,
        batch_size: int = 32,
) -> Dict:
    """Run regression lens analysis for one or more molecules (batched for efficiency).
    
    Applies the readout layer after each transformer block to see what
    the model would predict based on representations at each layer.
    
    Args:
        tl_model: TransformerLens encoder model
        regressor: Full regressor with MLP head for readout
        scaler: Scaler for denormalization
        smiles: List of SMILES strings to analyze
        tokenizer: Tokenizer for the model
        device: Device to use for computation
        batch_size: Number of molecules to process at once (default: 32)
        
    Returns:
        Dictionary with layer-wise predictions for each molecule
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    results = {}
    
    # Process in batches for efficiency
    for batch_start in range(0, len(smiles), batch_size):
        batch_smiles = smiles[batch_start:batch_start + batch_size]
        
        # Tokenize batch
        inputs = tokenizer(batch_smiles, return_tensors="pt", padding=True, truncation=True).to(device)
        
        with torch.no_grad():
            # Run model with cache for entire batch
            _, cache = tl_model.run_with_cache(
                inputs["input_ids"],
                one_zero_attention_mask=inputs["attention_mask"]
            )

            # Process each molecule in batch
            for batch_idx, smile in enumerate(batch_smiles):
                result = {}
                
                for layer in range(tl_model.cfg.n_layers + 1):
                    if layer == 0:
                        # After embedding but before any transformer blocks
                        representation = cache["hook_full_embed"][batch_idx, 0, :]
                        layer_name = "Embedding"
                    else:
                        representation = cache[f"blocks.{layer-1}.hook_normalized_resid_post"][batch_idx, 0, :]
                        layer_name = f"{layer}"
                
                    norm_prediction = regressor.mlp_head(representation).squeeze().item()
                    prediction = norm_prediction * scaler.scale_[0] + scaler.mean_[0]

                    result[layer_name] = prediction

                results[smile] = result
    
    return results

def compare_molecule_groups_regression_lens(
        tl_model: tl.HookedEncoder,
        regressor: FaithfulTLRegressor,
        scaler,
        group_smiles: Dict,
        tokenizer: RobertaTokenizerFast,
        targets: List[float],
        results_dir: Optional[str] = None,
        device: Optional[str] = None,
        batch_size: int = 64,
) -> Dict:
    """Run regression lens analysis for one or more molecule groups (batched).
    
    Applies the readout layer after each transformer block to see what
    the model would predict based on representations at each layer.
    
    Args:
        tl_model: TransformerLens encoder model
        regressor: Full regressor with MLP head for readout
        scaler: Scaler for denormalization
        group_smiles: Dictionary mapping group names to lists of SMILES
        tokenizer: Tokenizer for the model
        targets: Optional list of actual target values (same order as concatenated smiles)
        results_dir: Optional directory to save predictions CSV
        device: Device to use for computation
        batch_size: Number of molecules to process at once (default: 64)
        
    Returns:
        Dictionary with layer-wise predictions for each molecule group, 
        plus 'variance_ratio' key with variance ratios if targets provided
    """
    results = {}
    all_smiles_ordered = []
    
    # Track start/end indices for each group's targets
    group_target_indices = {}
    current_idx = 0
    
    for group, smiles in group_smiles.items():
        print(f"Processing {group}: {len(smiles)} molecules...")
        group_results = run_regression_lens(tl_model, regressor, scaler, smiles, tokenizer, device, batch_size)
        results[group] = group_results
        all_smiles_ordered.extend(smiles)
        
        # Store the indices for this group's targets
        group_target_indices[group] = (current_idx, current_idx + len(smiles))
        current_idx += len(smiles)

        # Compute per-layer mean and std across all molecules in the group
        layer_names = list(next(iter(group_results.values())).keys())
        means = {}
        variances = {}

        for layer in layer_names:
            values = np.array([group_results[smile][layer] for smile in smiles], dtype=np.float64)
            means[layer] = np.mean(values)
            variances[layer] = np.var(values)

        results[group]["mean"] = means
        results[group]["variance"] = variances
        
    # Collect all predictions across all groups for each layer
    all_predictions_by_layer = {}
    for layer in layer_names:
        all_preds = []
        for group, smiles in group_smiles.items():
            for smile in smiles:
                all_preds.append(results[group][smile][layer])
        all_predictions_by_layer[layer] = np.array(all_preds)
    
    # Compute variance ratio
    target_variance = np.var(targets)
    variance_ratios = {}
    for layer in layer_names:
        pred_variance = np.var(all_predictions_by_layer[layer])
        variance_ratios[layer] = pred_variance / target_variance

    print("Targets (first 20):", targets[:20])
    print(f"max target and min: {min(targets), max(targets)}")
    for layer in layer_names:
        print(f"max and min are {max(all_predictions_by_layer[layer]), min(all_predictions_by_layer[layer])}")
        print(f"Layer {layer} predictions (first 20):", all_predictions_by_layer[layer][:20])
    
    # Compute global R^2 per layer
    r2_scores = {}
    for layer in layer_names:
            r2_scores[layer] = r2_score(targets, all_predictions_by_layer[layer])
    
    # Compute per-group R^2 per layer
    for group, smiles in group_smiles.items():
        start_idx, end_idx = group_target_indices[group]
        group_targets = targets[start_idx:end_idx]
        group_r2_scores = {}
        
        for layer in layer_names:
            # Collect predictions for this group at this layer
            group_predictions = np.array([results[group][smile][layer] for smile in smiles])
            group_r2_scores[layer] = r2_score(group_targets, group_predictions)
        
        results[group]["r2_scores"] = group_r2_scores
        print(f"{group} R² scores by layer: {group_r2_scores}")
    
    results["target_variance"] = target_variance
    results["variance_ratio"] = variance_ratios
    results["r2_scores"] = r2_scores
    print(f"Variance ratios by layer: {variance_ratios}")
    print(f"Global R² scores by layer: {r2_scores}")
    
    # Save predictions to CSV if directory provided
    if results_dir is not None:
        os.makedirs(results_dir, exist_ok=True)
            
    return results

def compare_molecule_groups_classification_lens(
        tl_model: tl.HookedEncoder,
        classifier: FaithfulTLRegressor,
        group_smiles: Dict[str, List[str]],
        tokenizer: RobertaTokenizerFast,
        targets: np.ndarray,
        threshold: float = 0.35,
        results_dir: Optional[str] = None,
        device: Optional[str] = None,
        batch_size: int = 64,
) -> Dict:
    """
    Classification lens analysis for molecule groups (multi-label).

    For each layer, apply the classifier head to the representation and compute
    multi-label metrics (micro/macro F1, AUROC, etc.) across all molecules, and
    optionally per group.

    Args:
        tl_model: TransformerLens encoder model
        classifier: Full classifier with multi-label MLP head
        group_smiles: dict[group_name] -> list of SMILES
        tokenizer: tokenizer used for the model
        targets: 2D array of shape (N, num_labels) in the same order as
                 concatenated group_smiles values
        threshold: threshold for converting probabilities to 0/1
        results_dir: optional directory to save any CSV/plots (not used here yet)
        device: device to use
        batch_size: batch size for lens

    Returns:
        results dict with:
            - results[group][smile][layer] = {"logits", "probs", "preds"}
            - results["layer_metrics"][layer_name] = {... per-layer metrics ...}
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    results: Dict = {}
    all_smiles_ordered: List[str] = []

    group_target_indices: Dict[str, tuple[int, int]] = {}
    current_idx = 0

    # 1) Run classification lens per group and track indices
    for group, smiles in group_smiles.items():
        print(f"Processing {group}: {len(smiles)} molecules...")
        group_results = run_classification_lens(
            tl_model=tl_model,
            classifier=classifier,
            smiles=smiles,
            tokenizer=tokenizer,
            device=device,
            batch_size=batch_size,
            threshold=threshold,
        )
        results[group] = group_results
        all_smiles_ordered.extend(smiles)

        group_target_indices[group] = (current_idx, current_idx + len(smiles))
        current_idx += len(smiles)

    # Sanity: targets must match number of total smiles
    assert targets.shape[0] == len(all_smiles_ordered), (
        f"targets.shape[0]={targets.shape[0]} != number of smiles={len(all_smiles_ordered)}"
    )

    # 2) Collect logits per layer across all groups
    # Use any group to get layer names
    first_group = next(iter(results.values()))
    first_smile = next(iter(first_group.values()))
    layer_names = list(first_smile.keys())  # e.g. ["Embedding", "1", "2", ...]

    logits_by_layer = {layer: [] for layer in layer_names}

    for layer in layer_names:
        for group, smiles in group_smiles.items():
            for smile in smiles:
                logits_by_layer[layer].append(results[group][smile][layer]["logits"])
        logits_by_layer[layer] = np.stack(logits_by_layer[layer], axis=0)  # (N, num_labels)

    # 3) Compute per-layer multi-label metrics
    y_true = targets.astype(int)
    layer_metrics: Dict[str, Dict[str, float]] = {}

    for layer in layer_names:
        logits = logits_by_layer[layer]
        probs = 1.0 / (1.0 + np.exp(-logits))
        y_pred = (probs >= threshold).astype(int)

        # Some models/labels may have degenerate cases; guard with try/except for AUROC
        try:
            auroc = roc_auc_score(y_true, probs, average="macro")
        except ValueError:
            auroc = float("nan")

        micro_f1 = f1_score(y_true, y_pred, average="micro", zero_division=0)
        macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        samples_f1 = f1_score(y_true, y_pred, average="samples", zero_division=0)
        micro_acc = accuracy_score(y_true, y_pred)
        hamming = hamming_loss(y_true, y_pred)
        jaccard_samples = jaccard_score(y_true, y_pred, average="samples", zero_division=0)

        layer_metrics[layer] = {
            "auroc": auroc,
            "micro_f1": micro_f1,
            "macro_f1": macro_f1,
            "samples_f1": samples_f1,
            "micro_accuracy": micro_acc,
            "hamming_loss": hamming,
            "jaccard_samples": jaccard_samples,
        }


    results["cluster_layer_f1"] = {}  # group -> layer -> f1 metrics

    for group, smiles in group_smiles.items():
        start_idx, end_idx = group_target_indices[group]
        y_true_g = targets[start_idx:end_idx].astype(int)  # (n_g, num_labels)

        per_layer = {}
        for layer in layer_names:
            logits_g = np.stack(
                [results[group][smile][layer]["logits"] for smile in smiles],
                axis=0,
            )
            probs_g = 1.0 / (1.0 + np.exp(-logits_g))
            y_pred_g = (probs_g >= threshold).astype(int)

            # 🔑 Mask labels that actually appear in this cluster
            present = y_true_g.sum(axis=0) > 0

            if present.any():
                macro_f1_present = f1_score(
                    y_true_g[:, present],
                    y_pred_g[:, present],
                    average="macro",
                    zero_division=0,
                )
            else:
                macro_f1_present = np.nan

            per_layer[layer] = {
                "micro_f1": f1_score(y_true_g, y_pred_g, average="micro", zero_division=0),
                "macro_f1": f1_score(y_true_g, y_pred_g, average="macro", zero_division=0),
                "macro_f1_present": macro_f1_present,  # ✅ ADD THIS
                "samples_f1": f1_score(y_true_g, y_pred_g, average="samples", zero_division=0),
                "n_labels_present": int(present.sum()),  # optional but VERY useful
            }

        results["cluster_layer_f1"][group] = per_layer

    results["layer_metrics"] = layer_metrics

    if results_dir is not None:
        os.makedirs(results_dir, exist_ok=True)


    return results

import numpy as np
from sklearn.metrics import f1_score

def compute_macro_f1_per_layer(
    lens_results: dict,
    smiles_list: list,
    targets: np.ndarray,
    threshold: float = 0.5,
):
    """
    Compute macro-F1 per layer from classification lens outputs.

    Args:
        lens_results: output of run_classification_lens(smiles=smiles_list, ...)
                      structure:
                        lens_results[smile][layer_name]["probs"] -> [num_labels]
        smiles_list: list of SMILES in the SAME order as rows in `targets`
        targets: array of shape (N, num_labels) with 0/1 labels
        threshold: probability threshold to binarize predictions

    Returns:
        dict[layer_name] -> macro_f1
    """
    # use the first molecule to get the list of layers
    first_smile = smiles_list[0]
    layer_names = list(lens_results[first_smile].keys())

    y_true = targets.astype(int)
    layer_macro_f1 = {}

    for layer in layer_names:
        # collect probs for all molecules at this layer
        probs_list = []
        for smile in smiles_list:
            probs_list.append(lens_results[smile][layer]["probs"])
        probs = np.stack(probs_list, axis=0)          # (N, num_labels)

        # threshold → predictions
        y_pred = (probs >= threshold).astype(int)

        # macro F1 over labels
        macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        layer_macro_f1[layer] = macro_f1

    return layer_macro_f1


def plot_individual_molecules_classification_lens(
    results,
    label_idx: int,
    results_dir: str,
    molecule_labels=None,
    target_values=None,
):
    """
    Plot probability for one label across layers for each molecule.
    """

    os.makedirs(results_dir, exist_ok=True)

    smiles_list = list(results.keys())
    if molecule_labels is None:
        molecule_labels = smiles_list

    # assume all molecules have same layer names
    first_smile = smiles_list[0]
    layer_names = list(results[first_smile].keys())
    n_layers = len(layer_names)

    for i, smile in enumerate(smiles_list):
        probs_per_layer = [
            results[smile][layer]["probs"][label_idx] for layer in layer_names
        ]

        plt.figure(figsize=(8, 4))
        plt.plot(range(n_layers), probs_per_layer, "o-")
        plt.xticks(range(n_layers), layer_names, rotation=45)
        plt.ylim(0, 1)
        title = molecule_labels[i]
        if target_values is not None:
            title += f" (true={target_values[i]})"
        plt.title(title)
        plt.ylabel("P(label=1)")
        plt.xlabel("Layer")
        plt.tight_layout()
        plt.savefig(Path(results_dir) / f"molecule_{i+1}_label{label_idx}.pdf", dpi=200)
        plt.close()

def plot_individual_molecules_regression_lens(
        results: dict,
        results_dir: str = "results/regression_lens",
        x_axis_labels: list = ["After Embedding \n Layer", "After Transformer \n Layer 1", "After Transformer \n Layer 2", "After Transformer \n Layer 3"],
        molecule_labels: list = ["Molecule 0", "Molecule 1", "Molecule 2"],
        y_label: str = "Log Solubility",
        title: str = "ESOL",
        actual_targets: Optional[List[float]] = None,
        target_labels: Optional[List[str]] = None
):
    """Plot regression lens results for individual molecules.
    
    Args:
        results: Dictionary mapping SMILES to layer-wise predictions
        results_dir: Directory to save the plot
        x_axis_labels: Labels for x-axis (layers)
        molecule_labels: Labels for each molecule
        y_label: Label for y-axis
        title: Plot title
        actual_targets: Optional list of actual target values for each molecule (in same order as results)
        target_labels: Optional list of labels for target types (e.g., ["max", "median", "min"])
    """
    os.makedirs(results_dir, exist_ok=True)

    plt.figure(figsize=(12, 8))
    
    # Plot predictions for each molecule
    for i, (smile, smile_results) in enumerate(results.items()):
        # Add target label if provided
        label = molecule_labels[i]
        if target_labels and i < len(target_labels):
            label = f"{molecule_labels[i]} ({target_labels[i]})"
        
        plt.plot(range(len(smile_results)), smile_results.values(), 'o-', alpha=0.7, label=label)
        
        # Add dashed horizontal line for actual target value if provided
        if actual_targets and i < len(actual_targets):
            plt.axhline(y=actual_targets[i], color=f'C{i}', linestyle='--', alpha=0.5, linewidth=2)

    plt.title(title, fontsize=18)
    plt.ylabel(y_label, fontsize=16)
    plt.xticks(range(len(smile_results)), x_axis_labels, rotation=45, fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper left', fontsize=16)
    plt.tight_layout()
    plt.savefig(Path(results_dir) / "individual_molecules.pdf", dpi=300, bbox_inches="tight")
    plt.close()

def plot_group_molecules_regression_lens(
        results: dict,
        results_dir: str = "results/regression_lens",
        x_axis_labels: list = ["After Embedding \n Layer", "After Transformer \n Layer 1", "After Transformer \n Layer 2", "After Transformer \n Layer 3"],
        mean_y_label: str = "Mean Log Solubility",
        var_y_label: str = "Variance Log Solubility",
        title: str = "ESOL",
):
    os.makedirs(results_dir, exist_ok=True)

    # Define special keys that are not group data
    special_keys = {"variance_ratio", "target_variance", "r2_scores"}
    
    # Determine layer order from the first group's mean dict (skip special keys)
    first_group = next(iter({k: v for k, v in results.items() if k not in special_keys}.values()))
    layer_names = list(first_group["mean"].keys())
    
    # Set up continuous color palette for all groups (excluding special keys)
    group_items = [(k, v) for k, v in results.items() if k not in special_keys]
    n_groups = len(group_items)
    colors = plt.cm.turbo(np.linspace(0, 1, n_groups))

    # Plot group means
    plt.figure(figsize=(12, 8))
    for i, (group_name, group_data) in enumerate(group_items):
        mean_values = [group_data["mean"][layer] for layer in layer_names]
        plt.plot(range(len(layer_names)), mean_values, 'o-', alpha=0.8, label=group_name, color=colors[i])

    plt.title(title, fontsize=18)
    plt.ylabel(mean_y_label, fontsize=16)
    plt.xticks(range(len(layer_names)), x_axis_labels, rotation=45, fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper left', fontsize=12)
    plt.tight_layout()
    plt.savefig(Path(results_dir) / "group_means.pdf", dpi=300, bbox_inches="tight")
    plt.close()

    # Plot group variances
    plt.figure(figsize=(12, 8))
    for i, (group_name, group_data) in enumerate(group_items):
        std_values = [group_data["variance"][layer] for layer in layer_names]
        plt.plot(range(len(layer_names)), std_values, 'o-', alpha=0.8, label=group_name, color=colors[i])

    plt.title(title, fontsize=18)
    plt.ylabel(var_y_label, fontsize=16)
    plt.xticks(range(len(layer_names)), x_axis_labels, rotation=45, fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper left', fontsize=12)
    plt.tight_layout()
    plt.savefig(Path(results_dir) / "group_vars.pdf", dpi=300, bbox_inches="tight")
    plt.close()
    
    # Plot variance ratio if it exists in results
    if "variance_ratio" in results:
        variance_ratios = results["variance_ratio"]
        ratios = [variance_ratios[layer] for layer in layer_names]
        
        plt.figure(figsize=(12, 8))
        # Use a color from the turbo colormap for consistency
        variance_color = plt.cm.turbo(0.3)
        plt.plot(range(len(layer_names)), ratios, 'o-', linewidth=2, markersize=10, color=variance_color)
                
        plt.title(f"{title} - Variance Ratio", fontsize=18)
        plt.ylabel("Variance Ratio", fontsize=16)
        plt.xticks(range(len(layer_names)), x_axis_labels, rotation=45, fontsize=14)
        plt.yticks(fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(loc='best', fontsize=14)
        plt.tight_layout()
        plt.savefig(Path(results_dir) / "variance_ratio.pdf", dpi=300, bbox_inches="tight")
        plt.close()

    # Plot variance ratio times R^2 if it exists in results
    if "variance_ratio" and "r2_scores" in results:
        variance_ratios = results["variance_ratio"]
        r2_scores = results["r2_scores"]
        ratios = [variance_ratios[layer] * r2_scores[layer] for layer in layer_names]
        
        plt.figure(figsize=(12, 8))
        # Use a color from the turbo colormap for consistency
        combined_color = plt.cm.turbo(0.5)
        plt.plot(range(len(layer_names)), ratios, 'o-', linewidth=2, markersize=10, color=combined_color)
                
        plt.title(f"{title} - Variance Ratio × R²", fontsize=18)
        plt.ylabel("Variance Ratio × R²", fontsize=16)
        plt.xticks(range(len(layer_names)), x_axis_labels, rotation=45, fontsize=14)
        plt.yticks(fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(loc='best', fontsize=14)
        plt.tight_layout()
        plt.savefig(Path(results_dir) / "variance_ratio_R2.pdf", dpi=300, bbox_inches="tight")
        plt.close()

    # Plot variance and R^2 if it exists in results
    if "variance_ratio" and "r2_scores" in results:
        variance_ratios = results["variance_ratio"]
        r2_scores = results["r2_scores"]
        
        # Extract values in the same order as layer_names
        variance_ratio_values = [variance_ratios[layer] for layer in layer_names]
        r2_values = [r2_scores[layer] for layer in layer_names]
        
        plt.figure(figsize=(12, 8))
        metric_colors = plt.cm.turbo(np.linspace(0, 1, 2))

        plt.plot(range(len(layer_names)), variance_ratio_values, 'o-', linewidth=2, markersize=10, color=metric_colors[0], label='Variance Ratio')
        plt.plot(range(len(layer_names)), r2_values, 'o-', linewidth=2, markersize=10, color=metric_colors[1], label='R²')
        
        plt.title(f"{title} - Variance ratio and R²", fontsize=18)
        plt.ylabel("Variance Ratio and R²", fontsize=16)
        plt.xticks(range(len(layer_names)), x_axis_labels, rotation=45, fontsize=14)
        plt.yticks(fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(loc='best', fontsize=14)
        plt.tight_layout()
        plt.savefig(Path(results_dir) / "variance_ratio_and_R2.pdf", dpi=300, bbox_inches="tight")
        plt.close()

    # Plot variance ratio × R² for each cluster/group
    if "variance_ratio" and "r2_scores" and "target_variance" in results:
        r2_scores = results["r2_scores"]
        r2_values = [r2_scores[layer] for layer in layer_names]
        target_variance = results["target_variance"]
        
        plt.figure(figsize=(12, 8))
        
        # Plot variance ratio × R² for each group
        for i, (group_name, group_data) in enumerate(group_items):
            # Calculate variance ratio for this group: group variance / target variance
            variance_ratios = [group_data["variance"][layer] / target_variance for layer in layer_names]
            variance_ratio_times_r2 = [var_ratio * r2 for var_ratio, r2 in zip(variance_ratios, r2_values)]
            plt.plot(range(len(layer_names)), variance_ratio_times_r2, 'o-', alpha=0.8, 
                    label=group_name, color=colors[i], linewidth=2, markersize=8)
        
        plt.title(f"{title} - Cluster Variance Ratio × Global R²", fontsize=18)
        plt.ylabel("Variance Ratio × R²", fontsize=16)
        plt.xticks(range(len(layer_names)), x_axis_labels, rotation=45, fontsize=14)
        plt.yticks(fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(loc='best', fontsize=12)
        plt.tight_layout()
        plt.savefig(Path(results_dir) / "cluster_variance_ratio_times_R2.pdf", dpi=300, bbox_inches="tight")
        plt.close()
        
        # Plot variance ratio for each cluster/group (separate plot)
        plt.figure(figsize=(12, 8))
        
        for i, (group_name, group_data) in enumerate(group_items):
            # Calculate variance ratio for this group: group variance / target variance
            variance_ratios = [group_data["variance"][layer] / target_variance for layer in layer_names]
            plt.plot(range(len(layer_names)), variance_ratios, 'o-', alpha=0.8, 
                    label=group_name, color=colors[i], linewidth=2, markersize=8)
        
        plt.title(f"{title} - Cluster Variance Ratio", fontsize=18)
        plt.ylabel("Variance Ratio", fontsize=16)
        plt.xticks(range(len(layer_names)), x_axis_labels, rotation=45, fontsize=14)
        plt.yticks(fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(loc='best', fontsize=12)
        plt.tight_layout()
        plt.savefig(Path(results_dir) / "cluster_variance_ratio.pdf", dpi=300, bbox_inches="tight")
        plt.close()
        
        # Plot R² for each group across layers (separate plot)
        plt.figure(figsize=(12, 8))
        
        # Plot R² for each group
        for i, (group_name, group_data) in enumerate(group_items):
            # Check if this group has R² scores
            if "r2_scores" in group_data:
                group_r2_values = [group_data["r2_scores"][layer] for layer in layer_names]
                plt.plot(range(len(layer_names)), group_r2_values, 'o-', alpha=0.8, 
                        label=group_name, color=colors[i], linewidth=2, markersize=8)
        
        plt.title(f"{title} - Cluster R²", fontsize=18)
        plt.ylabel("R² Score", fontsize=16)
        plt.xticks(range(len(layer_names)), x_axis_labels, rotation=45, fontsize=14)
        plt.yticks(fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(loc='best', fontsize=12)
        plt.tight_layout()
        plt.savefig(Path(results_dir) / "cluster_R2_across_layers.pdf", dpi=300, bbox_inches="tight")
        plt.close()
        
        # Plot global R² across layers (separate plot)
        plt.figure(figsize=(12, 8))
        
        # Use a color from the turbo colormap for consistency
        r2_color = plt.cm.turbo(0.7)  # Use a distinctive color from turbo palette
        plt.plot(range(len(layer_names)), r2_values, 'o-', alpha=0.8, 
                color=r2_color, linewidth=2, markersize=8, label='R²')
        
        plt.title(f"{title} - Global R²", fontsize=18)
        plt.ylabel("R² Score", fontsize=16)
        plt.xticks(range(len(layer_names)), x_axis_labels, rotation=45, fontsize=14)
        plt.yticks(fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(loc='best', fontsize=12)
        plt.tight_layout()
        plt.savefig(Path(results_dir) / "global_R2_across_layers.pdf", dpi=300, bbox_inches="tight")
        plt.close()


def plot_classification_layer_metrics(
    results: dict,
    results_dir: str = "results/classification_lens",
    title: str = "Classification lens",
    x_axis_labels: list | None = None,
    metrics_to_plot: list[str] = ("macro_f1", "micro_f1", "auroc"),
):
    """
    Plot per-layer multi-label metrics (global, across all groups).

    Expects:
        results["layer_metrics"][layer_name] = {
            "auroc", "micro_f1", "macro_f1", "samples_f1",
            "micro_accuracy", "hamming_loss", "jaccard_samples"
        }
    """
    os.makedirs(results_dir, exist_ok=True)

    layer_metrics = results["layer_metrics"]
    layer_names = list(layer_metrics.keys())
    if x_axis_labels is None:
        x_axis_labels = layer_names

    # Prepare data
    xs = range(len(layer_names))
    colors = plt.cm.turbo(np.linspace(0, 1, len(metrics_to_plot)))

    plt.figure(figsize=(12, 8))
    for i, metric in enumerate(metrics_to_plot):
        values = [layer_metrics[layer].get(metric, np.nan) for layer in layer_names]
        plt.plot(xs, values, "o-", label=metric, color=colors[i], linewidth=2, markersize=8)

    plt.title(title, fontsize=18)
    plt.xlabel("Layer", fontsize=16)
    plt.xticks(xs, x_axis_labels, rotation=45, fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best", fontsize=12)
    plt.tight_layout()
    plt.savefig(Path(results_dir) / "layer_metrics.pdf", dpi=300, bbox_inches="tight")
    plt.close()


def plot_group_molecules_classification_lens(
    results: dict,
    group_smiles: Dict[str, List[str]],
    targets: np.ndarray,
    results_dir: str = "results/classification_lens",
    threshold: float = 0.35,
    x_axis_labels: Optional[List[str]] = None,
    mean_y_label: str = "Mean correct-class probability",
    var_y_label: str = "Variance of correct-class probability",
    title: str = "Classification lens (group summary)",
):
    """
    Classification analogue of plot_group_molecules_regression_lens.

    For each group and each layer, we compute the mean and variance of the
    *correct-class probability* across all molecules and labels in that group.

    Correct-class probability is defined as:
        p   if y = 1
        1-p if y = 0

    Args
    ----
    results:
        Output of compare_molecule_groups_classification_lens. Expected structure:
            results[group][smile][layer] = {"logits", "probs", "binary"}
            results["layer_metrics"][layer] = {...}
    group_smiles:
        Same dict[group_name] -> list[SMILES] that was passed into
        compare_molecule_groups_classification_lens. The order of the groups
        and smiles inside each group must match how `targets` was constructed.
    targets:
        2D array of shape (N, num_labels) containing the multi-label targets,
        in the same order as concatenating the group_smiles values.
    results_dir:
        Directory to save the plots.
    threshold:
        Classification threshold (stored only for reference; the plots
        themselves use probabilities).
    x_axis_labels:
        Optional custom labels for the x-axis (layers). If None, layer names
        from the lens results are used directly.
    mean_y_label / var_y_label:
        Axis labels for the mean / variance plots.
    title:
        Base title for the plots.
    """
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Filter out non-group entries (e.g. "layer_metrics")
    group_items = [(g, results[g]) for g in group_smiles.keys() if g in results]
    if not group_items:
        raise ValueError("No group data found in results for the provided group_smiles.")

    # Use the first group to infer layer names
    first_group_name, first_group = group_items[0]
    first_smile = next(iter(first_group.keys()))
    layer_names = list(first_group[first_smile].keys())
    n_layers = len(layer_names)

    if x_axis_labels is None:
        x_axis_labels = layer_names

    # Sanity: targets length must match total smiles
    total_smiles = sum(len(smiles) for _, smiles in group_smiles.items())
    if targets.shape[0] != total_smiles:
        raise ValueError(
            f"targets.shape[0]={targets.shape[0]} != total number of smiles={total_smiles}. "
            "Make sure you built `targets` by concatenating group_smiles in the same order."
        )

    # Build per-group start/end indices into targets
    group_target_indices: Dict[str, tuple[int, int]] = {}
    current_idx = 0
    for group_name, smiles in group_smiles.items():
        n = len(smiles)
        group_target_indices[group_name] = (current_idx, current_idx + n)
        current_idx += n

    # Prepare colours
    colors = plt.cm.turbo(np.linspace(0, 1, len(group_items)))

    # --- 1) Group means of correct-class probability ---
    plt.figure(figsize=(12, 8))
    for i, (group_name, group_data) in enumerate(group_items):
        smiles = group_smiles[group_name]
        start, end = group_target_indices[group_name]
        y_true_group = targets[start:end].astype(float)  # (n_g, num_labels)

        mean_correct_by_layer: List[float] = []

        for layer in layer_names:
            # Stack probabilities in the same order as smiles list
            probs = np.stack(
                [group_data[smile][layer]["probs"] for smile in smiles],
                axis=0,
            )  # (n_g, num_labels)

            # Correct-class probability: p if y=1, (1-p) if y=0
            correct_probs = probs * y_true_group + (1.0 - probs) * (1.0 - y_true_group)
            mean_correct_by_layer.append(float(np.mean(correct_probs)))

        plt.plot(
            range(n_layers),
            mean_correct_by_layer,
            "o-",
            alpha=0.85,
            label=group_name,
            color=colors[i],
        )

    plt.title(title, fontsize=18)
    plt.ylabel(mean_y_label, fontsize=16)
    plt.xlabel("Layer", fontsize=16)
    plt.xticks(range(n_layers), x_axis_labels, rotation=45, fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best", fontsize=12)
    plt.tight_layout()
    plt.savefig(results_dir / "group_mean_correct_prob.pdf", dpi=300, bbox_inches="tight")
    plt.close()

    # --- 2) Group variances of correct-class probability ---
    plt.figure(figsize=(12, 8))
    for i, (group_name, group_data) in enumerate(group_items):
        smiles = group_smiles[group_name]
        start, end = group_target_indices[group_name]
        y_true_group = targets[start:end].astype(float)

        var_correct_by_layer: List[float] = []

        for layer in layer_names:
            probs = np.stack(
                [group_data[smile][layer]["probs"] for smile in smiles],
                axis=0,
            )
            correct_probs = probs * y_true_group + (1.0 - probs) * (1.0 - y_true_group)
            var_correct_by_layer.append(float(np.var(correct_probs)))

        plt.plot(
            range(n_layers),
            var_correct_by_layer,
            "o-",
            alpha=0.85,
            label=group_name,
            color=colors[i],
        )

    plt.title(title, fontsize=18)
    plt.ylabel(var_y_label, fontsize=16)
    plt.xlabel("Layer", fontsize=16)
    plt.xticks(range(n_layers), x_axis_labels, rotation=45, fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best", fontsize=12)
    plt.tight_layout()
    plt.savefig(results_dir / "group_var_correct_prob.pdf", dpi=300, bbox_inches="tight")
    plt.close()

    # --- 3) Global layer-wise metrics from results["layer_metrics"] (if present) ---
    if "layer_metrics" in results:
        layer_metrics = results["layer_metrics"]
        xs = range(n_layers)

        plt.figure(figsize=(12, 8))
        # Pick some common metrics if they exist
        preferred_metrics = ["macro_f1", "micro_f1", "auroc"]
        # Check which are actually present
        example_layer = next(iter(layer_metrics.values()))
        metric_names = [m for m in preferred_metrics if m in example_layer]
        metric_colors = plt.cm.turbo(np.linspace(0, 1, len(metric_names)))

        for c, metric in zip(metric_colors, metric_names):
            values = [layer_metrics[layer].get(metric, np.nan) for layer in layer_names]
            plt.plot(xs, values, "o-", label=metric, color=c, linewidth=2, markersize=8)

        plt.title(f"{title} – global layer metrics", fontsize=18)
        plt.xlabel("Layer", fontsize=16)
        plt.xticks(xs, x_axis_labels, rotation=45, fontsize=14)
        plt.yticks(fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(loc="best", fontsize=12)
        plt.tight_layout()
        plt.savefig(results_dir / "layer_metrics.pdf", dpi=300, bbox_inches="tight")
        plt.close()

    # --- 4) Per-group (cluster) F1 per layer from results["cluster_layer_f1"] (if present) ---
    if "cluster_layer_f1" in results:
        cl_f1 = results["cluster_layer_f1"]  # group -> layer -> {micro_f1, macro_f1, ...}

        # Ensure consistent group order
        f1_groups = [g for g, _ in group_items if g in cl_f1]
        if not f1_groups:
            raise ValueError("results['cluster_layer_f1'] present but no matching groups found.")

        metric = "macro_f1_present"  # change to "micro_f1" if you prefer

        # Heatmap: groups x layers
        Z = np.array([[cl_f1[g][layer].get(metric, np.nan) for layer in layer_names] for g in f1_groups], dtype=float)

        plt.figure(figsize=(max(8, 0.35 * len(layer_names)), max(4, 0.35 * len(f1_groups))))
        im = plt.imshow(Z, aspect="auto")
        plt.colorbar(im, label=metric)
        plt.yticks(np.arange(len(f1_groups)), f1_groups)
        plt.xticks(np.arange(len(layer_names)), x_axis_labels, rotation=90)
        plt.xlabel("Layer")
        plt.ylabel("Cluster / group")
        plt.title(f"{title} – {metric} per group by layer")
        plt.tight_layout()
        plt.savefig(results_dir / f"group_{metric}_heatmap.pdf", dpi=300, bbox_inches="tight")
        plt.close()

        # Line plot: one curve per group
        plt.figure(figsize=(max(8, 0.35 * len(layer_names)), 5))
        xs = np.arange(len(layer_names))
        for g in f1_groups:
            ys = [cl_f1[g][layer].get(metric, np.nan) for layer in layer_names]
            plt.plot(xs, ys, "o-", alpha=0.85, label=g)

        plt.xticks(xs, x_axis_labels, rotation=90)
        plt.xlabel("Layer")
        plt.ylabel(metric)
        plt.title(f"{title} – {metric} across layers (per group)")
        plt.legend(ncol=2, fontsize=8)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(results_dir / f"group_{metric}_lines.pdf", dpi=300, bbox_inches="tight")
        plt.close()


def plot_cluster_layer_f1_lines(results, metric="macro_f1", layer_names=None, title=None, max_clusters=None, results_dir=None):
    cl = results["cluster_layer_f1"]
    clusters = sorted(cl.keys())

    if max_clusters is not None:
        clusters = clusters[:max_clusters]

    if layer_names is None:
        layer_names = list(next(iter(cl.values())).keys())

    x = np.arange(len(layer_names))

    plt.figure(figsize=(max(8, 0.35 * len(layer_names)), 5))
    for c in clusters:
        y = [cl[c][layer][metric] for layer in layer_names]
        plt.plot(x, y, label=c)

    plt.xticks(x, layer_names, rotation=90)
    plt.xlabel("Layer")
    plt.ylabel("Macro F1")
    plt.title(title or f"Per-cluster Macro F1 across layers (Frozen Model)")
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(results_dir / f"group_{metric}_lines.pdf", dpi=300, bbox_inches="tight")
    plt.show()

def compute_confidence_per_label_per_layer(
    lens_results: dict,
    smiles_list: list,
):
    """
    Mean predicted probability per label per layer.

    Returns:
        DataFrame: index = layer names, columns = label_idx
    """
    first_smile = smiles_list[0]
    layer_names = list(lens_results[first_smile].keys())

    conf_rows = {}

    for layer in layer_names:
        probs = np.stack(
            [lens_results[smile][layer]["probs"] for smile in smiles_list],
            axis=0,  # (N, num_labels)
        )
        conf_rows[layer] = probs.mean(axis=0)  # (num_labels,)

    df = pd.DataFrame.from_dict(conf_rows, orient="index")
    df.index.name = "layer"
    df.columns.name = "label_idx"
    return df

def compute_confidence_positive_only(
    lens_results,
    smiles_list,
    targets,
):
    first_smile = smiles_list[0]
    layer_names = list(lens_results[first_smile].keys())

    conf_rows = {}

    for layer in layer_names:
        probs = np.stack(
            [lens_results[s][layer]["probs"] for s in smiles_list],
            axis=0,
        )  # (N, L)

        conf = []
        for j in range(probs.shape[1]):
            mask = targets[:, j] == 1
            if mask.any():
                conf.append(probs[mask, j].mean())
            else:
                conf.append(np.nan)

        conf_rows[layer] = conf

    return pd.DataFrame.from_dict(conf_rows, orient="index")

def compute_correct_class_confidence(
    lens_results,
    smiles_list,
    targets,
):
    first_smile = smiles_list[0]
    layer_names = list(lens_results[first_smile].keys())

    conf_rows = {}

    for layer in layer_names:
        probs = np.stack(
            [lens_results[s][layer]["probs"] for s in smiles_list],
            axis=0,
        )

        correct_probs = probs * targets + (1 - probs) * (1 - targets)
        conf_rows[layer] = correct_probs.mean(axis=0)

    return pd.DataFrame.from_dict(conf_rows, orient="index")

def compute_f1_per_label_per_layer(
    lens_results: dict,
    smiles_list: list,
    targets: np.ndarray,
    threshold: float = 0.35,
):
    """
    Compute per-label F1 for each layer.

    Args:
        lens_results: output of run_classification_lens
                      lens_results[smile][layer]["probs"] -> (num_labels,)
        smiles_list: list of SMILES in SAME order as targets
        targets: (N, num_labels) binary ground truth
        threshold: probability threshold

    Returns:
        DataFrame: index = layer names, columns = label_idx, values = F1
    """
    # infer layer names from first molecule
    first_smile = smiles_list[0]
    layer_names = list(lens_results[first_smile].keys())

    y_true = targets.astype(int)
    f1_rows = {}

    for layer in layer_names:
        # stack predictions across molecules
        probs = np.stack(
            [lens_results[smile][layer]["probs"] for smile in smiles_list],
            axis=0,  # (N, num_labels)
        )

        y_pred = (probs >= threshold).astype(int)

        # per-label F1
        f1s = [
            f1_score(y_true[:, j], y_pred[:, j], zero_division=0)
            for j in range(y_true.shape[1])
        ]

        f1_rows[layer] = f1s

    df = pd.DataFrame.from_dict(f1_rows, orient="index")
    df.index.name = "layer"
    df.columns.name = "label_idx"
    return df

def compute_auroc_per_label_per_layer(
    lens_results: dict,
    smiles_list: list,
    targets: np.ndarray,
):
    first_smile = smiles_list[0]
    layer_names = list(lens_results[first_smile].keys())

    y_true = targets.astype(int)
    auroc_rows = {}

    for layer in layer_names:
        probs = np.stack(
            [lens_results[smile][layer]["probs"] for smile in smiles_list],
            axis=0,  # (N, num_labels)
        )

        aurocs = []
        for j in range(y_true.shape[1]):
            # AUROC undefined if label never appears
            if y_true[:, j].sum() == 0 or y_true[:, j].sum() == len(y_true):
                aurocs.append(np.nan)
            else:
                aurocs.append(
                    roc_auc_score(y_true[:, j], probs[:, j])
                )

        auroc_rows[layer] = aurocs

    df = pd.DataFrame.from_dict(auroc_rows, orient="index")
    df.index.name = "layer"
    df.columns.name = "label_idx"
    return df

def _ordered_layers(layer_index):
    # Put Embedding first, then numeric layers in increasing order
    numeric = sorted([x for x in layer_index if str(x).isdigit()], key=lambda x: int(x))
    out = []
    if "Embedding" in layer_index:
        out.append("Embedding")
    out.extend(numeric)
    return out

def plot_top_bottom_labels_f1(df_f1_layer_label, label_names, k=5, title=None, save_path=None, LABEL_COLORS=None):
    layers = _ordered_layers(df_f1_layer_label.index)
    df = df_f1_layer_label.loc[layers]  # reorder

    final_layer = layers[-1]  # last numeric layer
    final_f1 = df.loc[final_layer]

    top = final_f1.sort_values(ascending=False).head(k).index.tolist()
    bottom = final_f1.sort_values(ascending=True).head(k).index.tolist()
    selected = top + bottom

    plt.figure(figsize=(10, 5))
    for j in top:
        plt.plot(df.index, df[j].values, marker="o", label=f"TOP: {label_names[j]}", color=LABEL_COLORS[label_names[j]],)
    for j in bottom:
        plt.plot(df.index, df[j].values, marker="o", linestyle="--", label=f"BOT: {label_names[j]}", color=LABEL_COLORS[label_names[j]],)

    plt.xlabel("Layer")
    plt.ylabel("F1")
    if title:
        plt.title(title)

    plt.ylim(0.0, 1.0)
    plt.xticks(rotation=45)
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200)
    plt.show()

    return [label_names[j] for j in top], [label_names[j] for j in bottom]

def plot_top_bottom_labels_confidence(
    df_conf_layer_label,
    top_labels_idx,
    bottom_labels_idx,
    label_names,
    title=None,
    save_path=None,
    LABEL_COLORS=None,
):
    plt.figure(figsize=(10, 5))

    for j in top_labels_idx:
        plt.plot(
            df_conf_layer_label.index,
            df_conf_layer_label[j].values,
            marker="o",
            label=f"TOP: {label_names[j]}",
            color=LABEL_COLORS[label_names[j]],
        )

    for j in bottom_labels_idx:
        plt.plot(
            df_conf_layer_label.index,
            df_conf_layer_label[j].values,
            marker="o",
            linestyle="--",
            label=f"BOT: {label_names[j]}",
            color=LABEL_COLORS[label_names[j]],
        )

    plt.xlabel("Layer")
    plt.ylabel("Mean predicted probability")
    if title:
        plt.title(title)

    plt.xticks(rotation=45)
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()



    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200)
    plt.show()
