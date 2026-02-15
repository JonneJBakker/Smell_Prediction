"""Ablation analysis module for ChemBERTa interpretability.

This module provides systematic ablation studies to understand the importance
of different model components (MLP neurons, attention heads) by measuring 
performance degradation when they are removed or zeroed out.

from utils.tl_ablation import run_ablation_analysis_with_metrics

# Assuming you have true target values for your test molecules
results = run_ablation_analysis_with_metrics(
    tl_model=tl_encoder,
    regressor=tl_regressor,
    test_molecules=test_smiles,
    true_targets=test_targets,  
    tokenizer=tokenizer,
    ablation_percentages=[0.0, 0.2, 0.5, 0.8],
    n_seeds=3,
    output_dir=Path("results"),
)


# Results now include metric degradation and combined ablation:
print(f"50% MLP ablation MAE: {results['mlp_ablation'][0.5]['mean_mae_denorm']:.4f}")
print(f"50% attention ablation R²: {results['attention_ablation'][0.5]['mean_r2_denorm']:.4f}")
print(f"50% combined ablation MAE: {results['combined_ablation'][0.5]['mean_mae_denorm']:.4f}")
```

Key functions:
- ablate_neurons_by_percentage(): Remove random percentages of MLP neurons 
- ablate_attention_heads_by_percentage(): Remove random percentages of attention heads
- ablate_both_components_by_percentage(): Remove percentages of both MLP neurons AND attention heads
- run_ablation_analysis_with_metrics(): Comprehensive ablation study with metrics
- plot_ablation_metrics(): Visualize metric degradation vs ablation percentage
"""

import random
import copy
from pathlib import Path
from typing import Dict, List, Optional
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import transformer_lens as tl
from transformers import RobertaTokenizerFast

from .tl_conversion import FaithfulTLRegressor
from .tl_validation import run_evaluation_metrics

def ablate_neurons_by_percentage(
    tl_model: tl.HookedEncoder,
    ablation_percentage: float,
    layers_to_ablate: Optional[List[int]] = None,
    seed: int = 42
) -> tl.HookedEncoder:
    """Create a copy of the model with a percentage of MLP neurons ablated.
    
    Args:
        tl_model: The TransformerLens model to ablate
        ablation_percentage: Percentage of neurons to ablate (0.0 to 1.0)
        layers_to_ablate: Specific layers to ablate, if None ablates all layers
        seed: Random seed for reproducible ablation
        
    Returns:
        New model with specified neurons ablated
    """
    # Deep copy preserves method overrides and exact behavior
    ablated_model = copy.deepcopy(tl_model).eval()
    
    if layers_to_ablate is None:
        layers_to_ablate = list(range(tl_model.cfg.n_layers))
    
    random.seed(seed)
    
    for layer_idx in layers_to_ablate:
        # Get MLP block
        mlp_block = ablated_model.blocks[layer_idx].mlp
        
        # Calculate number of neurons to ablate
        d_mlp = mlp_block.W_in.shape[1]  # Number of neurons in this layer
        n_ablate = int(d_mlp * ablation_percentage)
        
        if n_ablate > 0:
            # Randomly select neurons to ablate
            neurons_to_ablate = random.sample(range(d_mlp), n_ablate)
            
            # Zero out weights and biases for selected neurons
            # This guarantees the neurons_to_ablate units have activations of 0.
            # We *should not* ablate b_out. These are not tied to any specific hidden neuron.
            with torch.no_grad():
                mlp_block.W_in[:, neurons_to_ablate] = 0.0
                mlp_block.b_in[neurons_to_ablate] = 0.0  # Per-neuron bias
                mlp_block.W_out[neurons_to_ablate, :] = 0.0
 
    return ablated_model


def ablate_attention_heads_by_percentage(
    tl_model: tl.HookedEncoder,
    ablation_percentage: float,
    layers_to_ablate: Optional[List[int]] = None,
    seed: int = 42
) -> tl.HookedEncoder:
    """Create a copy of the model with a percentage of attention heads ablated.
    
    Args:
        tl_model: The TransformerLens model to ablate
        ablation_percentage: Percentage of attention heads to ablate (0.0 to 1.0)
        layers_to_ablate: Specific layers to ablate, if None ablates all layers
        seed: Random seed for reproducible ablation
        
    Returns:
        New model with specified attention heads ablated
    """
    # Deep copy preserves method overrides and exact behavior
    ablated_model = copy.deepcopy(tl_model).eval()
    
    if layers_to_ablate is None:
        layers_to_ablate = list(range(tl_model.cfg.n_layers))
    
    random.seed(seed)
    
    for layer_idx in layers_to_ablate:
        # Get attention block
        attn_block = ablated_model.blocks[layer_idx].attn
        n_heads = tl_model.cfg.n_heads
        
        # Calculate number of heads to ablate
        n_ablate = int(n_heads * ablation_percentage)
        
        if n_ablate > 0:
            # Randomly select heads to ablate
            heads_to_ablate = random.sample(range(n_heads), n_ablate)
            
            # Zero out weights for selected heads
            # Similarly, we shouldn't zero the output bias b_O
            with torch.no_grad():
                attn_block.W_Q[heads_to_ablate, :, :] = 0.0
                attn_block.W_K[heads_to_ablate, :, :] = 0.0  
                attn_block.W_V[heads_to_ablate, :, :] = 0.0
                attn_block.W_O[heads_to_ablate, :, :] = 0.0
                attn_block.b_Q[heads_to_ablate, :] = 0.0
                attn_block.b_K[heads_to_ablate, :] = 0.0
                attn_block.b_V[heads_to_ablate, :] = 0.0
    
    return ablated_model

def run_ablation_analysis_with_metrics(
        tl_model: tl.HookedEncoder,
        tl_regressor: FaithfulTLRegressor,
        tokenizer: RobertaTokenizerFast,
        test_data: pd.DataFrame,
        smiles_column: str = "smiles",
        target_column: str = "measured log solubility in mols per litre",
        ablation_percentages: List[float] = None,
        n_seeds: int = 5,
        device: Optional[str] = None,
        output_dir: Optional[Path] = Path("results"),
        display_denormalized: bool = True,
        scaler = None,
        batch_size: int = 128,
        mode = 'classification',
) -> Dict:
    """Run comprehensive ablation analysis using proper evaluation metrics.
    
    Args:
        tl_model: TransformerLens model to analyze
        tl_regressor: Original regressor for comparison
        tokenizer: Tokenizer for the model
        test_data: Test dataset DataFrame
        smiles_column: Name of SMILES column
        target_column: Name of target column
        ablation_percentages: List of ablation percentages to test
        n_seeds: Number of random seeds to average over
        device: Device to use for computation
        output_dir: Directory to save results
        display_denormalized: If True, show results in original scale
        scaler: Scaler for normalization
        batch_size: Batch size for evaluation (default: 256, increase for speed)
        
    Returns:
        Dictionary with comprehensive ablation results including metrics
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    
    # Enable memory-efficient settings
    if device == "cuda":
        torch.cuda.empty_cache()
    
    if ablation_percentages is None:
        ablation_percentages = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    test_molecules = test_data[smiles_column].to_list()
    targets = test_data[target_column].values

    print("Running ablation analysis with evaluation metrics...")
    print(f"Testing {len(ablation_percentages)} ablation percentages with {n_seeds} seeds each")
    print(f"Test molecules: {len(test_molecules)}")

    # Get baseline performance (no ablation) to ensure we still have the baseline
    # if user does not input 0.0 as ablation_percentage
    baseline_results = run_evaluation_metrics(model=tl_regressor,
                                              test_data=test_data,
                                              tokenizer=tokenizer,
                                              smiles_column=smiles_column,
                                              target_column=target_column,
                                              use_tl_model=True,
                                              scaler=scaler,
                                              batch_size=batch_size)

    results = {
        "baseline": baseline_results,
        "mlp_ablation": {},
        "attention_ablation": {},
        "combined_ablation": {},
        "config": {
            "ablation_percentages": ablation_percentages,
            "n_seeds": n_seeds,
            "n_test_molecules": len(test_molecules),
            "display_denormalized": display_denormalized,
            "target_column": target_column,
        }
    }

    # Test MLP ablation
    print("\nTesting MLP ablation...")
    for pct in ablation_percentages:
        print(f"  Ablating {pct:.1%} of MLP neurons...")
        pct_results = []
        
        for seed in range(n_seeds):
            ablated_model = ablate_neurons_by_percentage(
                tl_model, pct, seed=seed
            )
            # Wrap the ablated encoder with the original TL regressor head so outputs are 1D predictions
            ablated_regressor = FaithfulTLRegressor(
                faithful_tl_model=ablated_model,
                mlp_head=tl_regressor.mlp_head,
                dropout_p=tl_regressor.dropout.p,
                scaler=scaler if scaler is not None else getattr(tl_regressor, "scaler", None),
                target_column=target_column,
            ).eval()

            eval_results = run_evaluation_metrics(
                ablated_regressor,
                test_data,
                tokenizer,
                smiles_column,
                target_column,
                use_tl_model=True,
                scaler=scaler,
                batch_size=batch_size,
            )
            pct_results.append(eval_results)

        # Aggregate metrics across seeds
        if mode == "regression":
            metrics_to_avg = ["mae", "mse", "rmse", "r2", "mae_norm", "mse_norm", "rmse_norm", "r2_norm"]
            aggregated = {}

            for metric in metrics_to_avg:
                values = [res[metric] for res in pct_results]
                aggregated[f"mean_{metric}"] = np.mean(values)
                aggregated[f"std_{metric}"] = np.std(values, ddof=1)

            # Primary method is to look at non-normalized outputs
            aggregated.update({
                "mae_degradation": aggregated[f"mean_mae"] - baseline_results["mae"],
                "mse_degradation": aggregated[f"mean_mse"] - baseline_results["mse"],
                "rmse_degradation": aggregated[f"mean_rmse"] - baseline_results["rmse"],
                "r2_degradation": baseline_results["r2"] - aggregated[f"mean_r2"],  # R² decrease is degradation
                "seeds_results": pct_results
            })

            results["mlp_ablation"][pct] = aggregated

        if mode == "classification":
            metrics_to_avg = ["micro_accuracy", "auroc", "micro_f1", "macro_f1", "samples_f1", "hamming_loss", "jaccard_samples"]

            aggregated = {}

            for metric in metrics_to_avg:
                values = [res[metric] for res in pct_results]
                aggregated[f"mean_{metric}"] = np.mean(values)
                aggregated[f"std_{metric}"] = np.std(values, ddof=1)

                # Primary method is to look at non-normalized outputs
            aggregated.update({
                "micro_f1_degradation": aggregated[f"mean_micro_f1"] - baseline_results["micro_f1"],
                "macro_f1_degradation": aggregated[f"mean_macro_f1"] - baseline_results["macro_f1"],
                "auroc_degradation": aggregated[f"mean_auroc"] - baseline_results["auroc"],
                "samples_f1_degradation": aggregated[f"mean_samples_f1"] - baseline_results["samples_f1"],
                "micro_accuracy_degradation": aggregated[f"mean_micro_accuracy"] - baseline_results["micro_accuracy"],
                "jaccard_samples_degradation": aggregated[f"mean_jaccard_samples"] - baseline_results["jaccard_samples"],
                "hamming_loss_degradation": aggregated[f"mean_hamming_loss"] - baseline_results["hamming_loss"],
                "seeds_results": pct_results
            })

            results["mlp_ablation"][pct] = aggregated
    # Test attention ablation
    print("\nTesting attention ablation...")
    for pct in ablation_percentages:
        print(f"  Ablating {pct:.1%} of attention heads...")
        pct_results = []
        
        for seed in range(n_seeds):
            ablated_model = ablate_attention_heads_by_percentage(
                tl_model, pct, seed=seed
            )
            # Wrap the ablated encoder with the original TL regressor head so outputs are 1D predictions
            ablated_regressor = FaithfulTLRegressor(
                faithful_tl_model=ablated_model,
                mlp_head=tl_regressor.mlp_head,
                dropout_p=tl_regressor.dropout.p,
                scaler=scaler if scaler is not None else getattr(tl_regressor, "scaler", None),
                target_column=target_column,
            ).eval()

            eval_results = run_evaluation_metrics(
                ablated_regressor,
                test_data,
                tokenizer,
                smiles_column,
                target_column,
                use_tl_model=True,
                scaler=scaler,
                batch_size=batch_size,
            )
            pct_results.append(eval_results)

        # Aggregate metrics across seeds
        if mode == "regression":
            metrics_to_avg = ["mae", "mse", "rmse", "r2", "mae_norm", "mse_norm", "rmse_norm", "r2_norm"]
            aggregated = {}

            for metric in metrics_to_avg:
                values = [res[metric] for res in pct_results]
                aggregated[f"mean_{metric}"] = np.mean(values)
                aggregated[f"std_{metric}"] = np.std(values, ddof=1)

            # Primary method is to look at non-normalized outputs
            aggregated.update({
                "mae_degradation": aggregated[f"mean_mae"] - baseline_results["mae"],
                "mse_degradation": aggregated[f"mean_mse"] - baseline_results["mse"],
                "rmse_degradation": aggregated[f"mean_rmse"] - baseline_results["rmse"],
                "r2_degradation": baseline_results["r2"] - aggregated[f"mean_r2"],  # R² decrease is degradation
                "seeds_results": pct_results
            })

            results["attention_ablation"][pct] = aggregated

        if mode == "classification":
            metrics_to_avg = ["micro_accuracy", "auroc", "micro_f1", "macro_f1", "samples_f1", "hamming_loss",
                              "jaccard_samples"]

            aggregated = {}

            for metric in metrics_to_avg:
                values = [res[metric] for res in pct_results]
                aggregated[f"mean_{metric}"] = np.mean(values)
                aggregated[f"std_{metric}"] = np.std(values, ddof=1)

                # Primary method is to look at non-normalized outputs
            aggregated.update({
                "micro_f1_degradation": aggregated[f"mean_micro_f1"] - baseline_results["micro_f1"],
                "macro_f1_degradation": aggregated[f"mean_macro_f1"] - baseline_results["macro_f1"],
                "auroc_degradation": aggregated[f"mean_auroc"] - baseline_results["auroc"],
                "samples_f1_degradation": aggregated[f"mean_samples_f1"] - baseline_results["samples_f1"],
                "micro_accuracy_degradation": aggregated[f"mean_micro_accuracy"] - baseline_results["micro_accuracy"],
                "jaccard_samples_degradation": aggregated[f"mean_jaccard_samples"] - baseline_results[
                    "jaccard_samples"],
                "hamming_loss_degradation": aggregated[f"mean_hamming_loss"] - baseline_results["hamming_loss"],
                "seeds_results": pct_results
            })

            results["attention_ablation"][pct] = aggregated


    # Test combined ablation
    print("\nTesting combined ablation...")
    for pct in ablation_percentages:
        print(f"  Ablating {pct:.1%} of attention heads and mlp neurons...")
        pct_results = []
        
        for seed in range(n_seeds):
            ablated_model = ablate_attention_heads_by_percentage(
                tl_model, pct, seed=seed
            )
            ablated_model = ablate_neurons_by_percentage(
                ablated_model, pct, seed=seed
            )
            # Wrap the ablated encoder with the original TL regressor head so outputs are 1D predictions
            ablated_regressor = FaithfulTLRegressor(
                faithful_tl_model=ablated_model,
                mlp_head=tl_regressor.mlp_head,
                dropout_p=tl_regressor.dropout.p,
                scaler=scaler if scaler is not None else getattr(tl_regressor, "scaler", None),
                target_column=target_column,
            ).eval()

            eval_results = run_evaluation_metrics(
                ablated_regressor,
                test_data,
                tokenizer,
                smiles_column,
                target_column,
                use_tl_model=True,
                scaler=scaler,
                batch_size=batch_size,
            )
            pct_results.append(eval_results)

        # Aggregate metrics across seeds
        if mode == "regression":
            metrics_to_avg = ["mae", "mse", "rmse", "r2", "mae_norm", "mse_norm", "rmse_norm", "r2_norm"]
            aggregated = {}

            for metric in metrics_to_avg:
                values = [res[metric] for res in pct_results]
                aggregated[f"mean_{metric}"] = np.mean(values)
                aggregated[f"std_{metric}"] = np.std(values, ddof=1)

            # Primary method is to look at non-normalized outputs
            aggregated.update({
                "mae_degradation": aggregated[f"mean_mae"] - baseline_results["mae"],
                "mse_degradation": aggregated[f"mean_mse"] - baseline_results["mse"],
                "rmse_degradation": aggregated[f"mean_rmse"] - baseline_results["rmse"],
                "r2_degradation": baseline_results["r2"] - aggregated[f"mean_r2"],  # R² decrease is degradation
                "seeds_results": pct_results
            })

            results["combined_ablation"][pct] = aggregated

        if mode == "classification":
            metrics_to_avg = ["micro_accuracy", "auroc", "micro_f1", "macro_f1", "samples_f1", "hamming_loss",
                              "jaccard_samples"]

            aggregated = {}

            for metric in metrics_to_avg:
                values = [res[metric] for res in pct_results]
                aggregated[f"mean_{metric}"] = np.mean(values)
                aggregated[f"std_{metric}"] = np.std(values, ddof=1)

                # Primary method is to look at non-normalized outputs
            aggregated.update({
                "micro_f1_degradation": aggregated[f"mean_micro_f1"] - baseline_results["micro_f1"],
                "macro_f1_degradation": aggregated[f"mean_macro_f1"] - baseline_results["macro_f1"],
                "auroc_degradation": aggregated[f"mean_auroc"] - baseline_results["auroc"],
                "samples_f1_degradation": aggregated[f"mean_samples_f1"] - baseline_results["samples_f1"],
                "micro_accuracy_degradation": aggregated[f"mean_micro_accuracy"] - baseline_results["micro_accuracy"],
                "jaccard_samples_degradation": aggregated[f"mean_jaccard_samples"] - baseline_results[
                    "jaccard_samples"],
                "hamming_loss_degradation": aggregated[f"mean_hamming_loss"] - baseline_results["hamming_loss"],
                "seeds_results": pct_results
            })

            results["combined_ablation"][pct] = aggregated

    print("Ablation analysis complete!")

    if output_dir:
        ablation_dir = output_dir / "ablation"
        ablation_dir.mkdir(exist_ok=True, parents=True)

        # Create summary DataFrame and save
        summary_data = []
        ablation_types = ["mlp_ablation", "attention_ablation", "combined_ablation"]
        type_labels = {"mlp_ablation": "MLP Ablation", "attention_ablation": "Attention Ablation", "combined_ablation": "Combined Ablation"}

        if mode == 'classification':
            for ablation_type in ablation_types:
                for pct, pct_data in results[ablation_type].items():
                    summary_data.append({
                        "ablation_type": type_labels[ablation_type],
                        "ablation_percentage": pct,
                        f"micro_f1": pct_data[f"mean_micro_f1"],
                        f"micro_f1_std": pct_data[f"std_micro_f1"],
                        f"macro_f1": pct_data[f"mean_macro_f1"],
                        f"macro_f1_std": pct_data[f"std_macro_f1"],
                        f"samples_f1": pct_data[f"mean_samples_f1"],
                        f"samples_f1_std": pct_data[f"std_samples_f1"],
                        f"jaccard_samples": pct_data[f"mean_jaccard_samples"],
                        f"jaccard_samples_std": pct_data[f"std_jaccard_samples"],
                        f"hamming_loss": pct_data[f"mean_hamming_loss"],
                        f"hamming_loss_std": pct_data[f"std_hamming_loss"],
                        f"micro_accuracy": pct_data[f"mean_micro_accuracy"],
                        f"micro_accuracy_std": pct_data[f"std_micro_accuracy"],
                        f"auroc": pct_data[f"mean_auroc"],
                        f"auroc_std": pct_data[f"std_auroc"],
                        "micro_f1_degradation": pct_data["micro_f1_degradation"],
                        "macro_f1_degradation": pct_data["macro_f1_degradation"],
                        "samples_f1_degradation": pct_data["samples_f1_degradation"],
                        "jaccard_samples_degradation": pct_data["jaccard_samples_degradation"],
                        "hamming_loss_degradation": pct_data["hamming_loss_degradation"],
                        "micro_accuracy_degradation": pct_data["micro_accuracy_degradation"],
                        "auroc_degradation": pct_data["auroc_degradation"],
                    })
        elif mode == 'regression':
            for ablation_type in ablation_types:
                for pct, pct_data in results[ablation_type].items():
                    summary_data.append({
                        "ablation_type": type_labels[ablation_type],
                        "ablation_percentage": pct,
                        f"mae": pct_data[f"mean_mae"],
                        f"mae_std": pct_data[f"std_mae"],
                        f"mse": pct_data[f"mean_mse"],
                        f"mse_std": pct_data[f"std_mse"],
                        f"rmse": pct_data[f"mean_rmse"],
                        f"rmse_std": pct_data[f"std_rmse"],
                        f"r2": pct_data[f"mean_r2"],
                        f"r2_std": pct_data[f"std_r2"],
                        f"mae_norm": pct_data[f"mean_mae_norm"],
                        f"mae_norm_std": pct_data[f"std_mae_norm"],
                        f"mse_norm": pct_data[f"mean_mse_norm"],
                        f"mse_norm_std": pct_data[f"std_mse_norm"],
                        f"rmse_norm": pct_data[f"mean_rmse_norm"],
                        f"rmse_norm_std": pct_data[f"std_rmse_norm"],
                        f"r2_norm": pct_data[f"mean_r2_norm"],
                        f"r2_norm_std": pct_data[f"std_r2_norm"],
                        "mae_degradation": pct_data["mae_degradation"],
                        "mse_degradation": pct_data["mse_degradation"],
                        "rmse_degradation": pct_data["rmse_degradation"],
                        "r2_degradation": pct_data["r2_degradation"]
                    })
        
        df = pd.DataFrame(summary_data)
        df.to_csv(ablation_dir / "ablation_metrics_summary.csv", index=False)
    
        with open(ablation_dir / "all_results.pkl", "wb") as f:
            pickle.dump(results, f)
        plot_ablation_metrics(results, output_dir)
    
    return results

def plot_ablation_metrics(results: Dict, output_dir: Path, title: str = "ESOL", mode = "classification") -> None:
    """Create individual PDF visualization plots for ablation analysis using evaluation metrics.
    
    Args:
        results: Results from run_ablation_analysis_with_metrics()
        output_dir: Directory to save plots
    """

    # Create ablation subdirectory
    ablation_dir = output_dir / "ablation"
    ablation_dir.mkdir(exist_ok=True, parents=True)
    
    # Extract data for plotting - convert to percentages for clearer display
    ablation_pcts = [pct * 100 for pct in list(results["mlp_ablation"].keys())]  # Convert to %

    if mode == "regression":
        # Extract data for all ablation types
        mlp_mae = [results["mlp_ablation"][pct/100][f"mean_mae"] for pct in ablation_pcts]
        attention_mae = [results["attention_ablation"][pct/100][f"mean_mae"] for pct in ablation_pcts]
        combined_mae = [results["combined_ablation"][pct/100][f"mean_mae"] for pct in ablation_pcts]

        mlp_r2 = [results["mlp_ablation"][pct/100][f"mean_r2"] for pct in ablation_pcts]
        attention_r2 = [results["attention_ablation"][pct/100][f"mean_r2"] for pct in ablation_pcts]
        combined_r2 = [results["combined_ablation"][pct/100][f"mean_r2"] for pct in ablation_pcts]

        mlp_rmse = [results["mlp_ablation"][pct/100][f"mean_rmse"] for pct in ablation_pcts]
        attention_rmse = [results["attention_ablation"][pct/100][f"mean_rmse"] for pct in ablation_pcts]
        combined_rmse = [results["combined_ablation"][pct/100][f"mean_rmse"] for pct in ablation_pcts]

        # Plot 1: MAE values
        fig2, ax2 = plt.subplots(1, 1, figsize=(10, 8))
        ax2.plot(ablation_pcts, mlp_mae, 'o-', label='FFN Ablation', linewidth=3, markersize=8, color='#1f77b4')
        ax2.plot(ablation_pcts, attention_mae, 's-', label='Attention Head Ablation', linewidth=3, markersize=8, color='#ff7f0e')
        ax2.plot(ablation_pcts, combined_mae, '^-', label='Combined Ablation', linewidth=3, markersize=8, color='#d62728')

        ax2.set_title(title, fontsize=18)
        ax2.set_xlabel('Ablation Percentage (%)', fontsize=16)
        ax2.set_ylabel(f'Mean Absolute Error', fontsize=16)
        ax2.tick_params(axis='both', which='major', labelsize=14)
        ax2.legend(fontsize=16)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(ablation_dir / "mae_ablation_plot.pdf", dpi=300, bbox_inches="tight")
        plt.close()

        # Plot 2: R² values with confidence intervals
        fig3, ax3 = plt.subplots(1, 1, figsize=(10, 8))

        # Extract standard errors for confidence intervals (95% CI = ±1.96*SE)
        mlp_r2_std = [results["mlp_ablation"][pct/100][f"std_r2"] for pct in ablation_pcts]
        attention_r2_std = [results["attention_ablation"][pct/100][f"std_r2"] for pct in ablation_pcts]
        combined_r2_std = [results["combined_ablation"][pct/100][f"std_r2"] for pct in ablation_pcts]

        # Calculate 95% confidence intervals (assuming normal distribution)
        n_seeds = len(results["mlp_ablation"][list(results["mlp_ablation"].keys())[0]]["seeds_results"])
        print(n_seeds)
        ci_multiplier = 1.96 / np.sqrt(n_seeds)  # 95% CI for sample mean

        mlp_r2_ci = [std * ci_multiplier for std in mlp_r2_std]
        attention_r2_ci = [std * ci_multiplier for std in attention_r2_std]
        combined_r2_ci = [std * ci_multiplier for std in combined_r2_std]

        # Plot with shaded confidence intervals
        # MLP ablation
        ax3.plot(ablation_pcts, mlp_r2, 'o-', label='FFN Ablation',
                linewidth=3, markersize=8, color='#1f77b4')
        ax3.fill_between(ablation_pcts,
                         np.array(mlp_r2) - np.array(mlp_r2_ci),
                         np.array(mlp_r2) + np.array(mlp_r2_ci),
                         alpha=0.3, color='#1f77b4')

        # Attention ablation
        ax3.plot(ablation_pcts, attention_r2, 's-', label='Attention Head Ablation',
                linewidth=3, markersize=8, color='#ff7f0e')
        ax3.fill_between(ablation_pcts,
                         np.array(attention_r2) - np.array(attention_r2_ci),
                         np.array(attention_r2) + np.array(attention_r2_ci),
                         alpha=0.3, color='#ff7f0e')

        # Combined ablation
        ax3.plot(ablation_pcts, combined_r2, '^-', label='Combined Ablation',
                linewidth=3, markersize=8, color='#d62728')
        ax3.fill_between(ablation_pcts,
                         np.array(combined_r2) - np.array(combined_r2_ci),
                         np.array(combined_r2) + np.array(combined_r2_ci),
                         alpha=0.3, color='#d62728')

        ax3.set_title(title, fontsize=18)
        ax3.set_xlabel('Ablation Percentage (%)', fontsize=16)
        ax3.set_ylabel(f'R² Score', fontsize=16)
        ax3.tick_params(axis='both', which='major', labelsize=14)
        ax3.legend(fontsize=16)
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(ablation_dir / "r2_ablation_plot.pdf", dpi=300, bbox_inches="tight")
        plt.close()

        # Plot 3: RMSE values
        fig4, ax4 = plt.subplots(1, 1, figsize=(10, 8))
        ax4.plot(ablation_pcts, mlp_rmse, 'o-', label='FFN Ablation', linewidth=3, markersize=8, color='#1f77b4')
        ax4.plot(ablation_pcts, attention_rmse, 's-', label='Attention Head Ablation', linewidth=3, markersize=8, color='#ff7f0e')
        ax4.plot(ablation_pcts, combined_rmse, '^-', label='Combined Ablation', linewidth=3, markersize=8, color='#d62728')

        ax4.set_xlabel('Ablation Percentage (%)', fontsize=16)
        ax4.set_ylabel(f'Root Mean Squared Error', fontsize=16)
        ax4.tick_params(axis='both', which='major', labelsize=14)
        ax4.legend(fontsize=16)
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(ablation_dir / "rmse_ablation_plot.pdf", dpi=300, bbox_inches="tight")
        plt.close()

        # Plot 4: Comparison bar chart at key percentages
        fig5, ax5 = plt.subplots(1, 1, figsize=(10, 8))
        key_pcts = [20, 50, 80]  # Key percentages to compare
        mlp_key_mae = [results["mlp_ablation"][pct/100][f"mean_mae"]
                       for pct in key_pcts if pct/100 in results["mlp_ablation"]]
        attention_key_mae = [results["attention_ablation"][pct/100][f"mean_mae"]
                            for pct in key_pcts if pct/100 in results["attention_ablation"]]
        combined_key_mae = [results["combined_ablation"][pct/100][f"mean_mae"]
                            for pct in key_pcts if pct/100 in results["combined_ablation"]]

        x = np.arange(len(key_pcts))
        width = 0.25

        ax5.bar(x - width, mlp_key_mae, width, label='FFN Ablation', alpha=0.8, color='#1f77b4')
        ax5.bar(x, attention_key_mae, width, label='Attention Head Ablation', alpha=0.8, color='#ff7f0e')
        ax5.bar(x + width, combined_key_mae, width, label='Combined Ablation', alpha=0.8, color='#d62728')

        ax5.set_xlabel('Ablation Percentage (%)', fontsize=16)
        ax5.set_ylabel(f'Mean Absolute Error', fontsize=16)
        ax5.set_xticks(x)
        ax5.set_xticklabels([f'{pct}%' for pct in key_pcts], fontsize=14)
        ax5.tick_params(axis='y', which='major', labelsize=14)
        ax5.legend(fontsize=16)
        ax5.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(ablation_dir / "mae_comparison_bar_plot.pdf", dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Ablation metrics plots saved to {ablation_dir}:")
        print(f"  • mae_ablation_plot.pdf")
        print(f"  • r2_ablation_plot.pdf")
        print(f"  • rmse_ablation_plot.pdf")
        print(f"  • mae_comparison_bar_plot.pdf")

    elif mode == "classification":
        # -----------------------------
        # CLASSIFICATION PLOTS
        # -----------------------------
        # Expect keys like: mean_micro_f1, std_micro_f1, mean_auroc, std_auroc, mean_hamming_loss
        mlp_f1_micro = [results["mlp_ablation"][pct / 100]["mean_micro_f1"] for pct in ablation_pcts]
        attention_f1_micro = [results["attention_ablation"][pct / 100]["mean_micro_f1"] for pct in ablation_pcts]
        combined_f1_micro = [results["combined_ablation"][pct / 100]["mean_micro_f1"] for pct in ablation_pcts]

        mlp_f1_macro = [results["mlp_ablation"][pct / 100]["mean_macro_f1"] for pct in ablation_pcts]
        attention_f1_macro = [results["attention_ablation"][pct / 100]["mean_macro_f1"] for pct in ablation_pcts]
        combined_f1_macro = [results["combined_ablation"][pct / 100]["mean_macro_f1"] for pct in ablation_pcts]

        mlp_auroc = [results["mlp_ablation"][pct / 100]["mean_auroc"] for pct in ablation_pcts]
        attention_auroc = [results["attention_ablation"][pct / 100]["mean_auroc"] for pct in ablation_pcts]
        combined_auroc = [results["combined_ablation"][pct / 100]["mean_auroc"] for pct in ablation_pcts]

        mlp_hamming = [results["mlp_ablation"][pct / 100]["mean_hamming_loss"] for pct in ablation_pcts]
        attention_hamming = [results["attention_ablation"][pct / 100]["mean_hamming_loss"] for pct in ablation_pcts]
        combined_hamming = [results["combined_ablation"][pct / 100]["mean_hamming_loss"] for pct in ablation_pcts]

        #1) Macro-F1 with CI
        fig1, ax1 = plt.subplots(1, 1, figsize=(10, 8))

        mlp_macro_f1_std = [results["mlp_ablation"][pct / 100]["std_macro_f1"] for pct in ablation_pcts]
        attention_macro_f1_std = [results["attention_ablation"][pct / 100]["std_macro_f1"] for pct in ablation_pcts]
        combined_macro_f1_std = [results["combined_ablation"][pct / 100]["std_macro_f1"] for pct in ablation_pcts]

        n_seeds = len(results["mlp_ablation"][list(results["mlp_ablation"].keys())[0]]["seeds_results"])
        ci_multiplier = 1.96 / np.sqrt(n_seeds)

        mlp_macro_f1_ci = [std * ci_multiplier for std in mlp_macro_f1_std]
        attention_macro_f1_ci = [std * ci_multiplier for std in attention_macro_f1_std]
        combined_macro_f1_ci = [std * ci_multiplier for std in combined_macro_f1_std]

        ax1.plot(ablation_pcts, mlp_f1_macro, "o-", label="FFN Ablation", linewidth=3, markersize=8)
        ax1.fill_between(
            ablation_pcts,
            np.array(mlp_f1_macro) - np.array(mlp_macro_f1_ci),
            np.array(mlp_f1_macro) + np.array(mlp_macro_f1_ci),
            alpha=0.3,
        )

        ax1.plot(ablation_pcts, attention_f1_macro, "s-", label="Attention Head Ablation", linewidth=3, markersize=8)
        ax1.fill_between(
            ablation_pcts,
            np.array(attention_f1_macro) - np.array(attention_macro_f1_ci),
            np.array(attention_f1_macro) + np.array(attention_macro_f1_ci),
            alpha=0.3,
        )

        ax1.plot(ablation_pcts, combined_f1_macro, "^-", label="Combined Ablation", linewidth=3, markersize=8)
        ax1.fill_between(
            ablation_pcts,
            np.array(combined_f1_macro) - np.array(combined_macro_f1_ci),
            np.array(combined_f1_macro) + np.array(combined_macro_f1_ci),
            alpha=0.3,
        )
        k = 1.3
        ax1.set_title(title, fontsize=18*k)
        ax1.set_xlabel("Ablation Percentage (%)", fontsize=16*k)
        ax1.set_ylabel("Macro F1-score", fontsize=16*k)
        ax1.tick_params(axis="both", which="major", labelsize=14*k)
        ax1.legend(fontsize=16*k)
        ax1.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(ablation_dir / "f1_macro_ablation_plot.pdf", dpi=300, bbox_inches="tight")
        plt.close()

        # 2) Micro-F1 with CI
        fig2, ax2 = plt.subplots(1, 1, figsize=(10, 8))

        mlp_micro_f1_std = [results["mlp_ablation"][pct / 100]["std_micro_f1"] for pct in ablation_pcts]
        attention_micro_f1_std = [results["attention_ablation"][pct / 100]["std_micro_f1"] for pct in ablation_pcts]
        combined_micro_f1_std = [results["combined_ablation"][pct / 100]["std_micro_f1"] for pct in ablation_pcts]

        n_seeds = len(results["mlp_ablation"][list(results["mlp_ablation"].keys())[0]]["seeds_results"])
        ci_multiplier = 1.96 / np.sqrt(n_seeds)

        mlp_micro_f1_ci = [std * ci_multiplier for std in mlp_micro_f1_std]
        attention_micro_f1_ci = [std * ci_multiplier for std in attention_micro_f1_std]
        combined_micro_f1_ci = [std * ci_multiplier for std in combined_micro_f1_std]

        ax2.plot(ablation_pcts, mlp_f1_micro, "o-", label="FFN Ablation", linewidth=3, markersize=8)
        ax2.fill_between(
            ablation_pcts,
            np.array(mlp_f1_micro) - np.array(mlp_micro_f1_ci),
            np.array(mlp_f1_micro) + np.array(mlp_micro_f1_ci),
            alpha=0.3,
        )

        ax2.plot(ablation_pcts, attention_f1_micro, "s-", label="Attention Head Ablation", linewidth=3, markersize=8)
        ax2.fill_between(
            ablation_pcts,
            np.array(attention_f1_micro) - np.array(attention_micro_f1_ci),
            np.array(attention_f1_micro) + np.array(attention_micro_f1_ci),
            alpha=0.3,
        )

        ax2.plot(ablation_pcts, combined_f1_micro, "^-", label="Combined Ablation", linewidth=3, markersize=8)
        ax2.fill_between(
            ablation_pcts,
            np.array(combined_f1_micro) - np.array(combined_micro_f1_ci),
            np.array(combined_f1_micro) + np.array(combined_micro_f1_ci),
            alpha=0.3,
        )

        ax2.set_title(title, fontsize=18)
        ax2.set_xlabel("Ablation Percentage (%)", fontsize=16)
        ax2.set_ylabel("Micro F1-score", fontsize=16)
        ax2.tick_params(axis="both", which="major", labelsize=14)
        ax2.legend(fontsize=16)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(ablation_dir / "f1_ablation_plot.pdf", dpi=300, bbox_inches="tight")
        plt.close()

        # 2) AUROC with CI
        fig3, ax3 = plt.subplots(1, 1, figsize=(10, 8))

        mlp_auroc_std = [results["mlp_ablation"][pct / 100]["std_auroc"] for pct in ablation_pcts]
        attention_auroc_std = [results["attention_ablation"][pct / 100]["std_auroc"] for pct in ablation_pcts]
        combined_auroc_std = [results["combined_ablation"][pct / 100]["std_auroc"] for pct in ablation_pcts]

        mlp_auroc_ci = [std * ci_multiplier for std in mlp_auroc_std]
        attention_auroc_ci = [std * ci_multiplier for std in attention_auroc_std]
        combined_auroc_ci = [std * ci_multiplier for std in combined_auroc_std]

        ax3.plot(ablation_pcts, mlp_auroc, "o-", label="FFN Ablation", linewidth=3, markersize=8)
        ax3.fill_between(
            ablation_pcts,
            np.array(mlp_auroc) - np.array(mlp_auroc_ci),
            np.array(mlp_auroc) + np.array(mlp_auroc_ci),
            alpha=0.3,
        )

        ax3.plot(ablation_pcts, attention_auroc, "s-", label="Attention Head Ablation", linewidth=3, markersize=8)
        ax3.fill_between(
            ablation_pcts,
            np.array(attention_auroc) - np.array(attention_auroc_ci),
            np.array(attention_auroc) + np.array(attention_auroc_ci),
            alpha=0.3,
        )

        ax3.plot(ablation_pcts, combined_auroc, "^-", label="Combined Ablation", linewidth=3, markersize=8)
        ax3.fill_between(
            ablation_pcts,
            np.array(combined_auroc) - np.array(combined_auroc_ci),
            np.array(combined_auroc) + np.array(combined_auroc_ci),
            alpha=0.3,
        )

        ax3.set_title(title, fontsize=18)
        ax3.set_xlabel("Ablation Percentage (%)", fontsize=16)
        ax3.set_ylabel("AUROC", fontsize=16)
        ax3.tick_params(axis="both", which="major", labelsize=14)
        ax3.legend(fontsize=16)
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(ablation_dir / "auroc_ablation_plot.pdf", dpi=300, bbox_inches="tight")
        plt.close()

        # 3) Hamming loss (lower is better)
        fig4, ax4 = plt.subplots(1, 1, figsize=(10, 8))
        ax4.plot(ablation_pcts, mlp_hamming, "o-", label="FFN Ablation", linewidth=3, markersize=8)
        ax4.plot(ablation_pcts, attention_hamming, "s-", label="Attention Head Ablation", linewidth=3, markersize=8)
        ax4.plot(ablation_pcts, combined_hamming, "^-", label="Combined Ablation", linewidth=3, markersize=8)

        ax4.set_xlabel("Ablation Percentage (%)", fontsize=16)
        ax4.set_ylabel("Hamming Loss", fontsize=16)
        ax4.tick_params(axis="both", which="major", labelsize=14)
        ax4.legend(fontsize=16)
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(ablation_dir / "hamming_ablation_plot.pdf", dpi=300, bbox_inches="tight")
        plt.close()

        # 4) Bar plot for micro-F1 at key percentages
        fig5, ax5 = plt.subplots(1, 1, figsize=(10, 8))
        key_pcts = [20, 50, 80]
        mlp_key_f1 = [
            results["mlp_ablation"][pct / 100]["mean_micro_f1"] for pct in key_pcts if
            pct / 100 in results["mlp_ablation"]
        ]
        attention_key_f1 = [
            results["attention_ablation"][pct / 100]["mean_micro_f1"]
            for pct in key_pcts
            if pct / 100 in results["attention_ablation"]
        ]
        combined_key_f1 = [
            results["combined_ablation"][pct / 100]["mean_micro_f1"]
            for pct in key_pcts
            if pct / 100 in results["combined_ablation"]
        ]

        x = np.arange(len(key_pcts))
        width = 0.25

        ax5.bar(x - width, mlp_key_f1, width, label="FFN Ablation", alpha=0.8)
        ax5.bar(x, attention_key_f1, width, label="Attention Head Ablation", alpha=0.8)
        ax5.bar(x + width, combined_key_f1, width, label="Combined Ablation", alpha=0.8)

        ax5.set_xlabel("Ablation Percentage (%)", fontsize=16)
        ax5.set_ylabel("Micro F1-score", fontsize=16)
        ax5.set_xticks(x)
        ax5.set_xticklabels([f"{pct}%" for pct in key_pcts], fontsize=14)
        ax5.tick_params(axis="y", which="major", labelsize=14)
        ax5.legend(fontsize=16)
        ax5.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(ablation_dir / "f1_comparison_bar_plot.pdf", dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Ablation classification plots saved to {ablation_dir}:")
        print("  • f1_ablation_plot.pdf")
        print("  • auroc_ablation_plot.pdf")
        print("  • hamming_ablation_plot.pdf")
        print("  • f1_comparison_bar_plot.pdf")