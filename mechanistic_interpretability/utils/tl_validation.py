# tl_validation.py - Validation and testing utilities for faithful TL conversion
"""
This module provides comprehensive validation and testing functions for 
verifying that TransformerLens models are faithful to their HuggingFace counterparts.

Key functions:
- validate_conversion(): Layer-by-layer equivalence checking
- run_evaluation_metrics(): Performance evaluation (RMSE, R²)
- test_prediction_equivalence(): End-to-end prediction comparison
"""
from __future__ import annotations

import math
from typing import Dict, Optional
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, accuracy_score, roc_auc_score, f1_score, \
    hamming_loss, jaccard_score
from torch.utils.data import DataLoader
from transformers import RobertaModel, RobertaTokenizerFast
import transformer_lens as tl

from .tl_conversion import load_chemberta_models, FaithfulTLRegressor
from .chemberta_dataset import ChembertaDataset
from mechanistic_interpretability.models.chemberta_regressor import ChembertaRegressorWithFeatures


def validate_conversion(hf_model: RobertaModel, tl_model: tl.HookedEncoder, 
                       test_input_ids: torch.Tensor, test_attention_mask: torch.Tensor) -> Dict[str, float]:
    """Verify that the TL model produces identical outputs to the HF model.
    
    Args:
        hf_model: Original HuggingFace RoBERTa model
        tl_model: Converted TransformerLens model
        test_input_ids: Test input token IDs
        test_attention_mask: Test attention mask
        
    Returns:
        Dictionary with maximum absolute differences at each layer
    """
    with torch.no_grad():
        # Run both models
        hf_out = hf_model(
            input_ids=test_input_ids,
            attention_mask=test_attention_mask,
            output_hidden_states=True,
            return_dict=True
        )
        
        _, tl_cache = tl_model.run_with_cache(
            test_input_ids,
            one_zero_attention_mask=test_attention_mask
        )
        
        # Also get the encoder output for final comparison
        tl_encoder_out = tl_model.encoder_output(
            test_input_ids,
            test_attention_mask
        )
    # Compare outputs layer by layer
    diffs = {}
    
    # Embedding layer
    embed_diff = (tl_cache["hook_full_embed"] - hf_out.hidden_states[0]).abs().max().item()
    diffs["embedding"] = embed_diff
    
    # Each transformer layer (important to specify after final LN for TL)
    for l in range(len(hf_model.encoder.layer)):
        layer_diff = (tl_cache[f"blocks.{l}.hook_normalized_resid_post"] - hf_out.hidden_states[l + 1]).abs().max().item()
        diffs[f"layer_{l}"] = layer_diff
    
    # Final encoder output (most important for interpretability)
    final_diff = (tl_encoder_out - hf_out.last_hidden_state).abs().max().item()
    diffs["final_output"] = final_diff
    
    return diffs


def run_evaluation_metrics(model, test_data, tokenizer,
                          smiles_column: str = "smiles", target_column: str = "measured log solubility in mols per litre", 
                          batch_size: int = 64, device: Optional[str] = None,
                          use_tl_model: bool = True, mode = "classification",
                          scaler = None) -> Dict[str, float]:
    """Run evaluation metrics (RMSE, R²) on test data.
    
    Args:
        model: The trained model (either Huggingface or TL)
        test_data: Test data
        tokenizer: Tokenizer
        smiles_col: Name of SMILES column
        target_col: Name of target column
        batch_size: Batch size for evaluation
        device: Device to use
        use_tl_model: Whether to use TL model instead of HF model
        scaler: Scaler containing training set statistics
        target_column: Target column name for normalization pipeline lookup
        
    Returns:
        Dictionary with evaluation metrics
    """
    
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    
    # Enable cuDNN autotuning for faster GPU operations
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
    
    # Load test data smiles and targets
    texts = test_data[smiles_column].tolist()
    labels = test_data[target_column].astype("float32").values
    model.eval()

    if mode == 'regression':
        # Use training set normalization parameters if available
        if scaler:
            # Use training set normalization parameters
            mean = scaler.mean_[0]
            std = scaler.scale_[0]

            print(f"Using training set normalization: mean={mean:.4f}, std={std:.4f}")
        else:
            print("Warning: No scaler provided, using test set statistics")
            # Normalize targets (should match training normalization)
            mean, std = labels.mean(), labels.std()

        labels_norm = (labels - mean) / std

        ds = ChembertaDataset(texts, labels_norm, tokenizer, features=None)
        loader = DataLoader(ds, batch_size=batch_size, num_workers=2, pin_memory=True if device.type == 'cuda' else False)

        # Run evaluation
        preds, lbls = [], []
        with torch.no_grad():
            for batch in loader:
                ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                y = batch["labels"].to(device)

                # Use automatic mixed precision for faster inference on GPU
                with torch.amp.autocast('cuda', enabled=device.type == 'cuda'):
                    y_hat = model(ids, attention_mask)
                    if not use_tl_model:
                        y_hat = y_hat.logits.squeeze(-1)

                preds.append(y_hat.cpu())
                lbls.append(y.cpu())

        preds = torch.cat(preds).numpy()
        lbls = torch.cat(lbls).numpy()

        # Calculate metrics
        mse_norm = mean_squared_error(lbls, preds)
        rmse_norm = math.sqrt(mse_norm)
        r2_norm = r2_score(lbls, preds)
        mae_norm = mean_absolute_error(lbls, preds)
        mse = mean_squared_error(lbls * std + mean, preds * std + mean)
        rmse = math.sqrt(mse)
        r2 = r2_score(lbls * std + mean, preds * std + mean)
        mae = mean_absolute_error(lbls * std + mean, preds * std + mean)

        return {
            "mse": mse,
            "rmse": rmse,
            "r2": r2,
            "mae": mae,
            "mse_norm": mse_norm,
            "rmse_norm": rmse_norm,
            "r2_norm": r2_norm,
            "mae_norm": mae_norm,
            "model_type": "TL" if use_tl_model else "HF",
        }
    elif mode == 'classification':
            # Multi-label classification: labels are taken from test_data[target_column]
            # target_column can be a single column or a list of columns
            texts = test_data[smiles_column].tolist()
            labels = test_data[target_column].values.astype("float32")  # shape: (n_samples, n_labels) or (n_samples,)

            # Build dataset with labels
            ds = ChembertaDataset(texts, labels, tokenizer, features=None)
            loader = DataLoader(
                ds,
                batch_size=batch_size,
                num_workers=2,
                pin_memory=True if device.type == 'cuda' else False,
            )

            preds, lbls = [], []

            with torch.no_grad():
                for batch in loader:
                    ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    y = batch["labels"].to(device)

                    # Use automatic mixed precision for faster inference on GPU
                    with torch.amp.autocast('cuda', enabled=device.type == 'cuda'):
                        y_hat = model(ids, attention_mask)
                        if not use_tl_model:
                            # HF model returns an object with .logits
                            y_hat = y_hat.logits

                    preds.append(y_hat.cpu())
                    lbls.append(y.cpu())

            logits = torch.cat(preds).numpy()  # (N, num_labels)
            labels = torch.cat(lbls).numpy()  # (N, num_labels)

            # Sigmoid → probabilities
            probs = 1.0 / (1.0 + np.exp(-logits))

            # Thresholding for multi-label classification
            threshold = 0.3492848181402972
            y_pred = (probs >= threshold).astype(int)

            # Overall metrics
            micro_accuracy = accuracy_score(labels, y_pred)  # subset accuracy
            auroc = roc_auc_score(labels, probs, average="macro", multi_class="ovr") \
                if labels.shape[1] > 1 else roc_auc_score(labels, probs)

            micro_f1 = f1_score(labels, y_pred, average="micro", zero_division=0)
            macro_f1 = f1_score(labels, y_pred, average="macro", zero_division=0)
            samples_f1 = f1_score(labels, y_pred, average="samples", zero_division=0)
            hamming = hamming_loss(labels, y_pred)
            jaccard_samples = jaccard_score(labels, y_pred, average="samples", zero_division=0)

            print(f"AUROC: {auroc}")
            print("\nMacro F1:", macro_f1)
            print("Micro F1:", micro_f1)

            return {
                "micro_accuracy": micro_accuracy,
                "auroc": auroc,
                "micro_f1": micro_f1,
                "macro_f1": macro_f1,
                "samples_f1": samples_f1,
                "hamming_loss": hamming,
                "jaccard_samples": jaccard_samples,
            }


def test_prediction_equivalence(hf_regressor: RobertaModel, tl_regressor: tl.HookedEncoder,
                                test_molecules: list[str], tokenizer: RobertaTokenizerFast,
                                device: Optional[str] = None) -> Dict[str, float]:
    """Test prediction equivalence between HF and TL models.
    
    Args:
        model_path: Path to the trained model
        test_molecules: List of SMILES strings to test
        tokenizer_name: Name of the tokenizer
        device: Device to use
        
    Returns:
        Dictionary with prediction differences
    """
    
    if not device:
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    
    results = {}
    max_diff = 0.0
    
    with torch.no_grad():
        for smiles in test_molecules:
            # Tokenize
            inputs = tokenizer(smiles, return_tensors="pt").to(device)
            
            # Get predictions
            # HF class was made to also output loss, hence the .logits
            hf_pred = hf_regressor(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"]
            ).logits.squeeze(0)
            
            tl_pred = tl_regressor(
                inputs["input_ids"],
                inputs["attention_mask"]
            ).squeeze(-1)
            
            diff_vec = abs(hf_pred - tl_pred)
            diff = diff_vec.max().item()
            max_diff = max(max_diff, diff)
            
            results[smiles] = {
                "hf_prediction": hf_pred,
                "tl_prediction": tl_pred,
                "difference": diff
            }
    
    results["max_difference"] = max_diff
    results["is_equivalent"] = max_diff < 1e-5
    
    return results


def comprehensive_validation(model_path: str, test_csv: str, 
                           tokenizer_name: str = "DeepChem/ChemBERTa-77M-MLM",
                           test_molecules: Optional[list[str]] = None,
                           device: Optional[str] = None,
                           target: str = "measured log solubility in mols per litre",
                           smiles: str = "smiles") -> Dict:
    """Run comprehensive validation of TL conversion.
    
    Args:
        model_path: Path to the trained model
        test_csv: Path to test CSV file
        tokenizer_name: Name of the tokenizer
        test_molecules: Optional list of test molecules
        device: Device to use
        
    Returns:
        Dictionary with all validation results
    """
    from .tl_conversion import load_chemberta_models
    
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    
    print("🔍 Running comprehensive validation...")
    
    # Load models
    print("Loading models...")
    hf_encoder, tl_encoder, tokenizer, hf_regressor, _ = load_chemberta_models(
        model_path, tokenizer_name, device
    )
    
    # Test conversion with a simple molecule
    print("Validating conversion...")
    test_smiles = "CCO"
    inputs = tokenizer(test_smiles, return_tensors="pt").to(device)
    conversion_results = validate_conversion(
        hf_encoder, tl_encoder,
        inputs["input_ids"], inputs["attention_mask"]
    )
    
    # Test evaluation metrics
    print("Running evaluation metrics...")
    hf_metrics = run_evaluation_metrics(
        model_path, test_csv, tokenizer_name, device=device, use_tl_model=False, smiles_col=smiles, target_col=target
    )
    tl_metrics = run_evaluation_metrics(
        model_path, test_csv, tokenizer_name, device=device, use_tl_model=True, smiles_col=smiles, target_col=target
    )
    
    # Test prediction equivalence
    if test_molecules is None:
        test_molecules = ["CCO", "c1ccccc1", "CC(C)O"]
    
    print("Testing prediction equivalence...")
    prediction_results = test_prediction_equivalence(
        model_path, test_molecules, tokenizer_name, device
    )
    
    # Compile results
    results = {
        "conversion_validation": conversion_results,
        "hf_metrics": hf_metrics,
        "tl_metrics": tl_metrics,
        "prediction_equivalence": prediction_results,
        "summary": {
            "conversion_faithful": conversion_results["final_output"] < 1e-5,
            "predictions_equivalent": prediction_results["is_equivalent"],
            "metrics_difference": {
                "rmse_diff": abs(hf_metrics["rmse"] - tl_metrics["rmse"]),
                "r2_diff": abs(hf_metrics["r2"] - tl_metrics["r2"])
            }
        }
    }
    
    # Print summary
    print("\n✅ Validation Summary:")
    print(f"  Conversion faithful: {results['summary']['conversion_faithful']}")
    print(f"  Predictions equivalent: {results['summary']['predictions_equivalent']}")
    print(f"  Final output diff: {conversion_results['final_output']:.2e}")
    print(f"  Max prediction diff: {prediction_results['max_difference']:.2e}")
    print(f"  RMSE difference: {results['summary']['metrics_difference']['rmse_diff']:.6f}")
    
    return results 