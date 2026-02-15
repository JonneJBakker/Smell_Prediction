# tl_conversion.py - Clean HuggingFace to TransformerLens conversion utilities
"""
This module provides utilities for converting HuggingFace RoBERTa models to 
TransformerLens format with faithful position ID handling.

Key functions:
- create_faithful_tl_model(): Main conversion function
- roberta_to_tl_state_dict(): Weight format conversion
- FaithfulTLRegressor: Wrapper for downstream tasks
"""
from __future__ import annotations

import copy
import types
import torch
from torch import nn
import pandas as pd
from transformers import RobertaModel, RobertaTokenizerFast
from pathlib import Path

import transformer_lens as tl
from typing import Optional

from .inverse_transform import inverse_transform_target
from transformers.models.roberta.modeling_roberta import create_position_ids_from_input_ids
from mechanistic_interpretability.models.chemberta_regressor import ChembertaRegressorWithFeatures
from mechanistic_interpretability.models.chemberta_classifier import ChembertaMultiLabelClassifier


def create_faithful_tl_model(hf_model: RobertaModel, device: Optional[str] = None) -> tl.HookedEncoder:
    """Create a TransformerLens model that is numerically identical to the HF model.
    
    Args:
        hf_model: The HuggingFace RoBERTa model to convert
        device: Device to place the model on
        
    Returns:
        A TransformerLens HookedEncoder that produces identical outputs to hf_model
    """
    cfg = hf_model.config
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create TL config matching HF model
    cfg_tl = tl.HookedTransformerConfig(
        n_layers=cfg.num_hidden_layers,
        d_model=cfg.hidden_size,
        n_heads=cfg.num_attention_heads,
        d_head=cfg.hidden_size // cfg.num_attention_heads,
        d_mlp=cfg.intermediate_size,
        n_ctx=cfg.max_position_embeddings,
        d_vocab=cfg.vocab_size,
        act_fn="gelu",
        attention_dir="bidirectional", # Specifies bidirectionality
        original_architecture="BERT", # Specifies important differences regarding BERT
        eps=cfg.layer_norm_eps,
        use_hook_tokens=True,
    )
    
    # Create TL model and load converted weights
    tl_model = tl.HookedEncoder(cfg_tl).to(device).eval()
    tl_model.load_state_dict(roberta_to_tl_state_dict(hf_model), strict=False)
    
    # CRITICAL: Override embedding to use HF-style position IDs
    hf_emb = hf_model.embeddings
    
    def hf_faithful_embed(self, tokens, one_zero_attention_mask=None):
        """Embedding function that exactly matches HF behavior."""
        # Use HF-style position IDs for faithful reproduction
        # This effectively skips over the padding tokens, otherwise
        # performance of the TL-wrapped model is completely off.
        pos_ids = create_position_ids_from_input_ids(
            tokens, 
            padding_idx=cfg.pad_token_id
        )
        type_ids = torch.zeros_like(tokens)
        
        # Use HF embedding weights but TL's layer norm
        w_tok = hf_emb.word_embeddings(tokens)
        w_pos = hf_emb.position_embeddings(pos_ids)
        w_type = hf_emb.token_type_embeddings(type_ids)
        
        return self.ln(w_tok + w_pos + w_type)
    
    # Bind the faithful embedding function
    tl_model.embed.forward = types.MethodType(hf_faithful_embed, tl_model.embed)
    
    return tl_model


def roberta_to_tl_state_dict(hf_model: RobertaModel):
    """HuggingFace and TransformerLens have different ways of formatting weights.
       This function converts HuggingFace RoBERTa weights to TransformerLens format.
    
    Args:
        hf_model: HuggingFace RoBERTa model
        
    Returns:
        Dictionary of weights in TransformerLens format
    """
    cfg = hf_model.config
    n_heads = cfg.num_attention_heads
    d_model = cfg.hidden_size
    d_head = d_model // n_heads
    
    def _split_qkv(mat: torch.Tensor):
        """
        Reshape QKV matrices from HF to TL format.
        HF: (d_model, d_model) -> TL: (n_heads, d_model, d_head).
        """
        return mat.view(n_heads, d_head, d_model).transpose(1, 2).contiguous()
    
    def _split_o(mat: torch.Tensor):
        """
        Reshape output projection from HF to TL format.
        HF: (d_model, d_model) -> TL: (n_heads, d_head, d_model).
        """
        return mat.T.contiguous().view(n_heads, d_head, d_model)
    
    def _split_bias(vec: torch.Tensor):
        """
        Reshape bias vectors from HF to TL format.
        HF: (d_model) -> TL: (n_heads, d_head).
        """
        return vec.view(n_heads, d_head).contiguous()
    
    # Start with embeddings
    tl_sd = {
        "embed.embed.W_E": hf_model.embeddings.word_embeddings.weight,
        "embed.pos_embed.W_pos": hf_model.embeddings.position_embeddings.weight,
        "embed.ln.w": hf_model.embeddings.LayerNorm.weight,
        "embed.ln.b": hf_model.embeddings.LayerNorm.bias,
    }
    
    # Handle token type embeddings (RoBERTa has only 1 type, TL expects 2).
    # Purely for architectural compatability, token type IDs aren't even used!
    tt = hf_model.embeddings.token_type_embeddings.weight
    if tt.size(0) == 1:
        tt = tt.repeat(2, 1)
    tl_sd["embed.token_type_embed.W_token_type"] = tt
    
    # Convert each transformer layer
    for l, blk in enumerate(hf_model.encoder.layer):
        p = f"blocks.{l}"
        
        # Attention weights
        tl_sd[f"{p}.attn.W_Q"] = _split_qkv(blk.attention.self.query.weight)
        tl_sd[f"{p}.attn.W_K"] = _split_qkv(blk.attention.self.key.weight)
        tl_sd[f"{p}.attn.W_V"] = _split_qkv(blk.attention.self.value.weight)
        tl_sd[f"{p}.attn.b_Q"] = _split_bias(blk.attention.self.query.bias)
        tl_sd[f"{p}.attn.b_K"] = _split_bias(blk.attention.self.key.bias)
        tl_sd[f"{p}.attn.b_V"] = _split_bias(blk.attention.self.value.bias)
        tl_sd[f"{p}.attn.W_O"] = _split_o(blk.attention.output.dense.weight)
        tl_sd[f"{p}.attn.b_O"] = blk.attention.output.dense.bias
        
        # Attention layer norm
        tl_sd[f"{p}.ln1.w"] = blk.attention.output.LayerNorm.weight
        tl_sd[f"{p}.ln1.b"] = blk.attention.output.LayerNorm.bias
        
        # MLP weights (note transpose for TL convention)
        tl_sd[f"{p}.mlp.W_in"] = blk.intermediate.dense.weight.T
        tl_sd[f"{p}.mlp.b_in"] = blk.intermediate.dense.bias
        tl_sd[f"{p}.mlp.W_out"] = blk.output.dense.weight.T
        tl_sd[f"{p}.mlp.b_out"] = blk.output.dense.bias
        
        # MLP layer norm
        tl_sd[f"{p}.ln2.w"] = blk.output.LayerNorm.weight
        tl_sd[f"{p}.ln2.b"] = blk.output.LayerNorm.bias
    
    return tl_sd


class FaithfulTLRegressor(nn.Module):
    """A regressor that wraps a faithful TL model for downstream tasks.
    
    This class provides a drop-in replacement for ChembertaRegressorWithFeatures
    that uses the faithful TL model internally.
    One should call validate_conversion() and test_prediction_equivalence()
    for ensurance the models behave similarly.
    """
    
    def __init__(self, faithful_tl_model: tl.HookedEncoder, mlp_head: nn.Module, dropout_p: float = 0.0,
                 scaler = None, target_column: str = "measured log solubility in mols per litre",
                 train_data: pd.DataFrame = None):
        super().__init__()
        self.tl_model = faithful_tl_model
        self.mlp_head = mlp_head
        self.dropout = nn.Dropout(dropout_p)
        self.pooling_strat = "cls_mean"
        self.scaler = scaler
        self.target_column = target_column
    
    def forward(self, input_ids, attention_mask=None, denormalize: bool = False):
        """Forward pass that matches HF model exactly.
        
        Args:
            input_ids: Tokenized input
            attention_mask: Attention mask
            denormalize: If True and scaler is available, 
                        denormalize predictions to original scale
                        
        Returns:
            Predictions in normalized space (default) or original space (if denormalize=True)
        """
        hidden = self.tl_model.encoder_output(input_ids, one_zero_attention_mask=attention_mask)
        cls_token = hidden[:, 0, :]  # Extract CLS token
        if self.pooling_strat == "cls_mean":
            mask = attention_mask.unsqueeze(-1).type_as(hidden)  # (batch, seq_len, 1)
            summed = (hidden * mask).sum(dim=1)  # (batch, hidden)
            denom = mask.sum(dim=1).clamp(min=1e-9)
            mean = summed/denom
            cls_token = torch.cat([mean, cls_token], dim=1)

        if self.pooling_strat == "mean":
            mask = attention_mask.unsqueeze(-1).type_as(hidden)  # (batch, seq_len, 1)
            summed = (hidden * mask).sum(dim=1)  # (batch, hidden)
            denom = mask.sum(dim=1).clamp(min=1e-9)
            mean = summed / denom
            cls_token = mean

        predictions = self.mlp_head(self.dropout(cls_token)).squeeze(-1)
        
        if denormalize and self.scaler and self.target_column:
            predictions = self.denormalize_predictions(predictions)
            
        return predictions
    
    def denormalize_predictions(self, predictions):
        """Denormalize predictions to original scale.
        
        Args:
            predictions: Tensor of predictions in normalized space
            
        Returns:
            Tensor of predictions in original scale
        """
        if not self.scaler or not self.target_column:
            return predictions
                    
        # Convert to numpy for inverse transform
        predictions_np = predictions.detach().cpu().numpy()
        
        # Apply inverse transform
        denormalized = inverse_transform_target(
            predictions_np, 
            self.normalization_pipeline, 
            self.target_column
        )
        
        # Convert back to tensor with same device/dtype
        return torch.from_numpy(denormalized).to(predictions.device).to(predictions.dtype)
    
    def predict_denormalized(self, input_ids, attention_mask=None):
        """Convenience method to get denormalized predictions."""
        return self.forward(input_ids, attention_mask, denormalize=True)


def load_chemberta_models(model_path: str, tokenizer_name: str = "DeepChem/ChemBERTa-77M-MLM", 
                          device: Optional[str] = None, scaler_path: str = None,
                          hyperparams_path: str = None, train_data: pd.DataFrame = None):
    """Load both HF and faithful TL versions of a ChemBERTa model.
    
    Args:
        model_path: Path to the finetuned ChemBERTa model
        tokenizer_name: Name of the tokenizer to use
        device: Device to place models on
        scaler_path: Optional path to scaler pickle file
        hyperparams_path: Optional path to hyperparameters JSON file. If None, will try to 
                         find it automatically in the same directory as model_path
        
    Returns:
        Tuple of (hf_encoder, tl_encoder, tokenizer, hf_regressor, tl_regressor, scaler)
    """
    
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load hyperparameters from JSON file
    if hyperparams_path is None:
        # Try to find hyperparameters.json in the same directory as the model
        model_dir = Path(model_path).parent
        hyperparams_path = model_dir / "hyperparameters.json"
    
    # Load hyperparameters with fallback to defaults
    hyperparams = {
        "dropout": 0.34,
        "hidden_channels": 384,
        "num_mlp_layers": 1,
    }
    
    if Path(hyperparams_path).exists():
        try:
            import json
            with open(hyperparams_path, 'r') as f:
                loaded_hyperparams = json.load(f)
            hyperparams.update(loaded_hyperparams)
            print(f"Loaded hyperparameters from {hyperparams_path}")
            print(f"Hyperparameters: {hyperparams}")
        except Exception as e:
            print(f"Could not load hyperparameters from {hyperparams_path}: {e}")
            print(f"Using defaults: {hyperparams}")
    else:
        print(f"Hyperparameters file not found at {hyperparams_path}")
        print(f"Using defaults: {hyperparams}")
    
    # Load original HF model using loaded hyperparameters
    hf_regressor = ChembertaMultiLabelClassifier(
        pretrained=tokenizer_name,
        num_features=0, # TODO: Very cool mech interp would be with numerical features as well, but for now not possible
        dropout=hyperparams["dropout"],
        hidden_channels=hyperparams["hidden_channels"],
        num_mlp_layers=hyperparams["num_mlp_layers"],
        num_labels=138,
        gamma=hyperparams["gamma"],
        alpha=hyperparams["alpha"],
        pooling_strat=hyperparams["pooling_strat"],
    ).to(device).eval()
    
    hf_regressor.load_state_dict(torch.load(model_path, map_location=device), strict=True)
    hf_encoder = hf_regressor.roberta  # Extract the RoBERTa encoder
    
    # Load scaler first if provided
    scaler = None
    if scaler_path and Path(scaler_path).exists():
        import pickle
        print("yes")
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        print(f"Loaded scaler from {scaler_path}")
    
    # Create faithful TL version using the extracted encoder
    tl_encoder = create_faithful_tl_model(hf_encoder, device)

    hf_internals = hf_regressor.mlp.model

    tl_head = copy.deepcopy(hf_internals).to(device).eval()
    tl_regressor = FaithfulTLRegressor(tl_encoder, tl_head, dropout_p=hf_regressor.dropout.p, scaler=scaler, train_data=train_data).to(device).eval()
    
    # Load tokenizer
    tokenizer = RobertaTokenizerFast.from_pretrained(tokenizer_name)
    
    return hf_encoder, tl_encoder, tokenizer, hf_regressor, tl_regressor, scaler

def convert_ablated_hf(hf_encoder, hf_regressor, tokenizer_name: str = "DeepChem/ChemBERTa-77M-MLM",
                          device: Optional[str] = None, scaler_path: str = None,
                          train_data: pd.DataFrame = None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tl_encoder = create_faithful_tl_model(hf_encoder, device)

    scaler = None
    if scaler_path and Path(scaler_path).exists():
        import pickle
        print("yes")
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        print(f"Loaded scaler from {scaler_path}")

    hf_internals = hf_regressor.mlp

    tl_head = copy.deepcopy(hf_internals).to(device).eval()
    tl_regressor = FaithfulTLRegressor(tl_encoder, tl_head, dropout_p=hf_regressor.dropout.p, scaler=scaler,
                                       train_data=train_data).to(device).eval()

    # Load tokenizer
    tokenizer = RobertaTokenizerFast.from_pretrained(tokenizer_name)
    return hf_encoder, tl_encoder, tokenizer, hf_regressor, tl_regressor