import copy
import re

import numpy as np
import pandas as pd
import torch
from peft import PeftModel
from sklearn.metrics import f1_score
from transformers import RobertaTokenizerFast, AutoModel

from mechanistic_interpretability.utils.tl_lora_ablation import iter_lora_param_refs


def capture_attn_probs_encoder(wrapper_model, input_ids, attention_mask):
    """
    wrapper_model: ChembertaMultiLabelClassifier (has .roberta)
    Returns list length n_layers with tensors (B, H, S, S)
    """
    encoder = wrapper_model.roberta.base_model.model  # matches your LoRA module names
    with torch.no_grad():
        out = encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
            return_dict=True,
        )
    return list(out.attentions)

def rank_heads_by_attn_change_encoder(hf_off, hf_on, tokenizer_name, smiles_list, device, batch_size=8, max_batches=None):
    rows = []
    cfg = hf_on.roberta.config
    n_layers = cfg.num_hidden_layers
    n_heads = cfg.num_attention_heads
    tokenizer = RobertaTokenizerFast.from_pretrained(tokenizer_name)

    hf_off.eval(); hf_on.eval()

    n = len(smiles_list)
    batch_count = 0

    for i in range(0, n, batch_size):
        if max_batches is not None and batch_count >= max_batches:
            break
        batch_count += 1

        batch_smiles = smiles_list[i:i+batch_size]
        toks = tokenizer(batch_smiles, return_tensors="pt", padding=True, truncation=True).to(device)

        attn_off = capture_attn_probs_encoder(hf_off, toks["input_ids"], toks["attention_mask"])
        attn_on  = capture_attn_probs_encoder(hf_on,  toks["input_ids"], toks["attention_mask"])

        for L in range(n_layers):
            a0 = attn_off[L].detach().float().cpu().numpy()
            a1 = attn_on[L].detach().float().cpu().numpy()
            # mean abs diff over batch and sequence dims -> (H,)
            diff = np.mean(np.abs(a1 - a0), axis=(0, 2, 3))

            for h in range(n_heads):
                rows.append({"layer": L, "head": h, "mean_abs_attn_diff": float(diff[h])})

    df = pd.DataFrame(rows).groupby(["layer", "head"], as_index=False).mean()
    return df.sort_values("mean_abs_attn_diff", ascending=False).reset_index(drop=True)

def ablate_lora_head_qkv(model, layer_idx: int, head_idx: int, *, projections=("query","key","value")):
    """
    Returns a deepcopy(model) where LoRA B rows for the given (layer, head) are zeroed
    for Q/K/V (or chosen projections). This removes LoRA's delta contribution for that head.
    """
    ablated = copy.deepcopy(model)

    cfg = ablated.roberta.config
    n_heads = cfg.num_attention_heads
    d_head = cfg.hidden_size // n_heads
    r0, r1 = head_idx * d_head, (head_idx + 1) * d_head

    pat = re.compile(rf"\.encoder\.layer\.{layer_idx}\.attention\.self\.(query|key|value)$")

    with torch.no_grad():
        for ref in iter_lora_param_refs(ablated):
            m = pat.search(ref.module_name)
            if not m:
                continue
            proj = m.group(1)
            if proj not in projections:
                continue

            # lora_B is (out_dim, r). Zero the head slice in out_dim rows.
            ref.lora_B[r0:r1, :].zero_()

    return ablated

def predict_multilabel(model, tokenizer, smiles_list, device, batch_size=16):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(smiles_list), batch_size):
            batch = smiles_list[i:i+batch_size]
            toks = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(device)
            out = model(**toks)
            logits = out.logits if hasattr(out, "logits") else out
            preds.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(preds, axis=0)

def macro_f1(model, tokenizer, df, smiles_col, target_cols, device, threshold=0.35):
    smiles = df[smiles_col].tolist()
    y_true = df[target_cols].values.astype(int)
    y_prob = predict_multilabel(model, tokenizer, smiles, device=device)
    y_pred = (y_prob >= threshold).astype(int)
    return f1_score(y_true, y_pred, average="macro", zero_division=0)

