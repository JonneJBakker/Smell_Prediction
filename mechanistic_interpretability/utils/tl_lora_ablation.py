"""
LoRA adapter ablation utilities for ChemBERTa / RoBERTa-style models.

This module focuses on *adapter-level* ablations for models trained with PEFT LoRA:
- Full adapter OFF vs ON
- Layer-wise ablation (disable LoRA deltas in selected transformer layers)
- Module-type ablation (e.g., attention vs FFN)
- Rank-component ablation (zero a percentage of LoRA rank directions)
- Continuous scaling (scale LoRA contribution by a factor in [0, 1])

Design goals:
- Works with Hugging Face Transformers + PEFT (PeftModel).
- Does **not** require merging adapters; ablations are applied to the LoRA weights.
- Provides a thin wrapper to run repeated ablations given an `evaluate_fn(model) -> dict`
  that returns your metrics (AUROC, F1, loss, etc.).

Notes:
- This file is intentionally self-contained so you can drop it into your project.
- If you already have an evaluation pipeline, pass it in as `evaluate_fn`.
"""

from __future__ import annotations

import copy
import dataclasses
import random
import re
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union
# add near the top (or inline below)
from torch.nn import ModuleDict

import torch

try:
    from peft import PeftModel  # type: ignore
except Exception as e:  # pragma: no cover
    PeftModel = object  # type: ignore

# -------------------------
# Helpers: finding LoRA modules
# -------------------------

@dataclasses.dataclass(frozen=True)
class LoraParamRef:
    """References a LoRA A/B parameter pair for a specific adapter name on a module."""
    module_name: str
    module: torch.nn.Module
    adapter_name: str
    lora_A: torch.nn.Parameter
    lora_B: torch.nn.Parameter


def _is_peft_model(model: torch.nn.Module) -> bool:
    return hasattr(model, "peft_config") or model.__class__.__name__.lower().startswith("peft")


def iter_lora_param_refs(model: torch.nn.Module) -> Iterable[LoraParamRef]:
    """
    Yield (module_name, module, adapter_name, lora_A_param, lora_B_param) for all LoRA layers.

    Supports common PEFT layouts:
    - module.lora_A / module.lora_B as dict-like {adapter_name: Linear}
    - module.lora_A / module.lora_B as Linear (single adapter)
    """
    for module_name, module in model.named_modules():
        if not (hasattr(module, "lora_A") and hasattr(module, "lora_B")):
            continue

        lora_A = getattr(module, "lora_A")
        lora_B = getattr(module, "lora_B")

        # PEFT typical: dict mapping adapter_name -> nn.Linear
        if isinstance(lora_A, (dict, ModuleDict)) and isinstance(lora_B, (dict, ModuleDict)):
            for adapter_name in lora_A.keys():
                if adapter_name not in lora_B:
                    continue
                a_obj = lora_A[adapter_name]
                b_obj = lora_B[adapter_name]
                if hasattr(a_obj, "weight") and hasattr(b_obj, "weight"):
                    yield LoraParamRef(
                        module_name=module_name,
                        module=module,
                        adapter_name=adapter_name,
                        lora_A=a_obj.weight,
                        lora_B=b_obj.weight,
                    )
        else:
            # Single-adapter layout: lora_A/lora_B are nn.Linear (or similar with .weight)
            if hasattr(lora_A, "weight") and hasattr(lora_B, "weight"):
                yield LoraParamRef(
                    module_name=module_name,
                    module=module,
                    adapter_name=getattr(model, "active_adapter", "default"),
                    lora_A=lora_A.weight,
                    lora_B=lora_B.weight,
                )


def list_lora_targets(model: torch.nn.Module) -> List[str]:
    """Return a sorted list of module names that contain LoRA weights."""
    names = sorted({ref.module_name for ref in iter_lora_param_refs(model)})
    return names


def _extract_layer_index(module_name: str) -> Optional[int]:
    """
    Try to extract a transformer block index from a module name.

    Works for common naming patterns like:
    - encoder.layer.0.
    - roberta.encoder.layer.0.
    - base_model.model.encoder.layer.0.
    """
    m = re.search(r"(?:^|\.)(?:layer|layers)\.(\d+)(?:\.|$)", module_name)
    if not m:
        return None
    return int(m.group(1))


def _matches_any_keyword(module_name: str, keywords: Sequence[str]) -> bool:
    return any(k in module_name for k in keywords)


# -------------------------
# Adapter ON/OFF
# -------------------------

def set_adapters_enabled(model: torch.nn.Module, enabled: bool) -> None:
    """
    Enable or disable adapters on a PEFT model.

    If PEFT exposes dedicated methods, we use them. Otherwise, we fall back to toggling
    a common attribute used by PEFT's LoRA layers (`disable_adapters`).
    """
    # PEFT >= 0.6 commonly offers these
    if hasattr(model, "enable_adapter") and hasattr(model, "disable_adapter"):
        if enabled:
            model.enable_adapter()  # type: ignore[attr-defined]
        else:
            model.disable_adapter()  # type: ignore[attr-defined]
        return

    # Some versions implement "set_adapter" to select active adapter; cannot fully disable.
    # Fall back to disabling at module level.
    for _, module in model.named_modules():
        if hasattr(module, "disable_adapters"):
            try:
                module.disable_adapters = (not enabled)  # type: ignore[attr-defined]
            except Exception:
                pass


# -------------------------
# Ablations
# -------------------------

def ablate_lora_full_off(model: torch.nn.Module) -> torch.nn.Module:
    """Return a deepcopy of model with all LoRA deltas set to 0 (LoRA OFF)."""
    ablated = copy.deepcopy(model)
    with torch.no_grad():
        for ref in iter_lora_param_refs(ablated):
            ref.lora_A.zero_()
            ref.lora_B.zero_()
    return ablated


def ablate_lora_layers(
    model: torch.nn.Module,
    layers_to_ablate: Sequence[int],
) -> torch.nn.Module:
    """Zero LoRA weights only in the specified transformer layer indices."""
    layers_to_ablate = set(int(x) for x in layers_to_ablate)
    ablated = copy.deepcopy(model)
    with torch.no_grad():
        for ref in iter_lora_param_refs(ablated):
            li = _extract_layer_index(ref.module_name)
            if li is not None and li in layers_to_ablate:
                ref.lora_A.zero_()
                ref.lora_B.zero_()
    return ablated


def ablate_lora_by_keywords(
    model: torch.nn.Module,
    keywords: Sequence[str],
    *,
    invert: bool = False
) -> torch.nn.Module:
    """
    Zero LoRA weights for modules whose name contains any of `keywords`.

    Example keywords:
      - ("attention",)        -> attention-path LoRA
      - ("intermediate", "output") -> FFN-path LoRA for many RoBERTa impls
      - ("query", "key", "value")  -> specific projections
    """
    ablated = copy.deepcopy(model)
    with torch.no_grad():
        for ref in iter_lora_param_refs(ablated):
            hit = _matches_any_keyword(ref.module_name, keywords)
            if (hit and not invert) or ((not hit) and invert):
                ref.lora_A.zero_()
                ref.lora_B.zero_()
    return ablated


def ablate_lora_rank_percentage(
    model: torch.nn.Module,
    ablation_percentage: float,
    *,
    seed: int = 42,
    layers_to_ablate: Optional[Sequence[int]] = None,
    keywords: Optional[Sequence[str]] = None,
) -> torch.nn.Module:
    """
    Randomly ablate a percentage of LoRA rank components (directions) by zeroing
    corresponding rows in A and columns in B.

    Args:
        model: PEFT model
        ablation_percentage: float in [0, 1]
        seed: RNG seed for reproducibility
        layers_to_ablate: if given, only apply to these transformer layer indices
        keywords: if given, only apply to module names containing any keyword
    """
    if not (0.0 <= ablation_percentage <= 1.0):
        raise ValueError("ablation_percentage must be in [0, 1]")

    rng = random.Random(seed)
    layer_set = set(layers_to_ablate) if layers_to_ablate is not None else None

    ablated = copy.deepcopy(model)

    with torch.no_grad():
        for ref in iter_lora_param_refs(ablated):
            if layer_set is not None:
                li = _extract_layer_index(ref.module_name)
                if li is None or li not in layer_set:
                    continue
            if keywords is not None and not _matches_any_keyword(ref.module_name, keywords):
                continue

            r = ref.lora_A.shape[0]
            k = int(round(r * ablation_percentage))
            if k <= 0:
                continue
            idx = rng.sample(range(r), k)

            # Zero rows in A and corresponding cols in B
            ref.lora_A[idx, :].zero_()
            ref.lora_B[:, idx].zero_()

    return ablated


def scale_lora_contribution(
    model: torch.nn.Module,
    scale: float,
    *,
    layers_to_scale: Optional[Sequence[int]] = None,
    keywords: Optional[Sequence[str]] = None,
) -> torch.nn.Module:
    """
    Scale LoRA contribution by multiplying LoRA B (or both A & B) by `scale`.

    Why scale B only?
      Because ΔW = B @ A. Scaling either A or B scales ΔW linearly.
      Scaling B is slightly simpler and keeps A as the learned basis.

    Args:
        scale: typically in [0, 1], but can be >1 for amplification experiments.
        layers_to_scale: optionally restrict scaling to certain layers
        keywords: optionally restrict scaling to module names containing keyword(s)
    """
    ablated = copy.deepcopy(model)
    layer_set = set(layers_to_scale) if layers_to_scale is not None else None

    with torch.no_grad():
        for ref in iter_lora_param_refs(ablated):
            if layer_set is not None:
                li = _extract_layer_index(ref.module_name)
                if li is None or li not in layer_set:
                    continue
            if keywords is not None and not _matches_any_keyword(ref.module_name, keywords):
                continue

            ref.lora_B.mul_(scale)

    return ablated


# -------------------------
# Running ablation sweeps
# -------------------------

@dataclasses.dataclass
class AblationResult:
    ablation_name: str
    ablation_kwargs: Dict[str, object]
    seed: Optional[int]
    metrics: Dict[str, float]


def run_ablation_sweep(
    model: torch.nn.Module,
    evaluate_fn: Callable[[torch.nn.Module], Dict[str, float]],
    *,
    ablation_builders: Sequence[Tuple[str, Callable[[], torch.nn.Module]]],
) -> List[AblationResult]:
    """
    Evaluate a list of ablated models produced by `ablation_builders`.

    Each builder is called with no args and must return a ready-to-eval model.
    """
    results: List[AblationResult] = []
    for name, builder in ablation_builders:
        ablated_model = builder()
        ablated_model.eval()
        metrics = evaluate_fn(ablated_model)
        results.append(AblationResult(
            ablation_name=name,
            ablation_kwargs={},
            seed=None,
            metrics=metrics,
        ))
    return results


def build_default_lora_ablation_suite(
    model: torch.nn.Module,
    *,
    rank_percentages: Sequence[float] = (0.1, 0.3, 0.5, 0.8, 1.0),
    rank_seeds: Sequence[int] = (0, 1, 2, 3, 4),
    layer_groups: Optional[Sequence[Sequence[int]]] = None,
    attention_keywords: Sequence[str] = ("attention",),
    ffn_keywords: Sequence[str] = ("intermediate", "output", "dense"),
    scaling_factors: Sequence[float] = (0.0, 0.2, 0.5, 0.8, 1.0),
) -> List[Tuple[str, Callable[[], torch.nn.Module]]]:
    """
    Create a suite of ablation builders covering:
      - full off vs on (identity)
      - layer-wise ablation (full-zero)
      - keyword/module-type ablation (attention vs FFN)
      - rank-percentage ablation
      - continuous scaling of LoRA contribution

    Returns:
      A list of (name, builder_fn) tuples.
    """
    builders: List[Tuple[str, Callable[[], torch.nn.Module]]] = []

    # Identity (LoRA ON)
    builders.append(("lora_on (no ablation)", lambda: copy.deepcopy(model)))

    # Full OFF
    builders.append(("lora_off (all adapters zeroed)", lambda: ablate_lora_full_off(model)))

    # Layer groups (if not provided, infer 0..N-1 and do early/mid/late)
    lora_layers = sorted({i for n in list_lora_targets(model) for i in [_extract_layer_index(n)] if i is not None})
    if layer_groups is None and lora_layers:
        max_layer = max(lora_layers)
        # Rough thirds
        one_third = max(1, (max_layer + 1) // 3)
        layer_groups = [
            list(range(0, one_third)),
            list(range(one_third, min(2 * one_third, max_layer + 1))),
            list(range(min(2 * one_third, max_layer + 1), max_layer + 1)),
        ]

    if layer_groups is not None:
        for grp in layer_groups:
            if len(grp) == 0:
                continue
            name = f"lora_layer_ablation layers={list(grp)}"
            builders.append((name, lambda grp=grp: ablate_lora_layers(model, grp)))

    # Module type ablations
    builders.append((
        f"lora_ablation keywords={list(attention_keywords)}",
        lambda: ablate_lora_by_keywords(model, attention_keywords),
    ))
    builders.append((
        f"lora_ablation keywords={list(ffn_keywords)}",
        lambda: ablate_lora_by_keywords(model, ffn_keywords),
    ))

    # Rank % ablations with seeds (averaging externally is recommended)
    for pct in rank_percentages:
        for sd in rank_seeds:
            name = f"lora_rank_ablation pct={pct:.2f} seed={sd}"
            builders.append((name, lambda pct=pct, sd=sd: ablate_lora_rank_percentage(model, pct, seed=sd)))

    # Continuous scaling factors
    for s in scaling_factors:
        name = f"lora_scale factor={s:.2f}"
        builders.append((name, lambda s=s: scale_lora_contribution(model, s)))

    return builders


# -------------------------
# Example (optional): wiring to your codebase
# -------------------------

def _example_evaluate_fn(model: torch.nn.Module) -> Dict[str, float]:
    """
    Example evaluation function.

    Replace this with your project's evaluation code. It must:
      - take a model in eval mode
      - return a flat dict of numeric metrics

    This stub raises to prevent accidental use.
    """
    raise RuntimeError(
        "Replace `_example_evaluate_fn` with your evaluation function "
        "(e.g., your AUROC/F1 computation over your validation set)."
    )


if __name__ == "__main__":  # pragma: no cover
    # This file is meant to be imported. If you run it directly, we print LoRA targets
    # for quick sanity checking.
    import argparse
    from transformers import AutoModelForSequenceClassification
    from peft import PeftModel

    parser = argparse.ArgumentParser(description="Inspect LoRA targets and optionally run a default ablation suite.")
    parser.add_argument("--base_model_name", type=str, default="DeepChem/ChemBERTa-77M-MLM")
    parser.add_argument("--num_labels", type=int, default=138)
    parser.add_argument("--base_state_dict", type=str, default=None, help="Path to a torch state_dict (.bin) to load.")
    parser.add_argument("--adapter_path", type=str, default=None, help="Path to a PEFT adapter directory or file.")
    parser.add_argument("--print_targets", action="store_true")
    args = parser.parse_args()

    base = AutoModelForSequenceClassification.from_pretrained(args.base_model_name, num_labels=args.num_labels)
    if args.base_state_dict:
        sd = torch.load(args.base_state_dict, map_location="cpu")
        base.load_state_dict(sd)

    if args.adapter_path:
        model = PeftModel.from_pretrained(base, args.adapter_path)
    else:
        model = base

    if args.print_targets:
        targets = list_lora_targets(model)
        print(f"Found {len(targets)} LoRA target modules:")
        for t in targets:
            print(" -", t)
    else:
        print("Run with --print_targets to list LoRA modules. For ablations, import this module and pass evaluate_fn.")
