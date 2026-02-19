"""
Minimal LoRA (Low-Rank Adaptation) for DiffusionEdge.

Targets nn.Linear attention projections (q_lin, k_lin, v_lin) in the
U-Net's BasicAttetnionLayer blocks.  No external dependencies.

Phase-0 recon results:
  - BasicAttetnionLayer.q_lin : nn.Linear(embed_dim, embed_dim)  ← target
  - BasicAttetnionLayer.k_lin : nn.Linear(embed_dim, embed_dim)  ← target
  - BasicAttetnionLayer.v_lin : nn.Linear(embed_dim, embed_dim)  ← target
  - LinearAttention.to_qkv   : nn.Conv2d  (skipped)
  - Attention.to_qkv          : nn.Conv2d  (skipped)
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# LoRA Linear wrapper
# ---------------------------------------------------------------------------

class LoRALinear(nn.Module):
    """Drop-in replacement for nn.Linear that adds a low-rank residual.

    Forward:  base(x) + scale * (dropout(x) @ A^T) @ B^T
    where scale = alpha / r.

    Initialization ensures B = 0 so initial output == base(x).
    """

    def __init__(
        self,
        base: nn.Linear,
        r: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert isinstance(base, nn.Linear), f"LoRALinear expects nn.Linear, got {type(base)}"
        assert r > 0, f"LoRA rank must be > 0, got {r}"

        self.base = base
        self.r = r
        self.alpha = alpha
        self.scale = alpha / r

        in_features = base.in_features
        out_features = base.out_features

        # Low-rank matrices  (r << min(in, out))
        self.lora_A = nn.Parameter(torch.empty(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))

        # Kaiming-uniform for A (like Linear default), zeros for B
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # B stays zeros → initial LoRA contribution is zero

        self.lora_dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Base linear
        out = self.base(x)
        # LoRA residual: x @ A^T @ B^T  (works for any batch dims)
        lora_out = self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T
        return out + self.scale * lora_out

    def extra_repr(self) -> str:
        return (
            f"in={self.base.in_features}, out={self.base.out_features}, "
            f"r={self.r}, alpha={self.alpha}, scale={self.scale:.4f}"
        )


# ---------------------------------------------------------------------------
# Injection
# ---------------------------------------------------------------------------

def _matches_any(name: str, patterns: Sequence[str]) -> bool:
    """Return True if *name* contains any pattern as a substring."""
    for p in patterns:
        if p in name:
            return True
    return False


def inject_lora(
    model: nn.Module,
    target_patterns: Sequence[str] = ("q_lin", "k_lin", "v_lin"),
    r: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
) -> List[str]:
    """Replace matching nn.Linear modules with LoRALinear in-place.

    Args:
        model: the root module (e.g. LatentDiffusion or Unet).
        target_patterns: substrings to match against fully-qualified module
            names.  Default matches the attention projections discovered in
            Phase 0 recon.
        r, alpha, dropout: LoRA hyper-parameters.

    Returns:
        List of injected module names.

    Raises:
        RuntimeError: if zero modules matched (likely wrong target names).
    """
    injected: List[str] = []

    # Build list of (parent_module, attr_name, child_module, full_name)
    targets_to_replace: list = []
    for full_name, module in model.named_modules():
        if isinstance(module, nn.Linear) and _matches_any(full_name, target_patterns):
            # Decompose "a.b.c" → parent path "a.b", attr "c"
            parts = full_name.rsplit(".", 1)
            if len(parts) == 2:
                parent_name, attr = parts
                parent = dict(model.named_modules())[parent_name]
            else:
                parent = model
                attr = full_name
            targets_to_replace.append((parent, attr, module, full_name))

    if not targets_to_replace:
        # Collect candidate names to help debugging
        candidates = [
            f"  {name}  {tuple(m.weight.shape)}"
            for name, m in model.named_modules()
            if isinstance(m, nn.Linear)
        ][:50]
        candidates_str = "\n".join(candidates) if candidates else "  (none found)"
        raise RuntimeError(
            f"inject_lora: no nn.Linear modules matched patterns {list(target_patterns)}.\n"
            f"Top candidate nn.Linear modules:\n{candidates_str}"
        )

    for parent, attr, linear, full_name in targets_to_replace:
        lora_layer = LoRALinear(linear, r=r, alpha=alpha, dropout=dropout)
        setattr(parent, attr, lora_layer)
        injected.append(full_name)

    print(f"[LoRA] Injected {len(injected)} LoRALinear modules (r={r}, alpha={alpha}, dropout={dropout}):")
    for name in injected:
        print(f"  → {name}")

    return injected


# ---------------------------------------------------------------------------
# Freeze policy
# ---------------------------------------------------------------------------

def set_trainable_lora_only(model: nn.Module, verbose: bool = True) -> int:
    """Freeze all parameters, then unfreeze only LoRA A/B matrices.

    Returns the count of trainable parameters.
    """
    # 1. Freeze everything
    for p in model.parameters():
        p.requires_grad = False

    # 2. Unfreeze LoRA params
    trainable_names: List[str] = []
    trainable_count = 0
    for name, p in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            p.requires_grad = True
            trainable_names.append(name)
            trainable_count += p.numel()

    total_count = sum(p.numel() for p in model.parameters())

    if verbose:
        print(f"[LoRA freeze] Total params: {total_count:,}  |  Trainable (LoRA): {trainable_count:,}"
              f"  ({100.0 * trainable_count / max(total_count, 1):.2f}%)")
        # Show first N trainable param names
        show_n = min(len(trainable_names), 12)
        for name in trainable_names[:show_n]:
            print(f"  ✓ {name}")
        if len(trainable_names) > show_n:
            print(f"  ... and {len(trainable_names) - show_n} more")

    if trainable_count == 0:
        raise RuntimeError(
            "[LoRA freeze] No LoRA parameters found! "
            "Did you call inject_lora() before set_trainable_lora_only()?"
        )

    # Sanity: assert only lora params are trainable
    for name, p in model.named_parameters():
        if p.requires_grad:
            assert "lora_" in name, (
                f"Non-LoRA parameter '{name}' is still trainable after freeze!"
            )

    return trainable_count


# ---------------------------------------------------------------------------
# Save / Load LoRA weights
# ---------------------------------------------------------------------------

def lora_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Extract only LoRA parameters from model state dict."""
    return {
        name: param.detach().cpu().clone()
        for name, param in model.state_dict().items()
        if "lora_A" in name or "lora_B" in name
    }


def save_lora_checkpoint(
    model: nn.Module,
    save_dir: Union[str, Path],
    step: int,
    r: int,
    alpha: float,
    targets: Sequence[str],
    base_ckpt_path: Optional[str] = None,
    tag: str = "lora",
) -> Path:
    """Save LoRA weights + config JSON to *save_dir*.

    Files written:
        <save_dir>/<tag>_step<step>.pt
        <save_dir>/<tag>_config.json  (overwritten each time)
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Weights
    weights_path = save_dir / f"{tag}_step{step}.pt"
    torch.save(lora_state_dict(model), weights_path)

    # Config
    config = {
        "r": r,
        "alpha": alpha,
        "targets": list(targets),
        "step": step,
        "base_ckpt_path": str(base_ckpt_path) if base_ckpt_path else None,
    }
    config_path = save_dir / f"{tag}_config.json"
    config_path.write_text(json.dumps(config, indent=2))

    print(f"[LoRA] Saved checkpoint → {weights_path}  ({weights_path.stat().st_size / 1024:.1f} KB)")
    return weights_path


def load_lora_state_dict(
    model: nn.Module,
    path: Union[str, Path],
    strict: bool = True,
) -> None:
    """Load LoRA-only state dict onto a model that already has LoRA injected.

    Supports two checkpoint formats:
      - Flat:    {"model.x.lora_A": tensor, "model.x.lora_B": tensor, ...}
      - Wrapped: {"step": int, "r": int, "alpha": float, "targets": list,
                  "lora": {"model.x.lora_A": tensor, ...}}
    """
    path = Path(path)
    raw = torch.load(str(path), map_location="cpu")

    # Auto-detect format
    if isinstance(raw, dict) and "lora" in raw and isinstance(raw["lora"], dict):
        sd = raw["lora"]  # wrapped format
    else:
        sd = raw  # flat format

    # Filter model state dict to only LoRA keys
    model_lora_keys = {
        name for name in model.state_dict().keys()
        if "lora_A" in name or "lora_B" in name
    }

    missing = model_lora_keys - set(sd.keys())
    unexpected = set(sd.keys()) - model_lora_keys

    if strict and (missing or unexpected):
        raise RuntimeError(
            f"LoRA state dict mismatch!\n"
            f"  Missing keys ({len(missing)}): {sorted(missing)[:10]}\n"
            f"  Unexpected keys ({len(unexpected)}): {sorted(unexpected)[:10]}"
        )

    # Load
    model.load_state_dict(sd, strict=False)
    loaded_count = len(sd) - len(unexpected)
    print(f"[LoRA] Loaded {loaded_count} LoRA tensors from {path}")


# ---------------------------------------------------------------------------
# Debug utilities
# ---------------------------------------------------------------------------

def print_linear_modules(model: nn.Module, max_show: int = 60) -> None:
    """Print all nn.Linear modules in *model* with their shapes."""
    linears = [
        (name, m.in_features, m.out_features)
        for name, m in model.named_modules()
        if isinstance(m, nn.Linear)
    ]
    print(f"\n[LoRA debug] All nn.Linear modules ({len(linears)} total):")
    for name, in_f, out_f in linears[:max_show]:
        print(f"  {name:70s}  ({in_f} → {out_f})")
    if len(linears) > max_show:
        print(f"  ... ({len(linears) - max_show} more)")


def print_lora_modules(model: nn.Module) -> None:
    """Print all LoRALinear modules currently in the model."""
    loras = [
        (name, m.base.in_features, m.base.out_features, m.r)
        for name, m in model.named_modules()
        if isinstance(m, LoRALinear)
    ]
    print(f"\n[LoRA debug] LoRALinear modules ({len(loras)} total):")
    for name, in_f, out_f, r in loras:
        print(f"  {name:70s}  ({in_f} → {out_f}, r={r})")
