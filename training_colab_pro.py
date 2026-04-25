"""
Colab Pro auto-tuned helpers for the bioresearch GRPO pipeline.
==============================================================

Companion to [training_core.py](training_core.py) and [training_a100.py](training_a100.py).
Adds two pieces the Colab Pro notebook needs that the existing helpers don't cover:

* **GPU auto-detection** (``detect_gpu``) — Colab Pro hands out either an
  A100 40 GB (Pro+ priority) or an L4 22.5 GB depending on subscription
  tier and current availability, so the notebook needs to pick the right
  GRPOConfig at runtime instead of hard-coding one.

* **Per-card GRPOConfig presets** (``auto_grpo_kwargs``) — different cards
  have different sweet spots for ``num_generations`` × ``max_completion_length``
  × ``max_steps``. We tabulate the proven presets here so the notebook stays
  a thin driver.

Reuses ``training_a100`` for the entire reward / rollout / before-after / Trackio
stack — those helpers are GPU-agnostic and we re-export them so the notebook
only ever has to ``import training_colab_pro as tcp``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

# Re-export every helper the notebook needs. The notebook imports `tcp` and
# touches `tcp.setup_trackio`, `tcp.collect_rollouts_with_pool`, etc.
from training_a100 import (  # noqa: F401  (re-exported)
    setup_trackio,
    trackio_finish,
    collect_rollouts,
    collect_rollouts_with_pool,
    save_rollouts,
    reward_distribution_diagnostic,
    before_after_table,
    render_sample_transcripts,
    make_lab_shaping_callback,
)


# ======================================================================
# GPU detection
# ======================================================================


@dataclass
class GpuProfile:
    """Snapshot of the active CUDA device, used to pick a GRPOConfig preset."""

    name: str  # canonical short name: "A100", "L4", "L40S", "T4", "Unknown"
    raw_name: str  # full ``torch.cuda.get_device_name(0)`` for logging
    vram_gb: float
    compute_cap: Tuple[int, int]
    bf16_supported: bool


# Map device-name substrings to canonical short names. Order matters: longer /
# more specific substrings first so "A100-SXM4" maps to "A100" before any
# accidental "A10" match. We canonicalise so downstream presets only have to
# branch on a small set of names instead of dozens of SKU variants.
_GPU_NAME_MAP = (
    ("A100", "A100"),
    ("L40S", "L40S"),
    ("L40", "L40S"),  # L40 (non-S) is rare on Colab; treat as L40S preset
    ("L4", "L4"),
    ("H100", "H100"),
    ("T4", "T4"),
    ("V100", "V100"),
)


def detect_gpu() -> GpuProfile:
    """Detect the active CUDA device and return a :class:`GpuProfile`.

    Falls back to an "Unknown" profile (Compute Capability 0,0, 0 GB VRAM,
    no bf16) if no GPU is attached — the notebook's ``auto_grpo_kwargs``
    treats Unknown as the L4 preset (conservative, won't OOM bigger cards).
    """
    import torch

    if not torch.cuda.is_available():
        return GpuProfile(
            name="Unknown",
            raw_name="cpu",
            vram_gb=0.0,
            compute_cap=(0, 0),
            bf16_supported=False,
        )

    raw = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    _, total = torch.cuda.mem_get_info()
    vram_gb = total / 1e9
    bf16 = bool(torch.cuda.is_bf16_supported())

    name = "Unknown"
    upper = raw.upper()
    for needle, canonical in _GPU_NAME_MAP:
        if needle in upper:
            name = canonical
            break

    return GpuProfile(
        name=name,
        raw_name=raw,
        vram_gb=vram_gb,
        compute_cap=cap,
        bf16_supported=bf16,
    )


# ======================================================================
# Per-card GRPOConfig presets
# ======================================================================


# Per-card GRPO knobs. Tuned empirically against the L40s and A100 runs:
#
# * ``num_generations`` — GRPO advantage variance scales 1/N. A100's 40 GB
#   comfortably hosts the KV cache for N=12; L4's 22.5 GB taps out around 8.
# * ``max_completion_length`` — Qwen 7B's JSON action+reasoning fits under
#   ~700 tokens for this env, so 768 is the comfort ceiling and 640 is the
#   tighter L4 budget.
# * ``max_steps`` — paired with the dataset (112 prompts at n_per_task=8)
#   and ``per_device_train_batch_size * gradient_accumulation_steps = 4`` to
#   land at ~8.6 epochs on A100 / ~6.4 on L4. Both are inside the no-overfit
#   regime for GRPO with KL ``beta=0.04`` on this task mix.
# * ``per_device_train_batch_size`` / ``gradient_accumulation_steps`` — kept
#   identical across cards so the per-step prompt budget is constant; only
#   the per-step generation budget changes.
_PRESETS: Dict[str, Dict[str, Any]] = {
    "A100": {
        "num_generations": 12,
        "max_completion_length": 768,
        "max_prompt_length": 1280,
        "max_steps": 240,
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 2,
    },
    "L40S": {
        "num_generations": 10,
        "max_completion_length": 640,
        "max_prompt_length": 1280,
        "max_steps": 200,
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 2,
    },
    "L4": {
        "num_generations": 8,
        "max_completion_length": 640,
        "max_prompt_length": 1024,
        "max_steps": 180,
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 2,
    },
    "H100": {
        # H100 has more VRAM and bandwidth than A100; spend it on a sharper
        # advantage estimator (N=16) and a longer training run.
        "num_generations": 16,
        "max_completion_length": 768,
        "max_prompt_length": 1280,
        "max_steps": 280,
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 2,
    },
    "T4": {
        # Conservative T4 fallback so this module is also safe to import in
        # the train_grpo_t4 notebook; mirrors the t4 notebook's settings.
        "num_generations": 4,
        "max_completion_length": 256,
        "max_prompt_length": 1024,
        "max_steps": 80,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 4,
    },
    "V100": {
        # V100 has 16 GB and no bf16 (sm_70). Treat as a slightly larger T4.
        "num_generations": 6,
        "max_completion_length": 384,
        "max_prompt_length": 1024,
        "max_steps": 120,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 4,
    },
}


# Lab rollout depth per card. Each lab task in the reward fn does up to this
# many extra ``model.generate`` calls per completion -- with ~57% of prompts
# being lab tasks and num_generations >= 8, this compounds aggressively, so
# we scale it to per-card wall-clock budget.
_LAB_STEPS: Dict[str, int] = {
    "A100": 4,
    "L40S": 6,
    "L4": 4,
    "H100": 6,
    "T4": 2,
    "V100": 3,
}


def auto_train_lab_max_steps(profile: GpuProfile) -> int:
    """Per-card ``training_core.TRAIN_LAB_MAX_STEPS`` recommendation.

    Each lab task in the reward function does up to this many extra
    ``model.generate`` calls per completion (see ``training_core._generate_once``).
    Defaults to 4 for unknown cards (the L4 setting).
    """
    return _LAB_STEPS.get(profile.name, 4)


def auto_grpo_kwargs(
    profile: GpuProfile,
    trackio_cfg: Dict[str, Any],
    *,
    output_dir: str = "grpo_bioresearch_colab_pro",
    learning_rate: float = 3e-6,
    beta: float = 0.04,
    seed: int = 42,
) -> Dict[str, Any]:
    """Return a fully-populated GRPOConfig kwarg dict tuned for ``profile``.

    Includes:

    * Per-card ``num_generations`` / ``max_completion_length`` / ``max_steps``
      / ``max_prompt_length`` / ``per_device_train_batch_size`` /
      ``gradient_accumulation_steps`` from ``_PRESETS``.
    * Cosine LR schedule with ``warmup_ratio=0.05`` (smoother than constant LR
      on long runs >200 steps, where late-stage updates can otherwise
      destabilise the policy).
    * ``save_steps=30`` for resilience against Colab session disconnects (every
      ~12-17 min depending on card).
    * ``bf16`` if the card supports it (A100/L4/L40S/H100), ``fp16`` otherwise
      (T4/V100).
    * Trackio reporting kwargs from ``trackio_cfg`` splatted in last so the
      caller can override ``report_to`` / ``run_name``.

    Falls back to the L4 preset for unknown cards. The caller should still
    add ``use_vllm`` / ``vllm_mode`` after this returns -- those are notebook
    concerns (vLLM availability detection) rather than per-card concerns.
    """
    preset_key = profile.name if profile.name in _PRESETS else "L4"
    preset = _PRESETS[preset_key]

    use_bf16 = bool(profile.bf16_supported)

    kwargs: Dict[str, Any] = {
        "output_dir": output_dir,
        "learning_rate": learning_rate,
        "logging_steps": 1,
        "save_strategy": "steps",
        "save_steps": 30,
        "bf16": use_bf16,
        "fp16": not use_bf16,
        "beta": beta,
        "max_grad_norm": 1.0,
        "seed": seed,
        # Cosine LR with warmup. The 5% warmup_ratio is a standard sweep-tested
        # value; on a 240-step run that's 12 warmup steps before the cosine
        # decay kicks in, which is enough to stabilise the early KL estimates
        # without burning much of the budget.
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.05,
        # Per-card preset
        **preset,
        # Trackio reporting goes last so caller can override
        **trackio_cfg,
    }
    return kwargs


__all__ = [
    # Re-exports from training_a100
    "setup_trackio",
    "trackio_finish",
    "collect_rollouts",
    "collect_rollouts_with_pool",
    "save_rollouts",
    "reward_distribution_diagnostic",
    "before_after_table",
    "render_sample_transcripts",
    "make_lab_shaping_callback",
    # New in this module
    "GpuProfile",
    "detect_gpu",
    "auto_grpo_kwargs",
    "auto_train_lab_max_steps",
]
