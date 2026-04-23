"""
Training core for the Inference-Mirror GRPO notebook.
=====================================================

This module is the ONE place where prompts, parsers, rollouts and reward
functions live for GRPO training against the Bioresearch OpenEnv environment.
The companion notebook ``notebooks/train_grpo_inference_mirror.ipynb`` is a
thin driver that imports everything from here; the smoke test
``tests/test_training_core.py`` covers the same code paths offline without
needing a GPU.

Design
------
* **Mirrors ``inference.py``.** We re-use ``SYSTEM_PROMPTS``,
  ``parse_response``, ``parse_lab_response``, ``build_user_prompt``, and
  ``build_lab_prompt`` directly — no duplicated strings or parsers.
* **Task-type-gated rewards.** ``make_reward_fn(task_type)`` returns a
  TRL-compatible reward closure that scores *only* rows whose
  ``task_type`` matches. Wire one per task into ``reward_funcs=[...]`` and
  each task gets its own reward curve in ``trainer.state.log_history``.
* **Multi-turn labs.** Lab rewards actually *run* an ``inference.py``-style
  multi-turn rollout (up to ``TRAIN_LAB_MAX_STEPS`` turns), seeding Turn 1
  with the TRL-generated completion and continuing with ``model.generate``
  under ``torch.inference_mode()`` so the terminal reward reflects real
  tool-use behaviour. Configurable via ``configure_model(...)`` and
  ``TRAIN_LAB_MAX_STEPS``.
* **Auto-reconnect.** ``env_reset`` / ``env_step`` wrap
  ``BioresearchEnv(...).sync()`` with a one-shot reconnect on any
  exception (chiefly ``ConnectionClosedOK`` from OpenEnv's persistent
  WebSocket). This keeps a 200-step training run from dying on a single
  idle timeout.

Public surface
--------------
    configure_env(base_url)                 # set the HTTP base URL
    configure_model(model, tokenizer, ...)  # register a HF model+tok
    env_reset(**kwargs) -> StepResult       # auto-reconnecting reset
    env_step(action)    -> StepResult       # auto-reconnecting step

    LEGACY_TASKS, LAB_TASKS, ALL_TASKS, DEFAULT_T4_TASKS
    build_mixed_dataset(task_list, n_per_task) -> datasets.Dataset
    make_reward_fn(task_type)               -> TRL reward closure
    run_eval_episode(task_id, task_type)    -> float   (inference.py parity)
"""

from __future__ import annotations

import json
import os
import random
from typing import Any, Callable, Dict, List, Optional

# Re-export task ordering from the server so task-list drift is impossible.
try:
    from server.bioresearch_environment import (
        LEGACY_TASK_TYPES as _LEGACY,
        LAB_TASK_TYPES as _LAB,
    )
except ImportError:  # pragma: no cover — defensive fallback
    _LEGACY = (
        "dna_classification",
        "dna_reasoning",
        "evidence_ranking",
        "protein_function",
        "kegg_pathway_reasoning",
        "perturbation_qa",
        "perturbation_direction_qa",
        "perturbation_benchmark",
        "clinical_diagnosis",
    )
    _LAB = (
        "protein_hypothesis_lab",
        "target_discovery_lab",
        "clinical_diagnosis_lab",
        "ligand_design",
        "curriculum_self_play",
    )

# Re-use inference.py's source of truth. Any change to prompts/parsers in
# that file propagates here automatically.
from inference import (
    SYSTEM_PROMPTS,
    build_lab_prompt,
    build_user_prompt,
    parse_lab_response,
    parse_response,
)

# ``inference.py``'s import style — works both when installed as the
# ``bioresearch`` package (``uv sync`` / ``pip install -e .``) and when the
# repo root is on ``sys.path`` directly (Colab after ``git clone``).
try:
    from bioresearch import BioresearchAction, BioresearchEnv
except ImportError:  # pragma: no cover — top-level fallback for Colab
    from client import BioresearchEnv  # type: ignore
    from models import BioresearchAction  # type: ignore


LEGACY_TASKS: List[str] = list(_LEGACY)
LAB_TASKS: List[str] = list(_LAB)
ALL_TASKS: List[str] = LEGACY_TASKS + LAB_TASKS

# T4-friendly default: five single-step tasks plus one lab task gives six
# reward curves in ~60 minutes wall-clock at ``max_steps=200``. Flip to
# ``ALL_TASKS`` on an A100 when you want full 14-curve coverage.
DEFAULT_T4_TASKS: List[str] = [
    "dna_classification",
    "dna_reasoning",
    "evidence_ranking",
    "perturbation_direction_qa",
    "clinical_diagnosis",
    "protein_hypothesis_lab",
]

# Lab rollout budget used *during training*. Deliberately lower than
# ``inference.MAX_LAB_STEPS`` (20) to keep per-step wall-clock reasonable
# on a T4. The held-out eval helper below uses the full 20-step budget.
TRAIN_LAB_MAX_STEPS: int = int(os.environ.get("TRAIN_LAB_MAX_STEPS", "4"))
EVAL_LAB_MAX_STEPS: int = int(os.environ.get("EVAL_LAB_MAX_STEPS", "12"))


# ======================================================================
# Auto-reconnecting env client
# ======================================================================

_ENV_STATE: Dict[str, Any] = {"client": None, "base_url": "http://127.0.0.1:8000"}


def _new_sync_client(base_url: str):
    c = BioresearchEnv(base_url=base_url).sync()
    c.connect()
    return c


def configure_env(base_url: str) -> None:
    """Register the OpenEnv HTTP base URL and open a sync client."""
    _ENV_STATE["base_url"] = base_url
    _ENV_STATE["client"] = _new_sync_client(base_url)


def _ensure_client():
    if _ENV_STATE["client"] is None:
        configure_env(_ENV_STATE["base_url"])
    return _ENV_STATE["client"]


def _recreate_client() -> None:
    old = _ENV_STATE.get("client")
    try:
        if old is not None:
            old.close()
    except Exception:
        pass
    _ENV_STATE["client"] = _new_sync_client(_ENV_STATE["base_url"])


def env_reset(**kwargs):
    """``client.reset(...)`` with one-shot reconnect on any exception.

    Returns a ``StepResult``; read ``.observation.<field>`` for task data.
    """
    client = _ensure_client()
    try:
        return client.reset(**kwargs)
    except Exception as exc:
        print(f"[training_core] env_reset failed ({type(exc).__name__}: {exc}); reconnecting\u2026")
        _recreate_client()
        return _ENV_STATE["client"].reset(**kwargs)


def env_step(action):
    """``client.step(action)`` with one-shot reconnect on any exception."""
    client = _ensure_client()
    try:
        return client.step(action)
    except Exception as exc:
        print(f"[training_core] env_step failed ({type(exc).__name__}: {exc}); reconnecting\u2026")
        _recreate_client()
        return _ENV_STATE["client"].step(action)


# ======================================================================
# Model registry — lab rollouts call ``model.generate`` inside the reward
# function. The notebook calls ``configure_model(model, tokenizer)`` once
# after loading; tests register a ``FakeModel`` stub.
# ======================================================================

_MODEL_REF: Dict[str, Any] = {"model": None, "tokenizer": None, "max_new_tokens": 256}


def configure_model(model, tokenizer, max_new_tokens: int = 256) -> None:
    """Register the HF model + tokenizer used for multi-turn lab rollouts."""
    _MODEL_REF["model"] = model
    _MODEL_REF["tokenizer"] = tokenizer
    _MODEL_REF["max_new_tokens"] = max_new_tokens


# ======================================================================
# Dataset builder
# ======================================================================


def _user_prompt_for(obs, task_type: str) -> str:
    """Per-task user prompt. Lab tasks use the opening brief verbatim;
    legacy tasks use ``inference.build_user_prompt`` which already
    serialises sequences, pathway graphs, and batch entries for us.
    """
    if task_type in LAB_TASKS:
        return (obs.question or "")[:1800]
    return build_user_prompt(obs)[:2400]


def build_mixed_dataset(
    task_list: List[str],
    n_per_task: int = 8,
    seed: int = 42,
):
    """Build a shuffled chat-format dataset with ``task_id`` + ``task_type`` columns.

    TRL forwards every extra column through ``**kwargs`` into the reward
    function, which is how each reward closure knows to reset the env
    against the exact brief its completion was graded on.
    """
    from datasets import Dataset

    rng = random.Random(seed)
    rows: List[Dict[str, Any]] = []
    for task_type in task_list:
        if task_type not in SYSTEM_PROMPTS:
            raise KeyError(f"task_type {task_type!r} has no SYSTEM_PROMPT in inference.py")
        seen_ids: set = set()
        # Cap attempts so we never hang if a task's pool is smaller than
        # ``n_per_task`` — sample with replacement once we exhaust unique ids.
        attempts = 0
        while len(seen_ids) < n_per_task and attempts < n_per_task * 4:
            attempts += 1
            result = env_reset(task_type=task_type)
            obs = result.observation
            if not obs.task_id:
                continue
            if obs.task_id in seen_ids and attempts < n_per_task * 2:
                continue
            seen_ids.add(obs.task_id)
            rows.append({
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPTS[task_type]},
                    {"role": "user", "content": _user_prompt_for(obs, task_type)},
                ],
                "task_id": obs.task_id,
                "task_type": task_type,
            })
    rng.shuffle(rows)
    return Dataset.from_list(rows)


# ======================================================================
# Multi-turn lab rollout (runs inside the lab reward function)
# ======================================================================


def _generate_once(messages: List[Dict[str, str]], max_new_tokens: Optional[int] = None) -> str:
    """Synchronous one-shot ``model.generate`` used for Turn 2..N of lab
    rollouts. Returns the decoded assistant message. Uses ``torch.inference_mode``
    so no graph is built — the only turn that carries gradients is Turn 1,
    which TRL itself generated.
    """
    model = _MODEL_REF.get("model")
    tokenizer = _MODEL_REF.get("tokenizer")
    if model is None or tokenizer is None:
        return ""

    import torch

    mnt = max_new_tokens if max_new_tokens is not None else _MODEL_REF.get("max_new_tokens", 256)
    chat_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(chat_text, return_tensors="pt").to(model.device)
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=mnt,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=pad_id,
        )
    gen_ids = out[0][inputs.input_ids.shape[1]:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True)


def _run_lab_rollout(
    task_id: str,
    task_type: str,
    first_turn_text: str,
    max_steps: int,
) -> float:
    """Run an ``inference.py``-style lab episode against the live env.

    Turn 1 is seeded with ``first_turn_text`` (the completion TRL generated
    — this is where the gradient signal comes from). Subsequent turns use
    ``_generate_once`` with ``torch.inference_mode``. Returns the terminal
    reward clamped to ``[0.01, 0.99]``.
    """
    # Pin the brief — reset returns the opening observation.
    result = env_reset(task_id=task_id, task_type=task_type)
    obs = result.observation

    system_prompt = SYSTEM_PROMPTS[task_type]

    # -------- Turn 1: use TRL's completion --------
    first_action = parse_lab_response(task_id, first_turn_text)
    result = env_step(first_action)
    if result.done:
        return max(0.01, min(0.99, float(result.reward or 0.0)))
    obs = result.observation

    # -------- Turns 2..max_steps: unroll with the model --------
    last_action = first_action
    step_idx = 1
    while step_idx < max_steps:
        user_prompt = build_lab_prompt(obs, step_idx)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        text = _generate_once(messages)
        if not text:
            break  # no model registered — fall through to forced submit.
        action = parse_lab_response(task_id, text)
        last_action = action
        result = env_step(action)
        step_idx += 1
        if result.done:
            return max(0.01, min(0.99, float(result.reward or 0.0)))
        obs = result.observation

    # Budget exhausted — force a final submit so we always get a terminal reward.
    forced = BioresearchAction(
        task_id=task_id,
        submit=True,
        answer=(last_action.answer or "") if last_action else "",
        reasoning=last_action.reasoning if last_action else None,
    )
    result = env_step(forced)
    return max(0.01, min(0.99, float(result.reward or 0.0)))


# ======================================================================
# Task-type-gated reward function factory
# ======================================================================


def _completion_text(completion) -> str:
    """Normalise TRL's chat-style and plain-string completion shapes."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion and isinstance(completion[0], dict):
        return completion[0].get("content", "") or ""
    return str(completion or "")


def _score_legacy(task_type: str, task_id: str, text: str) -> float:
    # Reset to pin the brief, then step with the parsed terminal action.
    # The env is deterministic so this gives a consistent GRPO baseline.
    env_reset(task_id=task_id, task_type=task_type)
    action = parse_response(task_type, task_id, text)
    result = env_step(action)
    return max(0.01, min(0.99, float(result.reward or 0.0)))


def _score_lab(task_type: str, task_id: str, text: str) -> float:
    return _run_lab_rollout(
        task_id=task_id,
        task_type=task_type,
        first_turn_text=text,
        max_steps=TRAIN_LAB_MAX_STEPS,
    )


def make_reward_fn(task_type: str) -> Callable:
    """Factory: returns a TRL-compatible reward closure that scores *only*
    rows whose ``task_type`` matches.

    Signature: ``fn(prompts, completions, **kwargs) -> list[float]``.

    Rows from other tasks return ``0.0`` so this closure's reward column
    in ``trainer.state.log_history`` is cleanly that task's curve.
    """
    if task_type not in ALL_TASKS:
        raise ValueError(f"Unknown task_type: {task_type!r}")
    is_lab = task_type in LAB_TASKS

    def _reward(prompts, completions, **kwargs) -> List[float]:
        task_ids = kwargs.get("task_id") or []
        task_types = kwargs.get("task_type") or []
        out: List[float] = []
        for idx, comp in enumerate(completions):
            row_type = task_types[idx] if idx < len(task_types) else ""
            if row_type != task_type:
                out.append(0.0)
                continue
            tid = task_ids[idx] if idx < len(task_ids) else None
            if not tid:
                # Without a task_id we can't replay the exact brief — score
                # neutrally rather than crashing the step.
                out.append(0.01)
                continue
            text = _completion_text(comp)
            try:
                score = _score_lab(task_type, tid, text) if is_lab \
                    else _score_legacy(task_type, tid, text)
            except Exception as exc:
                print(
                    f"[training_core] reward_{task_type} error on {tid}: "
                    f"{type(exc).__name__}: {exc}"
                )
                score = 0.01
            out.append(float(score))
        return out

    _reward.__name__ = f"reward_{task_type}"
    _reward.__qualname__ = f"reward_{task_type}"
    return _reward


# ======================================================================
# Held-out evaluation (cell 13 of the notebook)
#
# Reuses ``inference.py``'s logic by calling ``_generate_once`` instead of
# an OpenAI endpoint — same rollout shape, same parsers, same reward path.
# ======================================================================


def run_eval_episode(task_id: str, task_type: str) -> float:
    """Evaluate one fresh brief under the trained policy.

    For legacy tasks: one model.generate + one env.step.
    For lab tasks: a full multi-turn rollout (up to ``EVAL_LAB_MAX_STEPS``)
    that mirrors ``inference.py._run_lab_episode`` end-to-end.
    """
    if task_type in LAB_TASKS:
        result = env_reset(task_id=task_id, task_type=task_type)
        obs = result.observation
        system_prompt = SYSTEM_PROMPTS[task_type]
        last_action = None
        for step_idx in range(EVAL_LAB_MAX_STEPS):
            user_prompt = build_lab_prompt(obs, step_idx)
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            text = _generate_once(messages)
            action = parse_lab_response(task_id, text) if text else \
                BioresearchAction(task_id=task_id, submit=True, answer="")
            last_action = action
            result = env_step(action)
            if result.done:
                return max(0.01, min(0.99, float(result.reward or 0.0)))
            obs = result.observation
        # Force a final submit if the budget ran out.
        forced = BioresearchAction(
            task_id=task_id,
            submit=True,
            answer=(last_action.answer or "") if last_action else "",
            reasoning=last_action.reasoning if last_action else None,
        )
        result = env_step(forced)
        return max(0.01, min(0.99, float(result.reward or 0.0)))

    # Legacy / single-step path.
    result = env_reset(task_id=task_id, task_type=task_type)
    obs = result.observation
    messages = [
        {"role": "system", "content": SYSTEM_PROMPTS[task_type]},
        {"role": "user", "content": _user_prompt_for(obs, task_type)},
    ]
    text = _generate_once(messages) or "unknown"
    action = parse_response(task_type, task_id, text)
    result = env_step(action)
    return max(0.01, min(0.99, float(result.reward or 0.0)))


__all__ = [
    "LEGACY_TASKS",
    "LAB_TASKS",
    "ALL_TASKS",
    "DEFAULT_T4_TASKS",
    "TRAIN_LAB_MAX_STEPS",
    "EVAL_LAB_MAX_STEPS",
    "configure_env",
    "configure_model",
    "env_reset",
    "env_step",
    "build_mixed_dataset",
    "make_reward_fn",
    "run_eval_episode",
]
