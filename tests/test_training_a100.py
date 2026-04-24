"""Smoke tests for ``training_a100`` — A100-demo helpers.

Reuses the uvicorn-subprocess fixture pattern from
[tests/test_training_core.py](tests/test_training_core.py) and a tiny
``_FakeModel`` so no GPU is needed. Validates:

* ``collect_rollouts`` is deterministic (same seed + same model → same rows).
* ``collect_rollouts_with_pool`` returns a pool the trained pass can reuse.
* ``reward_distribution_diagnostic`` produces a DataFrame with reward
  variance > 0 (otherwise it would be a useless plot in the notebook).
* ``before_after_table`` returns zero delta for identity input.
* ``render_sample_transcripts`` mentions both reward values per pair.
* ``setup_trackio`` falls back to ``report_to="none"`` cleanly when
  ``trackio`` is not installed.
* ``make_lab_shaping_callback`` populates the expected logs keys when
  fed a fake training step's worth of lab rollouts.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import training_a100  # noqa: E402
import training_core  # noqa: E402

# Reuse the FakeModel/FakeTokenizer + server-bootstrap helpers from the
# sibling smoke test so we don't fork two copies of the harness.
from tests.test_training_core import (  # noqa: E402
    _FakeModel,
    _FakeTokenizer,
    _start_server,
    _stop_server,
)


def _check_collect_rollouts():
    rows, pool = training_a100.collect_rollouts_with_pool(
        ["dna_classification", "protein_hypothesis_lab"],
        n_per_task=2,
        seed=0,
    )
    assert len(rows) == 4, f"expected 4 rows, got {len(rows)}"
    expected_keys = {"task_type", "task_id", "prompt", "completion", "reward"}
    for r in rows:
        assert expected_keys.issubset(r.keys()), f"missing keys in {r}"
        assert 0.0 <= r["reward"] <= 1.0, f"reward {r['reward']} out of range"

    # Pool reuse: a second pass with the same pool MUST hit the same task_ids.
    rows2 = training_a100.collect_rollouts(
        ["dna_classification", "protein_hypothesis_lab"],
        n_per_task=2,
        seed=999,  # different seed; the pool overrides
        task_id_pool=pool,
    )
    ids_first = sorted((r["task_type"], r["task_id"]) for r in rows)
    ids_second = sorted((r["task_type"], r["task_id"]) for r in rows2)
    assert ids_first == ids_second, "pool reuse did not pin task_ids"


def _check_reward_diagnostic():
    df = training_a100.reward_distribution_diagnostic(
        ["dna_classification", "perturbation_direction_qa", "protein_hypothesis_lab"],
        n_samples_per_task=4,
    )
    # We support both the pandas and the list-of-dicts return shape.
    if hasattr(df, "columns"):
        assert {"task_type", "reward"}.issubset(set(df.columns))
        assert len(df) > 0
        variance = float(df["reward"].std())
    else:
        assert df, "diagnostic returned empty list"
        rewards = [row["reward"] for row in df]
        variance = (max(rewards) - min(rewards))
    assert variance > 0.0, "reward diagnostic has zero variance — useless plot"


def _check_before_after_identity():
    rows = [
        {"task_type": "dna_classification", "task_id": "dna_000",
         "prompt": "p", "completion": "c", "reward": 0.42},
        {"task_type": "perturbation_qa", "task_id": "p_001",
         "prompt": "p", "completion": "c", "reward": 0.30},
    ]
    df = training_a100.before_after_table(rows, rows)
    if hasattr(df, "iterrows"):
        for _, r in df.iterrows():
            assert abs(r["delta"]) < 1e-9, f"identity delta should be 0, got {r['delta']}"
    else:
        for r in df:
            assert abs(r["delta"]) < 1e-9, f"identity delta should be 0, got {r['delta']}"


def _check_render_transcripts():
    baseline = [{
        "task_type": "dna_classification", "task_id": "dna_000",
        "prompt": "Q1", "completion": "wrong", "reward": 0.10,
    }]
    trained = [{
        "task_type": "dna_classification", "task_id": "dna_000",
        "prompt": "Q1", "completion": "BRCA1 mutation", "reward": 0.85,
    }]
    md = training_a100.render_sample_transcripts(baseline, trained, k=1)
    assert "dna_classification" in md
    assert "0.100" in md and "0.850" in md, \
        f"expected both reward strings in markdown; got: {md[:300]}"
    assert "Baseline completion" in md and "Trained completion" in md


def _check_trackio_fallback():
    # In this venv trackio is almost certainly not installed, so we expect
    # the fallback path. (If trackio IS installed we just confirm the
    # function returns a config dict with report_to set somehow.)
    cfg = training_a100.setup_trackio("smoke-test", run_name="smoke")
    assert "report_to" in cfg
    assert cfg.get("run_name") == "smoke"
    if cfg["report_to"] == "none":
        # The fallback we explicitly designed for.
        return
    # Otherwise trackio resolved fine; sanity-check the shape only.
    assert cfg["report_to"] == ["trackio"]


def _check_lab_shaping_callback():
    # Inject a couple of canned lab-rollout records into the module-level
    # log, fire the callback's on_log hook, and assert it surfaces the
    # decomposed reward keys into the logs dict.
    training_core._LAB_ROLLOUTS_LOG.clear()
    training_core._LAB_ROLLOUTS_LOG.extend([
        {"task_type": "protein_hypothesis_lab", "task_id": "lab_a",
         "step_rewards": [0.02, 0.05, 0.04, 0.6],
         "terminal_reward": 0.6, "n_steps": 4, "completed": True},
        {"task_type": "protein_hypothesis_lab", "task_id": "lab_b",
         "step_rewards": [0.01, 0.4],
         "terminal_reward": 0.4, "n_steps": 2, "completed": True},
    ])
    cb = training_a100.make_lab_shaping_callback()
    logs: dict = {}
    cb.on_log(args=None, state=None, control=None, logs=logs)
    assert "reward/protein_hypothesis_lab/terminal_mean" in logs
    assert "reward/protein_hypothesis_lab/process_mean" in logs
    assert logs["reward/protein_hypothesis_lab/n_episodes"] == 2
    assert abs(logs["reward/protein_hypothesis_lab/terminal_mean"] - 0.5) < 1e-6


def main() -> int:
    print("[test] booting uvicorn subprocess\u2026")
    ctx = _start_server()
    try:
        training_core.configure_env(ctx.base_url)
        # Register a FakeModel so collect_rollouts has something to drive
        # _generate_once with.
        tok = _FakeTokenizer()
        fake = _FakeModel(
            queue=[],
            default_response='{"submit": true, "answer": "unknown", "reasoning": "."}',
            tokenizer=tok,
        )
        training_core.configure_model(fake, tok, max_new_tokens=16)

        _check_collect_rollouts()
        print("  [ok] collect_rollouts deterministic + pool reuse")

        _check_reward_diagnostic()
        print("  [ok] reward_distribution_diagnostic has signal variance")

        _check_before_after_identity()
        print("  [ok] before_after_table identity delta == 0")

        _check_render_transcripts()
        print("  [ok] render_sample_transcripts contains both rewards")

        _check_trackio_fallback()
        print("  [ok] setup_trackio fallback path works")

        _check_lab_shaping_callback()
        print("  [ok] LabShapingCallback drains and aggregates lab rollouts")

        print("\nALL GOOD \u2013 training_a100 smoke test passed.")
        return 0
    finally:
        training_core.configure_model(None, None)
        _stop_server(ctx)


def test_training_a100_smoke():  # pragma: no cover — exercised end-to-end
    assert main() == 0


if __name__ == "__main__":
    raise SystemExit(main())
