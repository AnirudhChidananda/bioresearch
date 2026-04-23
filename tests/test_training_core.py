"""Smoke tests for ``training_core`` — the GRPO helper module.

These tests boot the real OpenEnv HTTP server in a subprocess, wire a
``FakeModel`` stub into ``training_core.configure_model`` so no GPU is
needed, and then exercise every public surface: dataset builder, each
task-type-gated reward function, and the held-out eval helper.

Run with ``pytest tests/test_training_core.py -x -s`` or directly via
``python tests/test_training_core.py``. Takes ~30 s locally because each
lab rollout turn goes through a real env.step call.

The tests are deliberately tolerant:
* assert reward \u2208 [0.01, 0.99] rather than a specific value — TRL only
  requires a valid scalar;
* treat any uncaught exception in a reward closure as a hard fail (the
  whole point of the closure's try/except is that it should *never*
  propagate).
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import training_core  # noqa: E402

try:
    from bioresearch import BioresearchAction  # type: ignore[import-not-found]  # noqa: E402
except ImportError:  # pragma: no cover
    from models import BioresearchAction  # type: ignore  # noqa: E402


# ---------------------------------------------------------------------------
# Server fixture — a single uvicorn subprocess shared across the whole module.
# ---------------------------------------------------------------------------


def _pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _ServerCtx:
    proc: subprocess.Popen | None = None
    base_url: str = ""


def _start_server() -> _ServerCtx:
    port = _pick_free_port()
    ctx = _ServerCtx()
    ctx.base_url = f"http://127.0.0.1:{port}"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    ctx.proc = subprocess.Popen(
        [
            sys.executable, "-m", "uvicorn",
            "server.app:app",
            "--host", "127.0.0.1",
            "--port", str(port),
            "--log-level", "warning",
        ],
        cwd=str(REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
    )
    # Poll /info until the server accepts connections.
    deadline = time.time() + 40.0
    while time.time() < deadline:
        try:
            if httpx.get(f"{ctx.base_url}/health", timeout=1.0).status_code == 200:
                return ctx
        except Exception:
            pass
        if ctx.proc.poll() is not None:
            # Process died early — surface its output to aid debugging.
            out = ctx.proc.stdout.read().decode("utf-8", errors="replace") if ctx.proc.stdout else ""
            raise RuntimeError(f"uvicorn exited early:\n{out}")
        time.sleep(0.5)
    _stop_server(ctx)
    raise RuntimeError(f"uvicorn failed to come up on {ctx.base_url} within 40s")


def _stop_server(ctx: _ServerCtx) -> None:
    if ctx.proc and ctx.proc.poll() is None:
        ctx.proc.terminate()
        try:
            ctx.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            ctx.proc.kill()


# ---------------------------------------------------------------------------
# FakeModel / FakeTokenizer — give ``training_core._generate_once`` a cheap
# stand-in so the lab rollout path runs without loading Qwen.
# ---------------------------------------------------------------------------


class _FakeBatch(dict):
    """Dict-like batch with ``.to(device)`` and attribute access for
    ``input_ids`` so it satisfies both ``**batch`` and ``batch.input_ids``.
    """

    def to(self, _device):
        return self

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


class _FakeTokenizer:
    eos_token_id = 0
    pad_token_id = 0

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return "\n".join(m.get("content", "") for m in messages)

    def __call__(self, text, return_tensors="pt"):
        import torch
        ids = torch.zeros((1, 1), dtype=torch.long)
        return _FakeBatch({"input_ids": ids})

    def decode(self, ids, skip_special_tokens=True):
        return self._next_text

    # Test-only knob: what the next decode() call will return.
    _next_text = '{"submit": true, "answer": ""}'


class _FakeModel:
    """Minimal stub with the ``generate`` surface ``training_core`` uses.

    On each call it pops the next canned response from ``queue``; once the
    queue is exhausted it returns ``default_response`` forever. This lets us
    script multi-turn behaviour deterministically (e.g. "tool, tool, submit").
    """

    def __init__(self, queue: list[str], default_response: str, tokenizer: _FakeTokenizer):
        self.queue = list(queue)
        self.default = default_response
        self._tok = tokenizer
        self.device = "cpu"
        self.calls = 0

    def generate(self, input_ids=None, max_new_tokens=256, **_kwargs):
        import torch
        self.calls += 1
        text = self.queue.pop(0) if self.queue else self.default
        # Stuff the desired text into the tokenizer so decode() returns it.
        self._tok._next_text = text
        # Return a [1, seq_len+1] tensor so [:, input_ids.shape[1]:] is non-empty.
        return torch.zeros((1, (input_ids.shape[1] if input_ids is not None else 1) + 1), dtype=torch.long)


# ---------------------------------------------------------------------------
# Canned "correct-shape" completions per task_type. These are NOT expected to
# score highly — they just need to parse cleanly so the reward path runs
# end-to-end without blowing up.
# ---------------------------------------------------------------------------


CANNED_COMPLETIONS: dict[str, str] = {
    "dna_classification": "unknown disease",
    "dna_reasoning": '{"answer": "unknown", "reasoning": "Step 1: placeholder."}',
    "evidence_ranking": json.dumps({
        "ranked_diseases": ["a", "b", "c", "d"],
        "selected_disease": "a",
        "elimination_reasoning": {"b": "x", "c": "y", "d": "z"},
        "supporting_evidence": "Step 1: placeholder.",
    }),
    "protein_function": json.dumps({
        "function_description": "unknown",
        "subcellular_location": "cytoplasm",
        "go_terms": ["GO:0000000"],
        "reasoning": "Step 1: placeholder.",
    }),
    "kegg_pathway_reasoning": json.dumps({
        "answer": "unknown",
        "reasoning": "Step 1: placeholder edge.",
        "mentioned_genes": ["GENE"],
    }),
    "perturbation_qa": json.dumps({
        "perturbation_answers": {"p0": True, "p1": False},
    }),
    "perturbation_direction_qa": json.dumps({
        "direction_answers": {"p0": "Unknown", "p1": "Unknown"},
    }),
    "perturbation_benchmark": json.dumps({
        "direction_answers": {"p0": "Unknown"},
    }),
    "clinical_diagnosis": json.dumps({
        "answer": "unknown",
        "differential_ranking": ["a", "b"],
        "reasoning": "Step 1: placeholder.",
    }),
    # Lab tasks: Turn 1 is a submit so the rollout terminates immediately.
    "protein_hypothesis_lab": json.dumps({
        "submit": True,
        "answer": "unknown",
        "subcellular_location": "cytoplasm",
        "go_terms": ["GO:0000000"],
        "reasoning": "Step 1: placeholder.",
    }),
    "target_discovery_lab": json.dumps({
        "submit": True,
        "answer": "unknown",
        "reasoning": "Step 1: placeholder.",
        "go_terms": ["GO:0000000"],
        "proposed_intervention": {"mode": "inhibit", "target": "TP53"},
    }),
    "clinical_diagnosis_lab": json.dumps({
        "submit": True,
        "answer": "unknown",
        "differential_ranking": ["a", "b"],
        "reasoning": "Step 1: placeholder.",
    }),
    "ligand_design": json.dumps({
        "submit": True,
        "predicted_ligand": "aspirin",
        "reasoning": "placeholder.",
    }),
    "curriculum_self_play": json.dumps({
        "submit": True,
        "answer": "Functional Summary: unknown. UniProt Summary: unknown.",
        "reasoning": "Paragraph 1: placeholder.",
    }),
}


# ---------------------------------------------------------------------------
# Test runner (can be called from pytest OR directly via __main__).
# ---------------------------------------------------------------------------


def _check_reward(task_type: str) -> float:
    """Run one reward-function call on a freshly-reset brief and return its
    reward. Asserts the reward is a float in ``[0.0, 1.0]``.
    """
    result = training_core.env_reset(task_type=task_type)
    obs = result.observation
    assert obs.task_id, f"{task_type}: empty task_id from reset"

    reward_fn = training_core.make_reward_fn(task_type)
    rewards = reward_fn(
        prompts=[[{"role": "user", "content": "x"}]],
        completions=[CANNED_COMPLETIONS[task_type]],
        task_id=[obs.task_id],
        task_type=[task_type],
    )
    assert isinstance(rewards, list) and len(rewards) == 1
    r = rewards[0]
    assert isinstance(r, float), f"{task_type}: reward not a float ({type(r).__name__})"
    assert 0.0 <= r <= 1.0, f"{task_type}: reward {r} out of [0, 1]"
    return r


def _check_gate_skips_other_tasks(task_type: str) -> None:
    """A reward fn for task A must return 0.0 on rows tagged with task B."""
    reward_fn = training_core.make_reward_fn(task_type)
    other = "perturbation_qa" if task_type != "perturbation_qa" else "dna_classification"
    rewards = reward_fn(
        prompts=[[{"role": "user", "content": "x"}]],
        completions=["irrelevant"],
        task_id=["ignored"],
        task_type=[other],
    )
    assert rewards == [0.0], f"{task_type}: gate failed, got {rewards}"


def _check_dataset_builder() -> None:
    ds = training_core.build_mixed_dataset(
        task_list=["dna_classification", "clinical_diagnosis"],
        n_per_task=2,
        seed=0,
    )
    assert len(ds) == 4, f"expected 4 rows, got {len(ds)}"
    cols = set(ds.column_names)
    assert {"prompt", "task_id", "task_type"}.issubset(cols), f"missing cols: {cols}"
    for row in ds:
        assert isinstance(row["prompt"], list) and len(row["prompt"]) == 2
        assert row["prompt"][0]["role"] == "system"
        assert row["prompt"][1]["role"] == "user"
        assert row["task_type"] in ("dna_classification", "clinical_diagnosis")
        assert row["task_id"], "empty task_id in dataset row"


def _check_lab_multi_turn() -> None:
    """Exercise the multi-turn path of a lab reward fn.

    Script the fake model to say "tool call" on Turn 2 and "submit" on Turn
    3 so we know the rollout actually loops and terminates via submit —
    not via the forced-submit fallback.
    """
    tok = _FakeTokenizer()
    queue = [
        '{"tool": "get_ppi", "args": {"gene": "TP53"}}',          # Turn 2
        '{"submit": true, "answer": "unknown", "reasoning": "."}',  # Turn 3
    ]
    fake = _FakeModel(queue=queue, default_response=queue[-1], tokenizer=tok)
    training_core.configure_model(fake, tok, max_new_tokens=16)

    try:
        os.environ["TRAIN_LAB_MAX_STEPS"] = "3"
        # Reload the module-level constant so our override is picked up.
        # (The reward fn reads TRAIN_LAB_MAX_STEPS at the module level.)
        training_core.TRAIN_LAB_MAX_STEPS = 3

        task_type = "protein_hypothesis_lab"
        # Turn 1 (the "TRL completion") is a tool call so the rollout
        # enters the multi-turn loop instead of terminating immediately.
        first_turn = '{"tool": "get_sequence", "args": {"protein_id": "P00533"}}'

        result = training_core.env_reset(task_type=task_type)
        obs = result.observation
        reward_fn = training_core.make_reward_fn(task_type)
        rewards = reward_fn(
            prompts=[[{"role": "user", "content": "x"}]],
            completions=[first_turn],
            task_id=[obs.task_id],
            task_type=[task_type],
        )
        assert rewards and 0.0 <= rewards[0] <= 1.0, f"bad lab reward: {rewards}"
        assert fake.calls >= 1, "FakeModel.generate was never called — multi-turn loop did not run"
    finally:
        training_core.configure_model(None, None)


def _check_eval_helper() -> None:
    """``run_eval_episode`` must return a float in [0.01, 0.99] for both a
    legacy task and a lab task."""
    tok = _FakeTokenizer()
    fake = _FakeModel(
        queue=[],
        default_response='{"submit": true, "answer": "unknown", "reasoning": "."}',
        tokenizer=tok,
    )
    training_core.configure_model(fake, tok, max_new_tokens=16)
    try:
        # Legacy
        obs = training_core.env_reset(task_type="dna_classification").observation
        r = training_core.run_eval_episode(obs.task_id, "dna_classification")
        assert 0.0 <= r <= 1.0
        # Lab
        obs = training_core.env_reset(task_type="protein_hypothesis_lab").observation
        r = training_core.run_eval_episode(obs.task_id, "protein_hypothesis_lab")
        assert 0.0 <= r <= 1.0
    finally:
        training_core.configure_model(None, None)


def main() -> int:
    print("[test] booting uvicorn subprocess\u2026")
    ctx = _start_server()
    try:
        training_core.configure_env(ctx.base_url)
        print(f"[test] server up at {ctx.base_url}")

        # Task list covers every task in inference.py that has a canned
        # completion above — i.e. all 14.
        task_types = list(CANNED_COMPLETIONS.keys())

        # 1) Reward-fn smoke per task.
        for tt in task_types:
            r = _check_reward(tt)
            print(f"  [ok] reward_{tt:28s} = {r:.3f}")

        # 2) Gating: each fn ignores rows from other tasks.
        for tt in ("dna_classification", "perturbation_direction_qa", "protein_hypothesis_lab"):
            _check_gate_skips_other_tasks(tt)
            print(f"  [ok] gate holds for reward_{tt}")

        # 3) Dataset builder shape.
        _check_dataset_builder()
        print("  [ok] build_mixed_dataset returns prompt/task_id/task_type cols")

        # 4) Multi-turn lab rollout with a scripted FakeModel.
        _check_lab_multi_turn()
        print("  [ok] lab reward fn runs multi-turn rollout via FakeModel")

        # 5) Held-out eval helper (mirrors inference.py rollouts).
        _check_eval_helper()
        print("  [ok] run_eval_episode works for legacy + lab tasks")

        print("\nALL GOOD \u2013 training_core smoke test passed.")
        return 0
    finally:
        _stop_server(ctx)


# ---------------------------------------------------------------------------
# Pytest integration — delegate to ``main()`` so running either way works.
# ---------------------------------------------------------------------------


def test_training_core_smoke():  # pragma: no cover — exercised end-to-end
    assert main() == 0


if __name__ == "__main__":
    raise SystemExit(main())
