"""
Bioresearch evaluation harness.

Runs N episodes on each task against the environment using a pluggable
policy, prints a summary table, and writes a JSON report to disk.

Supported policies:
    --policy random        Picks random answers from the disease vocabulary.
    --policy gold          Submits the ground-truth answer (ceiling).
    --policy heuristic     Simple heuristic — returns the disease name closest
                           to the question text by token overlap.
    --policy openai        Uses the OpenAI / HuggingFace inference router
                           (requires HF_TOKEN or API_KEY env var).

Usage:
    python evaluate.py --policy heuristic --episodes 10
    python evaluate.py --policy openai --model Qwen/Qwen2.5-72B-Instruct
    python evaluate.py --policy gold --tasks virtual_tumor_board --episodes 5
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

from models import BioresearchAction  # noqa: E402
from server.bioresearch_environment import BioresearchEnvironment  # noqa: E402
from server.data_loader import DataLoader  # noqa: E402

ALL_TASKS = [
    "dna_classification",
    "dna_reasoning",
    "evidence_ranking",
    "protein_function",
    "virtual_tumor_board",
]


# ═══════════════════════════════════════════════════════════════════════════
# Policies
# ═══════════════════════════════════════════════════════════════════════════

Policy = Callable[[Dict[str, Any]], BioresearchAction]


def _random_policy(data: DataLoader, rng: random.Random) -> Policy:
    diseases = data.all_disease_answers

    def policy(ctx: Dict[str, Any]) -> BioresearchAction:
        task_type = ctx["observation"].task_type
        task_id = ctx["observation"].task_id
        answer = rng.choice(diseases)
        if task_type == "protein_function":
            return BioresearchAction(
                task_id=task_id, answer="unknown protein function",
                go_terms=[], subcellular_location="unknown",
            )
        if task_type == "virtual_tumor_board":
            turn = ctx["observation"].turn_count
            if turn >= 3:
                return BioresearchAction(
                    task_id=task_id, tool_name="submit_consensus",
                    tool_args={"answer": answer, "reasoning": "random guess"},
                )
            return BioresearchAction(
                task_id=task_id, tool_name="blast_lookup", tool_args={},
            )
        return BioresearchAction(task_id=task_id, answer=answer)

    return policy


def _gold_policy(data: DataLoader) -> Policy:
    """Submits the gold answer — an achievability ceiling for each task."""

    def policy(ctx: Dict[str, Any]) -> BioresearchAction:
        obs = ctx["observation"]
        task_id = obs.task_id
        task_type = obs.task_type

        if task_id.startswith("dna_"):
            sample = data.get_dna_sample_by_id(task_id)
            gold_answer = sample.answer
            gold_reasoning = sample.reasoning
        else:
            sample = data.get_protein_sample_by_id(task_id)
            gold_answer = sample.protein_function
            gold_reasoning = sample.protein_function

        if task_type == "protein_function":
            return BioresearchAction(
                task_id=task_id,
                answer=sample.protein_function,
                reasoning=f"Domain analysis: {sample.interpro_formatted[:200]}",
                go_terms=sample.go_ids[:10] if sample.go_ids else None,
                subcellular_location=sample.subcellular_location,
            )

        if task_type == "evidence_ranking":
            candidates = obs.candidate_diseases or [gold_answer]
            ranking = sorted(candidates, key=lambda c: 0 if c.lower() == gold_answer.lower() else 1)
            distractors = [c for c in candidates if c.lower() != gold_answer.lower()]
            elim = {d: f"The pathway does not implicate {d}; mutation mechanism is different." for d in distractors}
            return BioresearchAction(
                task_id=task_id, answer=gold_answer, reasoning=gold_reasoning,
                ranked_diseases=ranking, elimination_reasoning=elim,
            )

        if task_type == "virtual_tumor_board":
            turn = obs.turn_count
            rollout_plan = [
                ("ask_specialist", {"role": "geneticist", "question": "variant?"}),
                ("ask_specialist", {"role": "pathway_analyst", "question": "pathway?"}),
                ("ask_specialist", {"role": "clinician", "question": "phenotype?"}),
                ("literature_snippet", {"disease": gold_answer}),
            ]
            if turn < len(rollout_plan):
                tool_name, tool_args = rollout_plan[turn]
                return BioresearchAction(task_id=task_id, tool_name=tool_name, tool_args=tool_args)
            return BioresearchAction(
                task_id=task_id, tool_name="submit_consensus",
                tool_args={
                    "answer": gold_answer,
                    "reasoning": (
                        "Based on the geneticist, pathway_analyst and clinician inputs, "
                        f"the variant causes {gold_answer}. "
                        f"Mechanism: {gold_reasoning[:400]}"
                    ),
                },
            )

        return BioresearchAction(task_id=task_id, answer=gold_answer, reasoning=gold_reasoning)

    return policy


def _heuristic_policy(data: DataLoader) -> Policy:
    """Pick the disease whose name best overlaps with the question text."""
    diseases = data.all_disease_answers

    def _score(question: str, disease: str) -> float:
        q_tokens = set(re.findall(r"\w+", question.lower()))
        d_tokens = set(re.findall(r"\w+", disease.lower()))
        if not q_tokens or not d_tokens:
            return 0.0
        return len(q_tokens & d_tokens) / len(d_tokens)

    def policy(ctx: Dict[str, Any]) -> BioresearchAction:
        obs = ctx["observation"]
        task_id = obs.task_id
        task_type = obs.task_type

        if task_type == "protein_function":
            return BioresearchAction(
                task_id=task_id,
                answer="putative enzymatic / binding protein, likely membrane-associated",
                reasoning="Inferred from domain annotations.",
                go_terms=None,
                subcellular_location="membrane",
            )

        question = obs.question or ""
        candidates = obs.candidate_diseases or diseases
        best = max(candidates, key=lambda d: _score(question, d))

        if task_type == "virtual_tumor_board":
            turn = obs.turn_count
            if turn == 0:
                return BioresearchAction(task_id=task_id, tool_name="ask_specialist",
                                         tool_args={"role": "geneticist", "question": "variant?"})
            if turn == 1:
                return BioresearchAction(task_id=task_id, tool_name="ask_specialist",
                                         tool_args={"role": "pathway_analyst", "question": "pathway?"})
            if turn == 2:
                return BioresearchAction(task_id=task_id, tool_name="ask_specialist",
                                         tool_args={"role": "clinician", "question": "phenotype?"})
            return BioresearchAction(
                task_id=task_id, tool_name="submit_consensus",
                tool_args={
                    "answer": best,
                    "reasoning": f"Heuristic consensus from specialists. Likely diagnosis: {best}.",
                },
            )

        if task_type == "evidence_ranking":
            ranking = sorted(candidates, key=lambda c: -_score(question, c))
            elim = {d: "Weak token overlap with the case pathway." for d in ranking[1:]}
            return BioresearchAction(
                task_id=task_id, answer=ranking[0],
                reasoning="Selected best overlap with case text.",
                ranked_diseases=ranking, elimination_reasoning=elim,
            )

        if task_type == "dna_reasoning":
            return BioresearchAction(
                task_id=task_id, answer=best,
                reasoning=(
                    "Step 1: The variant alters gene function in the pathway. "
                    "Step 2: This activates downstream signaling. "
                    "Step 3: Which leads to the disease phenotype."
                ),
            )

        return BioresearchAction(task_id=task_id, answer=best)

    return policy


def _openai_policy(model: str, base_url: str, api_key: str, temperature: float = 0.3) -> Policy:
    """Use an OpenAI-compatible LLM to produce answers."""
    from openai import OpenAI  # local import
    client = OpenAI(base_url=base_url, api_key=api_key)

    def ask(system: str, user: str) -> str:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=temperature,
            max_tokens=800,
        )
        return (resp.choices[0].message.content or "").strip()

    def parse_json(text: str) -> Dict[str, Any]:
        m = re.search(r"\{[\s\S]*\}", text)
        if not m:
            return {}
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            return {}

    def policy(ctx: Dict[str, Any]) -> BioresearchAction:
        obs = ctx["observation"]
        task_id = obs.task_id
        task_type = obs.task_type

        if task_type == "dna_classification":
            system = "You are a genomics expert. Reply with ONLY the disease name."
            answer = ask(system, obs.question).strip().strip(".")
            return BioresearchAction(task_id=task_id, answer=answer)

        if task_type == "dna_reasoning":
            system = 'Reply with JSON only: {"answer": "<disease>", "reasoning": "Step 1:... Step 2:..."}'
            out = parse_json(ask(system, obs.question))
            return BioresearchAction(
                task_id=task_id, answer=out.get("answer", ""),
                reasoning=out.get("reasoning"),
            )

        if task_type == "evidence_ranking":
            system = (
                'Reply with JSON: {"answer": "<top>", "ranked_diseases": [...4...], '
                '"elimination_reasoning": {"<disease>": "<why eliminated>"}, '
                '"supporting_evidence": "<synthesis>"}'
            )
            cand_str = "\n".join(f"- {c}" for c in (obs.candidate_diseases or []))
            user = f"{obs.question}\n\nCandidates:\n{cand_str}"
            out = parse_json(ask(system, user))
            return BioresearchAction(
                task_id=task_id, answer=out.get("answer", ""),
                reasoning=out.get("supporting_evidence"),
                ranked_diseases=out.get("ranked_diseases"),
                elimination_reasoning=out.get("elimination_reasoning"),
            )

        if task_type == "protein_function":
            system = (
                'Reply with JSON: {"answer": "<function>", "reasoning": "<domain analysis>", '
                '"go_terms": ["GO:..."], "subcellular_location": "<location>"}'
            )
            out = parse_json(ask(system, obs.question))
            return BioresearchAction(
                task_id=task_id, answer=out.get("answer", ""),
                reasoning=out.get("reasoning"),
                go_terms=out.get("go_terms"),
                subcellular_location=out.get("subcellular_location"),
            )

        if task_type == "virtual_tumor_board":
            turn = obs.turn_count
            if turn < 3:
                roles = ["geneticist", "pathway_analyst", "clinician"]
                return BioresearchAction(
                    task_id=task_id, tool_name="ask_specialist",
                    tool_args={"role": roles[turn], "question": f"Your view on case {task_id}?"},
                )
            system = (
                'You are an orchestrator. Given specialist inputs, reply with JSON: '
                '{"answer": "<disease>", "reasoning": "<synthesis>"}'
            )
            history = "\n".join(
                f"Turn {h['turn']} — {h['tool']}: {h.get('output_preview', '')[:200]}"
                for h in (obs.history_summary or [])
            )
            cand_str = "\n".join(f"- {c}" for c in (obs.candidate_diseases or []))
            user = f"Case history:\n{history}\n\nCandidates:\n{cand_str}\n\nYour final diagnosis:"
            out = parse_json(ask(system, user))
            return BioresearchAction(
                task_id=task_id, tool_name="submit_consensus",
                tool_args={
                    "answer": out.get("answer", ""),
                    "reasoning": out.get("reasoning", ""),
                },
            )

        return BioresearchAction(task_id=task_id, answer="unknown")

    return policy


# ═══════════════════════════════════════════════════════════════════════════
# Runner
# ═══════════════════════════════════════════════════════════════════════════

def run_episode(env: BioresearchEnvironment, policy: Policy, task_type: str, task_id: Optional[str] = None) -> Dict[str, Any]:
    reset_kwargs: Dict[str, Any] = {"task_type": task_type}
    if task_id:
        reset_kwargs["task_id"] = task_id

    obs = env.reset(**reset_kwargs)
    start = time.time()
    turns = 0
    while not obs.done:
        action = policy({"observation": obs})
        obs = env.step(action)
        turns += 1
        if turns > 20:  # safety cap
            break

    return {
        "task_type": task_type,
        "task_id": obs.task_id,
        "reward": obs.reward,
        "turns": turns,
        "elapsed_s": round(time.time() - start, 3),
        "score_breakdown": (obs.metadata or {}).get("score_breakdown"),
    }


def summarise(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_task: Dict[str, List[float]] = {}
    for r in results:
        by_task.setdefault(r["task_type"], []).append(float(r["reward"] or 0.0))

    summary = {}
    for task, rewards in by_task.items():
        summary[task] = {
            "n": len(rewards),
            "mean": round(sum(rewards) / len(rewards), 4) if rewards else 0.0,
            "min": round(min(rewards), 4) if rewards else 0.0,
            "max": round(max(rewards), 4) if rewards else 0.0,
        }
    all_rewards = [float(r["reward"] or 0.0) for r in results]
    summary["overall"] = {
        "n": len(all_rewards),
        "mean": round(sum(all_rewards) / len(all_rewards), 4) if all_rewards else 0.0,
    }
    return summary


def print_table(summary: Dict[str, Any]) -> None:
    print()
    print(f"{'Task':<25} {'N':>4} {'Mean':>8} {'Min':>8} {'Max':>8}")
    print("-" * 60)
    for task, stats in summary.items():
        if task == "overall":
            continue
        print(f"{task:<25} {stats['n']:>4} {stats['mean']:>8.4f} {stats['min']:>8.4f} {stats['max']:>8.4f}")
    print("-" * 60)
    print(f"{'OVERALL':<25} {summary['overall']['n']:>4} {summary['overall']['mean']:>8.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a policy on the Bioresearch environment.")
    parser.add_argument("--policy", choices=["random", "gold", "heuristic", "openai"], default="heuristic")
    parser.add_argument("--tasks", nargs="+", default=ALL_TASKS, choices=ALL_TASKS)
    parser.add_argument("--episodes", type=int, default=5, help="episodes per task")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="eval_results.json")
    parser.add_argument("--baseline-only", action="store_true", help="use the held-out baseline split of task_ids")
    parser.add_argument("--model", default=os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct"))
    parser.add_argument("--base-url", default=os.getenv("API_BASE_URL", "https://router.huggingface.co/v1"))
    parser.add_argument("--api-key", default=os.getenv("HF_TOKEN") or os.getenv("API_KEY"))
    parser.add_argument("--temperature", type=float, default=0.3)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    data = DataLoader()

    if args.policy == "random":
        policy = _random_policy(data, rng)
    elif args.policy == "gold":
        policy = _gold_policy(data)
    elif args.policy == "heuristic":
        policy = _heuristic_policy(data)
    elif args.policy == "openai":
        if not args.api_key:
            raise SystemExit("ERROR: --policy openai requires --api-key or HF_TOKEN env var.")
        policy = _openai_policy(args.model, args.base_url, args.api_key, args.temperature)
    else:
        raise ValueError(f"Unknown policy: {args.policy}")

    env = BioresearchEnvironment()
    results: List[Dict[str, Any]] = []

    print(f"Running {args.policy} policy, {args.episodes} episodes per task ({args.tasks})")
    print()

    for task in args.tasks:
        ids_source = data.get_all_sample_ids(
            "dna_classification" if task in ("dna_classification", "dna_reasoning", "evidence_ranking", "virtual_tumor_board") else task,
            baseline_only=args.baseline_only,
        )
        rng.shuffle(ids_source)
        ids = ids_source[: args.episodes]

        for tid in ids:
            try:
                r = run_episode(env, policy, task, task_id=tid)
            except Exception as exc:
                r = {"task_type": task, "task_id": tid, "reward": 0.0, "error": str(exc), "turns": 0, "elapsed_s": 0.0}
            results.append(r)
            print(f"  {task:<22} {tid:<12} reward={r.get('reward', 0.0):.4f} turns={r['turns']}")

    summary = summarise(results)
    print_table(summary)

    output_path = Path(args.output)
    output_path.write_text(json.dumps({
        "policy": args.policy,
        "model": args.model if args.policy == "openai" else None,
        "summary": summary,
        "results": results,
    }, indent=2, default=str))
    print(f"\nWrote detailed results to {output_path}")


if __name__ == "__main__":
    main()
