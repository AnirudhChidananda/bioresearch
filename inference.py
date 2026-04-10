"""
Bioresearch Inference Script
===================================
Runs an LLM agent against the Bioresearch OpenEnv environment across
all 4 tasks: dna_classification, dna_reasoning, evidence_ranking,
and protein_function.

MANDATORY ENV VARS:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.

STDOUT FORMAT:
    [START] task=<task_name> env=bioresearch model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""

import asyncio
import json
import os
import re
import textwrap
from typing import Any, Dict, List, Optional

from openai import OpenAI

from bioresearch import BioresearchAction, BioresearchEnv

IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME") or os.getenv("IMAGE_NAME")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
BENCHMARK = "bioresearch"
EPISODES_PER_TASK = int(os.getenv("EPISODES_PER_TASK", "5"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.3"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "1500"))
SUCCESS_THRESHOLD = 0.15

TASK_LIST = ["dna_classification", "dna_reasoning", "evidence_ranking", "protein_function"]

# ── Logging helpers ──────────────────────────────────────────────────────

def log_start(task: str, model: str) -> None:
    print(f"[START] task={task} env={BENCHMARK} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    err = error if error else "null"
    action_short = action[:200].replace("\n", " ") if action else "null"
    print(
        f"[STEP] step={step} action={action_short} reward={reward:.2f} done={str(done).lower()} error={err}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


# ── Prompt builders ──────────────────────────────────────────────────────

SYSTEM_PROMPTS = {
    "dna_classification": textwrap.dedent("""\
        You are a genomics expert. Given a DNA variant and its pathway context,
        identify which disease this mutation contributes to.
        Reply with ONLY the disease name, nothing else."""),

    "dna_reasoning": textwrap.dedent("""\
        You are a genomics expert. Given a DNA variant and its pathway context,
        identify the disease AND explain the biological mechanism step-by-step.
        Reply with valid JSON only:
        {"answer": "<disease name>", "reasoning": "Step 1: ... Step 2: ... Step 3: ..."}"""),

    "evidence_ranking": textwrap.dedent("""\
        You are a genomics expert. Given a DNA variant, pathway context, and 4 candidate diseases,
        rank the candidates from most to least likely. For each rejected disease, explain why
        this pathway does NOT lead to it. For your top pick, provide a step-by-step mechanism.
        Reply with valid JSON only:
        {"ranked_diseases": ["most_likely", ...], "selected_disease": "...",
         "elimination_reasoning": {"rejected_disease": "why eliminated", ...},
         "supporting_evidence": "Step 1: ... Step 2: ..."}"""),

    "protein_function": textwrap.dedent("""\
        You are a protein biologist. Given a protein sequence and metadata,
        predict its biological function, subcellular location, and relevant GO terms.
        Reply with valid JSON only:
        {"function_description": "...", "subcellular_location": "...",
         "go_terms": ["GO:0000000", ...], "reasoning": "..."}"""),
}


def build_user_prompt(obs) -> str:
    """Build the user prompt from an observation."""
    parts = [obs.question]

    if obs.sequence_data:
        if "reference_sequence" in obs.sequence_data:
            ref = obs.sequence_data["reference_sequence"]
            var = obs.sequence_data["variant_sequence"]
            if len(ref) > 300:
                ref = ref[:150] + " [...] " + ref[-150:]
                var = var[:150] + " [...] " + var[-150:]
            parts.append(f"\nReference sequence (truncated): {ref}")
            parts.append(f"Variant sequence (truncated): {var}")
        elif "sequence" in obs.sequence_data:
            seq = obs.sequence_data["sequence"]
            if len(seq) > 300:
                seq = seq[:150] + " [...] " + seq[-150:]
            parts.append(f"\nProtein sequence (truncated): {seq}")

    if obs.candidate_diseases:
        parts.append(f"\nCandidate diseases: {', '.join(obs.candidate_diseases)}")

    return "\n".join(parts)


# ── Response parsing ─────────────────────────────────────────────────────

def _try_parse_json(text: str) -> Optional[Dict[str, Any]]:
    """Attempt to extract JSON from model output."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    return None


def parse_response(task_type: str, task_id: str, text: str) -> BioresearchAction:
    """Parse LLM response into a BioresearchAction."""
    if task_type == "dna_classification":
        return BioresearchAction(task_id=task_id, answer=text.strip())

    parsed = _try_parse_json(text)

    if task_type == "dna_reasoning":
        if parsed:
            return BioresearchAction(
                task_id=task_id,
                answer=parsed.get("answer", text.strip()[:200]),
                reasoning=parsed.get("reasoning", ""),
            )
        return BioresearchAction(task_id=task_id, answer=text.strip()[:200], reasoning=text.strip())

    if task_type == "evidence_ranking":
        if parsed:
            return BioresearchAction(
                task_id=task_id,
                answer=parsed.get("selected_disease", ""),
                reasoning=parsed.get("supporting_evidence", ""),
                ranked_diseases=parsed.get("ranked_diseases"),
                elimination_reasoning=parsed.get("elimination_reasoning"),
            )
        return BioresearchAction(task_id=task_id, answer=text.strip()[:200], reasoning=text.strip())

    if task_type == "protein_function":
        if parsed:
            go_terms = parsed.get("go_terms", [])
            if isinstance(go_terms, str):
                go_terms = [t.strip() for t in go_terms.split(",")]
            return BioresearchAction(
                task_id=task_id,
                answer=parsed.get("function_description", text.strip()[:300]),
                reasoning=parsed.get("reasoning", ""),
                subcellular_location=parsed.get("subcellular_location"),
                go_terms=go_terms if go_terms else None,
            )
        return BioresearchAction(task_id=task_id, answer=text.strip()[:300], reasoning=text.strip())

    return BioresearchAction(task_id=task_id, answer=text.strip())


# ── LLM call ─────────────────────────────────────────────────────────────

def call_llm(client: OpenAI, system_prompt: str, user_prompt: str) -> str:
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        return (completion.choices[0].message.content or "").strip()
    except Exception as exc:
        print(f"[DEBUG] LLM call failed: {exc}", flush=True)
        return ""


# ── Main loop ────────────────────────────────────────────────────────────

async def run_task(env: BioresearchEnv, client: OpenAI, task_type: str) -> List[float]:
    """Run EPISODES_PER_TASK episodes for a single task type."""
    rewards: List[float] = []

    for ep in range(EPISODES_PER_TASK):
        log_start(task=task_type, model=MODEL_NAME)
        step_rewards: List[float] = []
        steps_taken = 0
        score = 0.0
        success = False

        try:
            result = await env.reset(task_type=task_type)
            obs = result.observation
            task_id = obs.task_id

            system_prompt = SYSTEM_PROMPTS[task_type]
            user_prompt = build_user_prompt(obs)

            llm_response = call_llm(client, system_prompt, user_prompt)
            if not llm_response:
                llm_response = "unknown"

            action = parse_response(task_type, task_id, llm_response)
            result = await env.step(action)

            reward = result.reward or 0.0
            reward = max(0.01, min(0.99, reward))
            done = result.done
            steps_taken = 1
            step_rewards.append(reward)

            action_summary = action.answer[:100].replace("\n", " ")
            log_step(step=1, action=action_summary, reward=reward, done=done, error=None)

            score = reward
            success = score >= SUCCESS_THRESHOLD

        except Exception as exc:
            print(f"[DEBUG] Episode error: {exc}", flush=True)
            score = 0.01
            step_rewards = [0.01]
            steps_taken = 1

        finally:
            log_end(success=success, steps=steps_taken, score=score, rewards=step_rewards)
            rewards.append(score)

    return rewards


async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    if IMAGE_NAME:
        env = await BioresearchEnv.from_docker_image(IMAGE_NAME)
    else:
        base_url = os.getenv("ENV_BASE_URL", "http://localhost:8000")
        env = BioresearchEnv(base_url=base_url)

    all_scores: Dict[str, List[float]] = {}
    try:
        for task_type in TASK_LIST:
            task_rewards = await run_task(env, client, task_type)
            all_scores[task_type] = task_rewards

        print("\n" + "=" * 60, flush=True)
        print("BASELINE SCORES SUMMARY", flush=True)
        print("=" * 60, flush=True)
        for task_type, scores in all_scores.items():
            mean = sum(scores) / len(scores) if scores else 0.0
            print(f"  {task_type:25s}  mean={mean:.3f}  scores={[round(s, 3) for s in scores]}", flush=True)
        overall = []
        for s in all_scores.values():
            overall.extend(s)
        if overall:
            print(f"  {'OVERALL':25s}  mean={sum(overall)/len(overall):.3f}", flush=True)
        print("=" * 60, flush=True)

    finally:
        try:
            await env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error: {e}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
