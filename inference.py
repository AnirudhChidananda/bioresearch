"""
Bioresearch Inference Script
===================================

Runs an LLM agent against the Bioresearch OpenEnv environment. Supports
both the legacy single-step tasks and the new long-horizon lab tasks.

Legacy tasks (single-shot):
    dna_classification, dna_reasoning, evidence_ranking, protein_function,
    clinical_diagnosis, perturbation_qa

Lab tasks (long-horizon tool-calling loop, up to 20 steps):
    target_discovery_lab, protein_hypothesis_lab, curriculum_self_play,
    clinical_diagnosis_lab, ligand_design

MANDATORY ENV VARS:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.

Optional:
    TASK_LIST      Comma-separated task list (defaults to legacy + lab).
    EPISODES_PER_TASK  Default 5 legacy / 2 lab.
    MAX_LAB_STEPS  Default 20.

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

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from openai import OpenAI

from bioresearch import BioresearchAction, BioresearchEnv


IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME") or os.getenv("IMAGE_NAME")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
BENCHMARK = "bioresearch"
EPISODES_PER_TASK = int(os.getenv("EPISODES_PER_TASK", "5"))
EPISODES_PER_LAB_TASK = int(os.getenv("EPISODES_PER_LAB_TASK", "2"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.3"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "1500"))
MAX_LAB_STEPS = int(os.getenv("MAX_LAB_STEPS", "20"))
SUCCESS_THRESHOLD = 0.15

LEGACY_TASKS = [
    "dna_classification",
    "dna_reasoning",
    "evidence_ranking",
    "protein_function",
    "clinical_diagnosis",
    "perturbation_qa",
]
LAB_TASKS = [
    "target_discovery_lab",
    "protein_hypothesis_lab",
    "curriculum_self_play",
    "clinical_diagnosis_lab",
    "ligand_design",
]

_env_task_list = os.getenv("TASK_LIST")
if _env_task_list:
    TASK_LIST = [t.strip() for t in _env_task_list.split(",") if t.strip()]
else:
    TASK_LIST = LEGACY_TASKS + LAB_TASKS


# =========================================================================
# Logging helpers
# =========================================================================


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


# =========================================================================
# System prompts
# =========================================================================


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

    # --- Lab tasks ------------------------------------------------------
    "target_discovery_lab": textwrap.dedent("""\
        You are a Principal Investigator running a drug-discovery lab. You can
        either call a TOOL to gather evidence or SUBMIT a final answer.

        Each turn reply with VALID JSON ONLY, matching one of these shapes:
          Tool call: {"tool": "<name>", "args": {"...": ...}}
          Final:     {"submit": true, "answer": "<disease>",
                      "reasoning": "Step 1: ... Step 2: ...",
                      "go_terms": ["GO:0000000", ...],
                      "proposed_intervention": {"mode": "inhibit", "target": "<GENE>"}}

        Tools: get_pathway, get_interpro, get_ppi, get_go (branch=leaf),
        get_sequence, get_subcellular_location, search_catalogue. Call at most
        8 tools, then submit."""),

    "protein_hypothesis_lab": textwrap.dedent("""\
        You are a protein-function specialist. Investigate the target protein
        via tools, then submit a grounded hypothesis.

        Each turn reply with VALID JSON ONLY:
          Tool call: {"tool": "<name>", "args": {"protein_id": "...", ...}}
          Final:     {"submit": true,
                      "answer": "<function description>",
                      "subcellular_location": "...",
                      "go_terms": ["GO:...", ...],
                      "reasoning": "Step 1: ... Step 2: ..."}"""),

    "curriculum_self_play": textwrap.dedent("""\
        You are doing a self-play training pass. Produce a paragraph-per-step
        chain-of-thought, then a structured final summary. You may optionally
        call tools first if allowed at this curriculum level.

        Each turn reply with VALID JSON ONLY:
          Tool call: {"tool": "<name>", "args": {...}}
          Final:     {"submit": true,
                      "answer": "Functional Summary: ... UniProt Summary: ...",
                      "reasoning": "Paragraph 1...\\n\\nParagraph 2...\\n\\n..."}"""),

    # --- v2 tasks ------------------------------------------------------
    "clinical_diagnosis": textwrap.dedent("""\
        You are a radiology attending. Given an imaging description and a
        list of differential candidates, rank them from most to least likely,
        choose a single final diagnosis, and explain your reasoning step by
        step (Step 1 – ..., Step 2 – ..., ...).

        Reply with valid JSON only:
        {"answer": "<final diagnosis>",
         "differential_ranking": ["most_likely", ...],
         "reasoning": "Step 1 – ... Step 2 – ... Step 3 – ..."}"""),

    "perturbation_qa": textwrap.dedent("""\
        You are a CRISPRi perturbation world model. For each pair in the batch
        decide whether knocking down query_gene significantly affects
        target_gene expression in the given cell line.

        Reply with valid JSON only:
        {"perturbation_answers": {"<pair_id>": true/false, ...}}"""),

    "clinical_diagnosis_lab": textwrap.dedent("""\
        You are a radiology attending running a diagnostic lab. You can call
        tools to gather supporting evidence or submit a final answer.

        Each turn reply with VALID JSON ONLY:
          Tool call: {"tool": "<name>", "args": {...}}
          Final:     {"submit": true,
                      "answer": "<final diagnosis>",
                      "differential_ranking": ["most_likely", ...],
                      "reasoning": "Step 1 – ... Step 2 – ..."}

        Tools: search_catalogue(keyword), get_pathway(gene=...), get_go(protein_id, branch=...)."""),

    "ligand_design": textwrap.dedent("""\
        You are a medicinal chemist. Propose a high-pIC50 small molecule for
        the target gene. You can call tools to explore candidates first.

        Each turn reply with VALID JSON ONLY:
          Tool call: {"tool": "<name>", "args": {"gene": "...", "smiles": "..."}}
          Final:     {"submit": true,
                      "predicted_ligand": "<SMILES or drug name>",
                      "reasoning": "short justification"}

        Tools: get_candidate_ligands(gene=..., k=5), get_drug_properties(smiles=...)."""),
}


# =========================================================================
# Prompt builders
# =========================================================================


def build_user_prompt(obs) -> str:
    """Build the user prompt from an observation (legacy tasks)."""
    parts = [obs.question]

    if obs.sequence_data:
        if "reference_sequence" in obs.sequence_data:
            ref = obs.sequence_data["reference_sequence"]
            var = obs.sequence_data.get("variant_sequence", "")
            if len(ref) > 300:
                ref = ref[:150] + " [...] " + ref[-150:]
            if len(var) > 300:
                var = var[:150] + " [...] " + var[-150:]
            parts.append(f"\nReference sequence (truncated): {ref}")
            if var:
                parts.append(f"Variant sequence (truncated): {var}")
        elif "sequence" in obs.sequence_data:
            seq = obs.sequence_data["sequence"]
            if len(seq) > 300:
                seq = seq[:150] + " [...] " + seq[-150:]
            parts.append(f"\nProtein sequence (truncated): {seq}")

    if obs.candidate_diseases:
        parts.append(f"\nCandidate diseases: {', '.join(obs.candidate_diseases)}")

    if obs.differentials:
        parts.append(f"\nDifferential candidates: {', '.join(obs.differentials)}")

    if obs.perturbation_batch:
        parts.append("\nPerturbation batch (answer every pair_id):")
        for pair in obs.perturbation_batch:
            parts.append(
                f"  - pair_id={pair.get('pair_id')}: knocking down "
                f"{pair.get('query_gene')} affects {pair.get('target_gene')} in "
                f"{pair.get('cell_line')}?"
            )

    if obs.ligand_candidates:
        parts.append("\nExisting candidate ligands:")
        for cand in obs.ligand_candidates[:5]:
            parts.append(f"  - {cand}")

    return "\n".join(parts)


def build_lab_prompt(obs, step_idx: int) -> str:
    """Build a rolling user prompt for lab-mode turns.

    Uses the observation's ``notebook`` field as compressed evidence memory.
    Each notebook entry is already cap-truncated by the environment to keep
    the prompt small even at step 20.
    """
    parts = []
    if step_idx == 0:
        parts.append(obs.question)
    else:
        parts.append(
            f"Phase: {obs.phase}. Remaining tool-call budget: {obs.remaining_steps}."
        )
        if obs.tool_result is not None:
            parts.append(f"Last tool result: {json.dumps(obs.tool_result)[:800]}")

    if obs.notebook:
        notebook_preview = []
        for entry in obs.notebook[-10:]:
            step = entry.get("step", "?")
            tool = entry.get("tool", "?")
            args = entry.get("args", {})
            result = entry.get("result", {})
            if isinstance(result, dict) and "error" not in result:
                summary_bits = []
                for k, v in result.items():
                    if isinstance(v, (str, int, float)) and k != "protein_id":
                        summary_bits.append(f"{k}={str(v)[:120]}")
                result_summary = "; ".join(summary_bits[:3])
            else:
                result_summary = json.dumps(result)[:120]
            notebook_preview.append(
                f"  step={step} tool={tool} args={json.dumps(args)[:80]} -> {result_summary}"
            )
        parts.append("Notebook so far:\n" + "\n".join(notebook_preview))

    parts.append("Reply with a SINGLE JSON object (tool call or submit).")
    return "\n\n".join(parts)


# =========================================================================
# Response parsing
# =========================================================================


def _try_parse_json(text: str) -> Optional[Dict[str, Any]]:
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return None


def parse_response(task_type: str, task_id: str, text: str) -> BioresearchAction:
    """Parse LLM response into a BioresearchAction (single-step legacy tasks)."""
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

    if task_type == "clinical_diagnosis":
        if parsed:
            ranking = parsed.get("differential_ranking") or parsed.get("ranked_diseases")
            if isinstance(ranking, str):
                ranking = [r.strip() for r in ranking.split(",") if r.strip()]
            return BioresearchAction(
                task_id=task_id,
                answer=parsed.get("answer", "") or parsed.get("final_diagnosis", "") or "",
                reasoning=parsed.get("reasoning", ""),
                differential_ranking=ranking,
            )
        return BioresearchAction(task_id=task_id, answer=text.strip()[:200], reasoning=text.strip())

    if task_type == "perturbation_qa":
        answers: Dict[str, bool] = {}
        if parsed:
            raw = parsed.get("perturbation_answers") or parsed.get("answers") or {}
            if isinstance(raw, dict):
                for k, v in raw.items():
                    if isinstance(v, bool):
                        answers[str(k)] = v
                    elif isinstance(v, str):
                        answers[str(k)] = v.strip().lower().startswith(("y", "t", "1"))
        return BioresearchAction(task_id=task_id, perturbation_answers=answers or None)

    return BioresearchAction(task_id=task_id, answer=text.strip())


def parse_lab_response(task_id: str, text: str) -> BioresearchAction:
    """Parse a lab-mode LLM response. Returns a tool-call action or a submit action."""
    parsed = _try_parse_json(text)
    if not parsed:
        # Model failed to emit valid JSON — force a submit with raw text as answer.
        return BioresearchAction(task_id=task_id, submit=True, answer=text.strip()[:500])

    if parsed.get("submit"):
        go_terms = parsed.get("go_terms") or None
        if isinstance(go_terms, str):
            go_terms = [t.strip() for t in go_terms.split(",") if t.strip()]
        diff = parsed.get("differential_ranking")
        if isinstance(diff, str):
            diff = [r.strip() for r in diff.split(",") if r.strip()]
        return BioresearchAction(
            task_id=task_id,
            submit=True,
            answer=parsed.get("answer", "") or "",
            reasoning=parsed.get("reasoning") or None,
            go_terms=go_terms,
            subcellular_location=parsed.get("subcellular_location") or None,
            proposed_intervention=parsed.get("proposed_intervention") or None,
            predicted_ligand=parsed.get("predicted_ligand") or None,
            differential_ranking=diff,
        )

    tool_name = parsed.get("tool") or parsed.get("tool_name")
    tool_args = parsed.get("args") or parsed.get("tool_args") or {}
    return BioresearchAction(
        task_id=task_id,
        tool_name=tool_name,
        tool_args=tool_args,
    )


# =========================================================================
# LLM call
# =========================================================================


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


# =========================================================================
# Per-episode runners
# =========================================================================


async def _run_legacy_episode(env, client: OpenAI, task_type: str) -> float:
    """One single-step legacy episode. Returns the terminal reward."""
    result = await env.reset(task_type=task_type)
    obs = result.observation
    task_id = obs.task_id

    system_prompt = SYSTEM_PROMPTS[task_type]
    user_prompt = build_user_prompt(obs)
    llm_response = call_llm(client, system_prompt, user_prompt) or "unknown"

    action = parse_response(task_type, task_id, llm_response)
    result = await env.step(action)

    reward = max(0.01, min(0.99, result.reward or 0.0))
    action_summary = action.answer[:100].replace("\n", " ")
    log_step(step=1, action=action_summary, reward=reward, done=result.done, error=None)
    return reward


async def _run_lab_episode(env, client: OpenAI, task_type: str) -> float:
    """One long-horizon lab episode. Returns the terminal reward."""
    result = await env.reset(task_type=task_type)
    obs = result.observation
    task_id = obs.task_id

    system_prompt = SYSTEM_PROMPTS[task_type]

    step_idx = 0
    step_rewards: List[float] = []
    final_reward = 0.01

    while step_idx < MAX_LAB_STEPS:
        user_prompt = build_lab_prompt(obs, step_idx)
        llm_response = call_llm(client, system_prompt, user_prompt)
        if not llm_response:
            action = BioresearchAction(task_id=task_id, submit=True, answer="")
        else:
            action = parse_lab_response(task_id, llm_response)

        result = await env.step(action)
        obs = result.observation
        step_idx += 1
        r = result.reward or 0.0
        step_rewards.append(r)

        desc = (
            f"submit answer={action.answer[:60]!r}" if action.submit
            else f"tool={action.tool_name} args={json.dumps(action.tool_args or {})[:80]}"
        )
        log_step(step=step_idx, action=desc, reward=r, done=result.done, error=None)

        if result.done:
            final_reward = max(0.01, min(0.99, r))
            break

    if not result.done:
        # Force a final submit if the model never did.
        action = BioresearchAction(task_id=task_id, submit=True, answer=action.answer or "")
        result = await env.step(action)
        step_idx += 1
        final_reward = max(0.01, min(0.99, result.reward or 0.0))
        log_step(step=step_idx, action="forced-submit", reward=final_reward, done=True, error=None)

    return final_reward


async def run_task(env, client: OpenAI, task_type: str) -> List[float]:
    is_lab = task_type in LAB_TASKS
    episodes = EPISODES_PER_LAB_TASK if is_lab else EPISODES_PER_TASK
    rewards: List[float] = []

    for _ in range(episodes):
        log_start(task=task_type, model=MODEL_NAME)
        try:
            if is_lab:
                score = await _run_lab_episode(env, client, task_type)
            else:
                score = await _run_legacy_episode(env, client, task_type)
        except Exception as exc:
            print(f"[DEBUG] Episode error: {exc}", flush=True)
            score = 0.01

        success = score >= SUCCESS_THRESHOLD
        log_end(success=success, steps=1 if not is_lab else MAX_LAB_STEPS, score=score, rewards=[score])
        rewards.append(score)

    return rewards


# =========================================================================
# Main
# =========================================================================


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
        overall = [s for scores in all_scores.values() for s in scores]
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
