"""
Bioresearch Gradio Playground

Interactive UI for testing the Bioresearch OpenEnv environment.
Start the OpenEnv server first, then run this script:

    uvicorn server.app:app --host 0.0.0.0 --port 8000
    python playground.py
"""

import json
import os

import gradio as gr

from server.data_loader import DataLoader

data_loader = DataLoader()

# Direct environment access for the playground (no server needed for grading)
from server.bioresearch_environment import BioresearchEnvironment
from models import BioresearchAction

env = BioresearchEnvironment()

TASK_TYPES = ["dna_classification", "dna_reasoning", "evidence_ranking", "protein_function"]

# ── State ────────────────────────────────────────────────────────────────

current_obs = {"task_id": "", "task_type": "", "question": "", "context": {}, "candidate_diseases": None}


def reset_env(task_type):
    obs = env.reset(task_type=task_type)
    current_obs["task_id"] = obs.task_id
    current_obs["task_type"] = obs.task_type
    current_obs["question"] = obs.question
    current_obs["context"] = obs.context
    current_obs["candidate_diseases"] = obs.candidate_diseases

    seq_display = ""
    if obs.sequence_data:
        for k, v in obs.sequence_data.items():
            display_val = v[:200] + "..." if len(v) > 200 else v
            seq_display += f"**{k}**: `{display_val}`\n\n"

    candidates_display = ""
    if obs.candidate_diseases:
        candidates_display = "**Candidate diseases**: " + ", ".join(obs.candidate_diseases)

    info = f"**Task ID**: {obs.task_id}\n**Task Type**: {obs.task_type}"

    return (
        obs.question,
        seq_display,
        candidates_display,
        info,
        "",  # clear reward
        "",  # clear breakdown
    )


def submit_action(answer, reasoning, go_terms_str, subcellular_location, ranked_str, elim_json_str):
    task_id = current_obs["task_id"]
    task_type = current_obs["task_type"]

    if not task_id:
        return "**Error**: No active episode. Click Reset first.", ""

    go_terms = None
    if go_terms_str and go_terms_str.strip():
        go_terms = [t.strip() for t in go_terms_str.split(",") if t.strip()]

    ranked_diseases = None
    if ranked_str and ranked_str.strip():
        ranked_diseases = [d.strip() for d in ranked_str.split(",") if d.strip()]

    elimination_reasoning = None
    if elim_json_str and elim_json_str.strip():
        try:
            elimination_reasoning = json.loads(elim_json_str)
        except json.JSONDecodeError:
            pass

    action = BioresearchAction(
        task_id=task_id,
        answer=answer or "",
        reasoning=reasoning if reasoning else None,
        go_terms=go_terms,
        subcellular_location=subcellular_location if subcellular_location else None,
        ranked_diseases=ranked_diseases,
        elimination_reasoning=elimination_reasoning,
    )

    obs = env.step(action)
    reward = obs.reward or 0.0
    breakdown = obs.metadata.get("score_breakdown", {}) if obs.metadata else {}

    if reward >= 0.7:
        color = "🟢"
    elif reward >= 0.3:
        color = "🟡"
    else:
        color = "🔴"

    reward_display = f"## {color} Reward: {reward:.4f}"
    breakdown_display = json.dumps(breakdown, indent=2, default=str)

    return reward_display, breakdown_display


# ── Dataset Explorer ─────────────────────────────────────────────────────

def browse_dna_sample(idx):
    idx = int(idx)
    task_id = f"dna_{idx:03d}"
    try:
        sample = data_loader.get_dna_sample_by_id(task_id)
    except KeyError:
        return "Invalid index", "", "", ""
    return (
        sample.question[:2000],
        sample.answer,
        sample.reasoning[:2000],
        f"Ref seq length: {len(sample.reference_sequence)} | Var seq length: {len(sample.variant_sequence)}",
    )


def browse_protein_sample(idx):
    idx = int(idx)
    task_id = f"protein_{idx:03d}"
    try:
        sample = data_loader.get_protein_sample_by_id(task_id)
    except KeyError:
        return "Invalid index", "", "", "", ""
    go_str = ", ".join(sample.go_ids[:10])
    return (
        f"**{sample.protein_names}** ({sample.organism})",
        sample.protein_function[:1000],
        sample.subcellular_location,
        go_str,
        f"Sequence length: {int(sample.length)} aa | InterPro: {sample.interpro_formatted[:500]}",
    )


# ── GRPO Reward Analysis ────────────────────────────────────────────────

def analyse_grpo(task_type, sample_idx, response1, response2, response3):
    """Submit 3 different responses to the same problem and compare scores."""
    task_id = f"dna_{int(sample_idx):03d}" if task_type != "protein_function" else f"protein_{int(sample_idx):03d}"

    scores = []
    breakdowns = []

    for resp_text in [response1, response2, response3]:
        if not resp_text or not resp_text.strip():
            scores.append(0.0)
            breakdowns.append({})
            continue

        env.reset(task_type=task_type, task_id=task_id)

        action = BioresearchAction(
            task_id=task_id,
            answer=resp_text.strip(),
            reasoning=resp_text.strip() if task_type != "dna_classification" else None,
        )
        obs = env.step(action)
        s = obs.reward or 0.01
        scores.append(s)
        bd = obs.metadata.get("score_breakdown", {}) if obs.metadata else {}
        breakdowns.append(bd)

    if len([s for s in scores if s > 0]) >= 2:
        valid = [s for s in scores if s > 0]
        mean = sum(valid) / len(valid)
        std = (sum((s - mean) ** 2 for s in valid) / len(valid)) ** 0.5
        advantages = [(s - mean) / std if std > 0 else 0.0 for s in scores]
    else:
        advantages = [0.0] * 3

    result = "## GRPO Reward Analysis\n\n"
    result += f"| Response | Score | Advantage |\n|----------|-------|----------|\n"
    for i, (s, a) in enumerate(zip(scores, advantages)):
        result += f"| Response {i+1} | {s:.4f} | {a:+.4f} |\n"

    result += f"\n**Score spread**: {max(scores) - min(scores):.4f}\n"
    result += f"**Mean**: {sum(scores)/len(scores):.4f}\n\n"
    result += "### Breakdowns\n\n"
    for i, bd in enumerate(breakdowns):
        result += f"**Response {i+1}**: ```{json.dumps(bd, indent=1, default=str)[:500]}```\n\n"

    return result


# ── Build UI ─────────────────────────────────────────────────────────────

with gr.Blocks(title="Bioresearch Playground", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🧬 Bioresearch OpenEnv Playground")
    gr.Markdown("Interactive testing UI for the biological reasoning environment.")

    with gr.Tabs():
        # Tab 1: Interactive Environment
        with gr.TabItem("Interactive Environment"):
            with gr.Row():
                task_dropdown = gr.Dropdown(choices=TASK_TYPES, value="dna_classification", label="Task Type")
                reset_btn = gr.Button("🔄 Reset", variant="primary")

            with gr.Row():
                with gr.Column(scale=2):
                    question_display = gr.Textbox(label="Question", lines=8, interactive=False)
                    sequence_display = gr.Markdown(label="Sequence Data")
                    candidates_display = gr.Markdown(label="Candidate Diseases")
                    info_display = gr.Markdown(label="Episode Info")
                with gr.Column(scale=1):
                    answer_input = gr.Textbox(label="Answer (disease name or function)", lines=2)
                    reasoning_input = gr.Textbox(label="Reasoning (step-by-step)", lines=5)
                    go_terms_input = gr.Textbox(label="GO Terms (comma-separated, T3 only)", lines=1)
                    location_input = gr.Textbox(label="Subcellular Location (T3 only)", lines=1)
                    ranked_input = gr.Textbox(label="Ranked Diseases (comma-separated, T4 only)", lines=1)
                    elim_input = gr.Textbox(label="Elimination Reasoning (JSON dict, T4 only)", lines=3)
                    submit_btn = gr.Button("Submit Action", variant="primary")

            reward_display = gr.Markdown(label="Reward")
            breakdown_display = gr.Code(label="Score Breakdown", language="json")

            reset_btn.click(
                reset_env, inputs=[task_dropdown],
                outputs=[question_display, sequence_display, candidates_display, info_display, reward_display, breakdown_display],
            )
            submit_btn.click(
                submit_action,
                inputs=[answer_input, reasoning_input, go_terms_input, location_input, ranked_input, elim_input],
                outputs=[reward_display, breakdown_display],
            )

        # Tab 2: Dataset Explorer
        with gr.TabItem("Dataset Explorer"):
            gr.Markdown("### DNA Reasoning Samples")
            with gr.Row():
                dna_idx = gr.Slider(0, data_loader.dna_count - 1, step=1, value=0, label="Sample Index")
                dna_browse_btn = gr.Button("Browse")
            dna_question = gr.Textbox(label="Question", lines=5, interactive=False)
            dna_answer = gr.Textbox(label="Gold Answer", interactive=False)
            dna_reasoning = gr.Textbox(label="Gold Reasoning", lines=5, interactive=False)
            dna_meta = gr.Textbox(label="Metadata", interactive=False)
            dna_browse_btn.click(
                browse_dna_sample, inputs=[dna_idx],
                outputs=[dna_question, dna_answer, dna_reasoning, dna_meta],
            )

            gr.Markdown("---\n### Protein Function Samples")
            with gr.Row():
                prot_idx = gr.Slider(0, data_loader.protein_count - 1, step=1, value=0, label="Sample Index")
                prot_browse_btn = gr.Button("Browse")
            prot_name = gr.Markdown(label="Protein Name")
            prot_function = gr.Textbox(label="Function", lines=3, interactive=False)
            prot_location = gr.Textbox(label="Subcellular Location", interactive=False)
            prot_go = gr.Textbox(label="GO IDs", interactive=False)
            prot_meta = gr.Textbox(label="Metadata", interactive=False)
            prot_browse_btn.click(
                browse_protein_sample, inputs=[prot_idx],
                outputs=[prot_name, prot_function, prot_location, prot_go, prot_meta],
            )

        # Tab 3: GRPO Reward Analysis
        with gr.TabItem("GRPO Reward Analysis"):
            gr.Markdown(
                "### Compare multiple responses to the same problem\n"
                "Submit 3 different answers for the same sample and see how GRPO would score them."
            )
            with gr.Row():
                grpo_task = gr.Dropdown(choices=TASK_TYPES, value="dna_classification", label="Task Type")
                grpo_idx = gr.Number(value=0, label="Sample Index", precision=0)
            grpo_r1 = gr.Textbox(label="Response 1", lines=3, placeholder="e.g. cushing syndrome")
            grpo_r2 = gr.Textbox(label="Response 2", lines=3, placeholder="e.g. parkinson's disease")
            grpo_r3 = gr.Textbox(label="Response 3", lines=3, placeholder="e.g. unknown disease")
            grpo_btn = gr.Button("Analyse", variant="primary")
            grpo_output = gr.Markdown(label="Analysis")
            grpo_btn.click(
                analyse_grpo,
                inputs=[grpo_task, grpo_idx, grpo_r1, grpo_r2, grpo_r3],
                outputs=[grpo_output],
            )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
