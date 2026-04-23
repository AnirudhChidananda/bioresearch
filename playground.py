"""
Bioresearch Gradio Playground

Interactive UI for testing the Bioresearch OpenEnv environment.
Runs directly against the environment (no server needed).

    python playground.py
"""

import json

import gradio as gr

from server.data_loader import DataLoader
from server.bioresearch_environment import BioresearchEnvironment
from models import BioresearchAction

data_loader = DataLoader()
env = BioresearchEnvironment()
lab_env = BioresearchEnvironment()  # separate env for the Lab Mode tab

# Dropdowns and label dicts all follow the canonical narrative order
# (Scene 1 variant reasoning → Scene 5 long-horizon labs). Source of truth
# lives in server/bioresearch_environment.py; we keep this list inlined so
# the playground still launches cleanly even if the server module fails
# to import (e.g. missing optional dep in a dev shell).
TASK_TYPES = [
    # Scene 1 — Variant reasoning
    "dna_classification",
    "dna_reasoning",
    "evidence_ranking",
    # Scene 2 — Protein function
    "protein_function",
    # Scene 3 — Systems biology
    "kegg_pathway_reasoning",
    "perturbation_qa",
    "perturbation_direction_qa",
    "perturbation_benchmark",
    # Scene 4 — Clinical
    "clinical_diagnosis",
]
LAB_TASK_TYPES = [
    # Scene 5 — Long-horizon labs
    "protein_hypothesis_lab",
    "target_discovery_lab",
    "clinical_diagnosis_lab",
    "ligand_design",
    "curriculum_self_play",
]
LAB_TOOLS = [
    "get_pathway", "get_interpro", "get_ppi", "get_go",
    "get_sequence", "get_subcellular_location", "search_catalogue",
    "get_drug_properties", "get_candidate_ligands", "get_structure",
]

# Scene-based labels avoid hardcoded "Task N" numbering so the UI survives
# future reorders without touching every label.
TASK_LABELS = {
    "dna_classification": "Scene 1 · DNA Mutation Disease Classification (Easy)",
    "dna_reasoning": "Scene 1 · DNA Mutation Biological Reasoning (Medium)",
    "evidence_ranking": "Scene 1 · Variant Pathogenicity Evidence Ranking (Medium-Hard)",
    "protein_function": "Scene 2 · Protein Function Hypothesis Generation (Hard)",
    "kegg_pathway_reasoning": "Scene 3 · KEGG Pathway-Graph Reasoning (Hard)",
    "perturbation_qa": "Scene 3 · CRISPRi Perturbation World-Modeling (Hard)",
    "perturbation_direction_qa": "Scene 3 · Directional CRISPRi World-Modeling (Hard)",
    "perturbation_benchmark": "Scene 3 · Perturbation Benchmark Umbrella (Very-Hard)",
    "clinical_diagnosis": "Scene 4 · Clinical Differential Diagnosis (Medium-Hard)",
}

TASK_DESCRIPTIONS = {
    "dna_classification": "Identify the disease caused by a DNA mutation. Reply with **only the disease name**.",
    "dna_reasoning": "Identify the disease **and** explain the biological mechanism step-by-step.",
    "evidence_ranking": "Rank 4 candidate diseases. Eliminate wrong ones with reasoning. Support your top pick.",
    "protein_function": "Predict protein function, subcellular location, and GO terms from sequence data.",
    "kegg_pathway_reasoning": "Identify the disease from a KEGG declarative pathway graph. Quote edges in your reasoning and list the genes you cite from the pathway.",
    "perturbation_qa": "Answer a batch of CRISPRi pairs: does knocking down X change Y's expression in cell line Z? JSON: pair_id -> true/false.",
    "perturbation_direction_qa": "3-class directional CRISPRi prediction. For every pair_id, provide 'Increase' | 'Decrease' | 'Unknown' in the JSON input below.",
    "perturbation_benchmark": "Umbrella CRISPRi benchmark across 4 variants (pert_dir, pert_de, gse_pert, gse_gene). Provide directional answers for every pair_id in the JSON.",
    "clinical_diagnosis": "Read the imaging description, rank the differentials, and commit to a final diagnosis with Step-by-Step reasoning.",
}

ANSWER_LABELS = {
    "dna_classification": "Disease Name",
    "dna_reasoning": "Disease Name",
    "evidence_ranking": "Selected Disease (your top pick)",
    "protein_function": "Function Description",
    "kegg_pathway_reasoning": "Disease Name",
    "perturbation_qa": "(use perturbation_answers JSON below)",
    "perturbation_direction_qa": "(use direction_answers JSON below)",
    "perturbation_benchmark": "(use direction_answers JSON below)",
    "clinical_diagnosis": "Final Diagnosis",
}

ANSWER_PLACEHOLDERS = {
    "dna_classification": "e.g. cushing syndrome",
    "dna_reasoning": "e.g. cushing syndrome",
    "evidence_ranking": "e.g. cushing syndrome",
    "protein_function": "e.g. Forms voltage-independent pH-gated sodium channels...",
    "kegg_pathway_reasoning": "e.g. amyotrophic lateral sclerosis",
    "perturbation_qa": "(ignored — use the JSON input)",
    "perturbation_direction_qa": "(ignored — use the JSON input)",
    "perturbation_benchmark": "(ignored — use the JSON input)",
    "clinical_diagnosis": "e.g. Bisphosphonate-associated atypical femoral fracture",
}

# ── Session state ────────────────────────────────────────────────────────

_session = {"task_id": "", "task_type": "", "active": False}


# ── Interactive Environment callbacks ────────────────────────────────────

def on_task_change(task_type):
    """When task type dropdown changes: update field visibility, labels, and auto-reset."""
    show_reasoning = task_type not in ("dna_classification",)
    show_protein = task_type == "protein_function"
    show_ranking = task_type in ("evidence_ranking", "clinical_diagnosis")
    show_elim = task_type in (
        "evidence_ranking",
        "perturbation_qa",
        "perturbation_direction_qa",
        "perturbation_benchmark",
        "kegg_pathway_reasoning",
    )

    obs = env.reset(task_type=task_type)
    _session["task_id"] = obs.task_id
    _session["task_type"] = obs.task_type
    _session["active"] = True

    question_md = _format_question(obs)
    seq_md = _format_sequences(obs)
    candidates_md = _format_candidates(obs)
    status_md = _format_status(obs, active=True)

    return (
        # Observation displays
        question_md,
        seq_md,
        candidates_md,
        gr.update(visible=show_ranking),  # candidates accordion
        status_md,
        # Task info
        f"### {TASK_LABELS.get(task_type, task_type)}\n\n{TASK_DESCRIPTIONS.get(task_type, '')}",
        # Answer field updates
        gr.update(label=ANSWER_LABELS.get(task_type, "Answer"),
                  placeholder=ANSWER_PLACEHOLDERS.get(task_type, ""),
                  value=""),
        # Reasoning field
        gr.update(visible=show_reasoning, value=""),
        # GO terms field
        gr.update(visible=show_protein, value=""),
        # Subcellular location field
        gr.update(visible=show_protein, value=""),
        # Ranked diseases / differential ranking field
        gr.update(visible=show_ranking, value=""),
        # Elimination reasoning / batch answers JSON field
        gr.update(visible=show_elim, value="", **_elim_field_meta(task_type)),
        # Clear results
        "",
        "",
    )


def _elim_field_meta(task_type: str) -> dict:
    if task_type == "perturbation_qa":
        return {
            "label": "Perturbation Answers (JSON: pair_id → true/false)",
            "placeholder": '{"pair_001": true, "pair_002": false}',
        }
    if task_type in ("perturbation_direction_qa", "perturbation_benchmark"):
        return {
            "label": "Direction Answers (JSON: pair_id → 'Increase'|'Decrease'|'Unknown')",
            "placeholder": '{"pair_001": "Increase", "pair_002": "Decrease"}',
        }
    if task_type == "kegg_pathway_reasoning":
        return {
            "label": "Mentioned Genes (JSON list)",
            "placeholder": '["TARDBP", "CxI", "Q"]',
        }
    return {
        "label": "Elimination Reasoning (JSON: disease → why eliminated)",
        "placeholder": '{"parkinsons disease": "The pathway involves cortisol, not dopamine..."}',
    }


def on_reset(task_type):
    """Reset button: sample a new problem for current task type."""
    return on_task_change(task_type)


def on_submit(task_type, answer, reasoning, go_terms_str, location, ranked_str, elim_str):
    """Submit the agent's action and display grading results."""
    if not _session["active"]:
        return (
            "### ⚠️ No Active Episode\n\nClick **Reset** or change the task type to start.",
            "",
            _format_status_inactive(),
        )

    go_terms = None
    if go_terms_str and go_terms_str.strip():
        go_terms = [t.strip() for t in go_terms_str.split(",") if t.strip()]

    ranked_list = None
    if ranked_str and ranked_str.strip():
        ranked_list = [d.strip() for d in ranked_str.split(",") if d.strip()]

    parsed_json = None
    if elim_str and elim_str.strip():
        try:
            parsed_json = json.loads(elim_str)
        except json.JSONDecodeError:
            parsed_json = None

    action_kwargs: dict = {
        "task_id": _session["task_id"],
        "answer": answer or "",
        "reasoning": reasoning if reasoning else None,
        "go_terms": go_terms,
        "subcellular_location": location if location else None,
    }
    if task_type == "evidence_ranking":
        action_kwargs["ranked_diseases"] = ranked_list
        action_kwargs["elimination_reasoning"] = parsed_json if isinstance(parsed_json, dict) else None
    elif task_type == "clinical_diagnosis":
        action_kwargs["differential_ranking"] = ranked_list
    elif task_type == "perturbation_qa":
        if isinstance(parsed_json, dict):
            action_kwargs["perturbation_answers"] = {k: bool(v) for k, v in parsed_json.items()}
    elif task_type in ("perturbation_direction_qa", "perturbation_benchmark"):
        if isinstance(parsed_json, dict):
            action_kwargs["direction_answers"] = {
                str(k): str(v) for k, v in parsed_json.items()
            }
    elif task_type == "kegg_pathway_reasoning":
        if isinstance(parsed_json, list):
            action_kwargs["mentioned_genes"] = [str(g) for g in parsed_json]

    action = BioresearchAction(**action_kwargs)

    obs = env.step(action)
    reward = obs.reward or 0.0
    breakdown = obs.metadata.get("score_breakdown", {}) if obs.metadata else {}
    _session["active"] = False

    reward_md = _format_reward(reward, breakdown)
    breakdown_json = json.dumps(breakdown, indent=2, default=str)
    status_md = _format_status_done(reward)

    return reward_md, breakdown_json, status_md


# ── Formatting helpers ───────────────────────────────────────────────────

def _format_question(obs):
    q = obs.question
    if len(q) > 3000:
        q = q[:3000] + "\n\n*[truncated for display]*"
    extras = []
    if getattr(obs, "pathway_graph", None):
        extras.append(f"\n\n[Pathway graph]\n  {obs.pathway_graph}")
    if getattr(obs, "direction_batch", None):
        lines = []
        for pair in obs.direction_batch[:20]:
            variant = pair.get("variant") or ""
            tag = f" [{variant}]" if variant else ""
            lines.append(
                f"  - {pair.get('pair_id')}{tag}: {pair.get('query_gene')} "
                f"-> {pair.get('target_gene')} ({pair.get('cell_line')})"
            )
        if lines:
            extras.append("\n\n[Pairs]\n" + "\n".join(lines))
    return q + "".join(extras)


def _format_sequences(obs):
    if not obs.sequence_data:
        return ""
    parts = []
    for key, val in obs.sequence_data.items():
        label = key.replace("_", " ").title()
        if len(val) > 300:
            display = f"`{val[:120]}` ... `{val[-120:]}`\n\n*({len(val)} total characters)*"
        else:
            display = f"`{val}`"
        parts.append(f"**{label}**\n\n{display}")
    return "\n\n---\n\n".join(parts)


def _format_candidates(obs):
    if not obs.candidate_diseases:
        return ""
    items = "\n".join(f"- {d}" for d in obs.candidate_diseases)
    return f"**Candidate Diseases** (rank these):\n\n{items}"


def _format_status(obs, active=True):
    return (
        f"🟢 **Episode Active**\n\n"
        f"- **Task ID**: `{obs.task_id}`\n"
        f"- **Task Type**: `{obs.task_type}`\n"
        f"- Submit your response below"
    )


def _format_status_inactive():
    return "⚪ **No Episode** — Select a task and click Reset."


def _format_status_done(reward):
    if reward >= 0.7:
        emoji = "🟢"
    elif reward >= 0.3:
        emoji = "🟡"
    else:
        emoji = "🔴"
    return (
        f"{emoji} **Episode Complete** — Reward: **{reward:.4f}**\n\n"
        f"Click **Reset** or switch task to start a new episode."
    )


def _format_reward(reward, breakdown):
    if reward >= 0.7:
        bar_color = "green"
        emoji = "🟢"
        label = "Good"
    elif reward >= 0.3:
        bar_color = "orange"
        emoji = "🟡"
        label = "Mediocre"
    else:
        bar_color = "red"
        emoji = "🔴"
        label = "Poor"

    pct = int(reward * 100)
    md = f"## {emoji} Reward: {reward:.4f}  ({label})\n\n"
    md += f'<div style="background:#e0e0e0;border-radius:8px;height:24px;width:100%;margin:8px 0">'
    md += f'<div style="background:{bar_color};border-radius:8px;height:24px;width:{pct}%"></div></div>\n\n'

    if breakdown:
        md += "| Component | Score |\n|-----------|-------|\n"
        for key, val in breakdown.items():
            if isinstance(val, (int, float)):
                md += f"| {key.replace('_', ' ').title()} | {val:.4f} |\n"
    return md


# ── Dataset Explorer callbacks ───────────────────────────────────────────

def browse_dna(idx):
    idx = int(idx)
    task_id = f"dna_{idx:03d}"
    try:
        s = data_loader.get_dna_sample_by_id(task_id)
    except KeyError:
        return "Invalid index", "", "", ""

    q = s.question if len(s.question) <= 2000 else s.question[:2000] + "..."
    r = s.reasoning if len(s.reasoning) <= 2000 else s.reasoning[:2000] + "..."
    meta = (
        f"**Task ID**: `{task_id}`  |  "
        f"**Reference seq**: {len(s.reference_sequence)} bp  |  "
        f"**Variant seq**: {len(s.variant_sequence)} bp"
    )
    return q, f"**{s.answer}**", r, meta


def browse_protein(idx):
    idx = int(idx)
    task_id = f"protein_{idx:03d}"
    try:
        s = data_loader.get_protein_sample_by_id(task_id)
    except KeyError:
        return "", "", "", "", ""

    header = f"### {s.protein_names}\n\n**Organism**: {s.organism}  |  **Length**: {int(s.length)} aa  |  **ID**: `{s.protein_id}`"
    go = ", ".join(s.go_ids[:15]) if s.go_ids else "None annotated"
    interpro = s.interpro_formatted if s.interpro_formatted else "None"
    return header, s.protein_function, s.subcellular_location, go, interpro


# ── GRPO Analysis callback ───────────────────────────────────────────────

def run_grpo_analysis(task_type, sample_idx, r1, r2, r3):
    prefix = "protein" if task_type == "protein_function" else "dna"
    task_id = f"{prefix}_{int(sample_idx):03d}"

    responses = [r1, r2, r3]
    scores = []
    breakdowns = []

    for text in responses:
        if not text or not text.strip():
            scores.append(0.0)
            breakdowns.append({})
            continue

        env.reset(task_type=task_type, task_id=task_id)
        action = BioresearchAction(
            task_id=task_id,
            answer=text.strip(),
            reasoning=text.strip() if task_type != "dna_classification" else None,
        )
        obs = env.step(action)
        scores.append(obs.reward or 0.01)
        breakdowns.append(obs.metadata.get("score_breakdown", {}) if obs.metadata else {})

    valid = [s for s in scores if s > 0]
    mean = sum(valid) / len(valid) if valid else 0.0
    std = (sum((s - mean) ** 2 for s in valid) / len(valid)) ** 0.5 if len(valid) >= 2 else 0.0
    advantages = [(s - mean) / std if std > 0 else 0.0 for s in scores]
    spread = max(scores) - min(scores) if scores else 0.0

    md = "## GRPO Group Analysis\n\n"
    md += f"**Task**: `{task_type}` | **Sample**: `{task_id}` | **Group size**: {len(valid)}\n\n"
    md += "| # | Response (preview) | Score | Advantage | Rating |\n"
    md += "|---|-------------------|-------|-----------|--------|\n"
    for i, (s, a, text) in enumerate(zip(scores, advantages, responses)):
        preview = (text or "").strip()[:50].replace("|", "\\|").replace("\n", " ")
        if not preview:
            preview = "*(empty)*"
        rating = "🟢" if s >= 0.7 else ("🟡" if s >= 0.3 else "🔴")
        md += f"| {i+1} | {preview} | {s:.4f} | {a:+.4f} | {rating} |\n"

    md += f"\n**Spread**: {spread:.4f}{'  ✅ Sufficient for GRPO' if spread >= 0.4 else '  ⚠️ Low variance'}\n"
    md += f"**Mean**: {mean:.4f}  |  **Std**: {std:.4f}\n\n"

    md += "---\n\n### Score Breakdowns\n\n"
    for i, bd in enumerate(breakdowns):
        if bd:
            md += f"<details><summary><strong>Response {i+1}</strong></summary>\n\n"
            md += f"```json\n{json.dumps(bd, indent=2, default=str)}\n```\n\n</details>\n\n"

    return md


# ── Lab Mode callbacks ───────────────────────────────────────────────────

_lab_session = {"task_id": "", "task_type": "", "active": False}


def _format_lab_status(obs, reward_so_far: float) -> str:
    return (
        f"🟢 **Lab Episode Active**\n\n"
        f"- **Task**: `{obs.task_type}`\n"
        f"- **Task ID**: `{obs.task_id}`\n"
        f"- **Phase**: `{obs.phase}`\n"
        f"- **Steps remaining**: {obs.remaining_steps}\n"
        f"- **Per-step reward sum**: {reward_so_far:+.3f}"
    )


def _format_notebook(obs) -> str:
    if not obs.notebook:
        return "*Notebook empty — call a tool to populate evidence.*"
    lines = ["| Step | Tool | Args | Result (preview) |", "|------|------|------|------------------|"]
    for entry in obs.notebook[-10:]:
        step = entry.get("step", "?")
        tool = entry.get("tool", "?")
        args = json.dumps(entry.get("args", {}))[:60].replace("|", "\\|")
        result = entry.get("result", {})
        if isinstance(result, dict):
            if "error" in result:
                preview = f"⚠️ {result['error']}"
            else:
                bits = []
                for k, v in result.items():
                    if isinstance(v, (str, int, float)):
                        bits.append(f"{k}={str(v)[:60]}")
                preview = "; ".join(bits[:2])
        else:
            preview = str(result)[:80]
        preview = preview[:120].replace("|", "\\|").replace("\n", " ")
        lines.append(f"| {step} | `{tool}` | `{args}` | {preview} |")
    return "\n".join(lines)


def on_lab_reset(task_type):
    obs = lab_env.reset(task_type=task_type)
    _lab_session["task_id"] = obs.task_id
    _lab_session["task_type"] = obs.task_type
    _lab_session["active"] = True
    _lab_session["reward_sum"] = 0.0

    question = obs.question
    if len(question) > 3000:
        question = question[:3000] + "\n\n*[truncated]*"

    status = _format_lab_status(obs, 0.0)
    notebook = _format_notebook(obs)
    tool_result = "*No tool called yet.*"
    reward_md = "*Submit an action to see the terminal reward.*"
    return question, status, notebook, tool_result, reward_md, ""


def on_lab_tool(task_type, tool_name, args_json):
    if not _lab_session["active"]:
        return ("*No active lab episode — click Reset.*",) * 4

    try:
        args = json.loads(args_json) if args_json and args_json.strip() else {}
    except json.JSONDecodeError as e:
        args = {}
        err = f"⚠️ Invalid args JSON: {e}"
    else:
        err = None

    action = BioresearchAction(
        task_id=_lab_session["task_id"],
        tool_name=tool_name,
        tool_args=args,
    )
    obs = lab_env.step(action)
    _lab_session["reward_sum"] += obs.reward or 0.0

    tool_result_md = f"```json\n{json.dumps(obs.tool_result or {}, indent=2, default=str)}\n```"
    if err:
        tool_result_md = err + "\n\n" + tool_result_md

    status = _format_lab_status(obs, _lab_session["reward_sum"])
    notebook = _format_notebook(obs)
    if obs.done:
        _lab_session["active"] = False
        status = f"🔴 **Episode ended unexpectedly** (step_reward={obs.reward:.3f})"
    return status, notebook, tool_result_md, f"*Step reward: {obs.reward:+.3f}*"


def on_lab_submit(
    task_type,
    answer,
    reasoning,
    go_terms_str,
    location,
    intervention_json,
    predicted_ligand="",
    differential_str="",
):
    if not _lab_session["active"]:
        return ("*No active lab episode — click Reset.*", "", "")

    go_terms = None
    if go_terms_str and go_terms_str.strip():
        go_terms = [t.strip() for t in go_terms_str.split(",") if t.strip()]

    intervention = None
    if intervention_json and intervention_json.strip():
        try:
            intervention = json.loads(intervention_json)
        except json.JSONDecodeError:
            intervention = None

    differential_ranking = None
    if differential_str and differential_str.strip():
        differential_ranking = [d.strip() for d in differential_str.split(",") if d.strip()]

    action = BioresearchAction(
        task_id=_lab_session["task_id"],
        submit=True,
        answer=answer or "",
        reasoning=reasoning if reasoning else None,
        go_terms=go_terms,
        subcellular_location=location if location else None,
        proposed_intervention=intervention,
        predicted_ligand=predicted_ligand if predicted_ligand and predicted_ligand.strip() else None,
        differential_ranking=differential_ranking,
    )
    obs = lab_env.step(action)
    breakdown = obs.metadata.get("score_breakdown", {}) if obs.metadata else {}
    _lab_session["active"] = False

    reward_md = _format_reward(obs.reward or 0.0, breakdown)
    bd_json = json.dumps(breakdown, indent=2, default=str)
    status_md = _format_status_done(obs.reward or 0.0)
    return reward_md, bd_json, status_md


# ═══════════════════════════════════════════════════════════════════════════
# UI Layout
# ═══════════════════════════════════════════════════════════════════════════

CSS = """
.task-info { padding: 12px 16px; border-radius: 8px; background: #f0f7ff; border-left: 4px solid #3b82f6; }
.status-box { padding: 10px 14px; border-radius: 8px; background: #f9fafb; border: 1px solid #e5e7eb; }
"""

with gr.Blocks(title="Bioresearch Playground") as demo:

    gr.Markdown("# 🧬 Bioresearch OpenEnv Playground\n*Interactive testing UI for the biological reasoning environment*")

    with gr.Tabs():

        # ──────────────────────────────────────────────────────────────
        # TAB 1: Interactive Environment
        # ──────────────────────────────────────────────────────────────
        with gr.TabItem("🔬 Interactive Environment"):

            with gr.Row():
                with gr.Column(scale=3):
                    task_dropdown = gr.Dropdown(
                        choices=TASK_TYPES,
                        value="dna_classification",
                        label="Task Type",
                        info="Select a task, then click Reset or just switch here to auto-load a new problem.",
                    )
                with gr.Column(scale=1):
                    reset_btn = gr.Button("🔄 Reset - New Problem", variant="primary", size="lg")

            with gr.Row(equal_height=False):
                # LEFT: Observation panel
                with gr.Column(scale=3):
                    task_info = gr.Markdown(
                        value=f"### {TASK_LABELS['dna_classification']}\n\n{TASK_DESCRIPTIONS['dna_classification']}",
                        # elem_classes=["task-info"],
                    )
                    # status_display = gr.Markdown(value=_format_status_inactive(), elem_classes=["status-box"])
                    status_display = gr.Markdown(value=_format_status_inactive())

                    with gr.Accordion("📋 Question", open=True):
                        question_display = gr.Textbox(
                            label="Question / Prompt",
                            lines=8,
                            interactive=False,
                            buttons=["copy"],
                        )

                    with gr.Accordion("🧬 Sequence Data", open=False):
                        sequence_display = gr.Markdown(value="*Reset to load a problem*")

                    with gr.Accordion("🎯 Candidate Diseases (evidence_ranking / clinical_diagnosis only)", open=True, visible=False) as candidates_accordion:
                        candidates_display = gr.Markdown(value="")

                # RIGHT: Action panel
                with gr.Column(scale=2):
                    gr.Markdown("### Your Response")

                    answer_input = gr.Textbox(
                        label=ANSWER_LABELS["dna_classification"],
                        placeholder=ANSWER_PLACEHOLDERS["dna_classification"],
                        lines=2,
                    )

                    reasoning_input = gr.Textbox(
                        label="Reasoning (step-by-step biological mechanism)",
                        placeholder="Step 1: The mutation in gene X causes...\nStep 2: This leads to...\nStep 3: Resulting in...",
                        lines=6,
                        visible=False,
                    )

                    go_terms_input = gr.Textbox(
                        label="GO Terms (comma-separated IDs)",
                        placeholder="GO:0005886, GO:0016020, GO:0005575",
                        lines=1,
                        visible=False,
                    )

                    location_input = gr.Textbox(
                        label="Subcellular Location",
                        placeholder="e.g. Cell membrane; Multi-pass membrane protein",
                        lines=1,
                        visible=False,
                    )

                    ranked_input = gr.Textbox(
                        label="Ranked Diseases (comma-separated, most likely first)",
                        placeholder="cushing syndrome, parkinsons disease, als, diabetes",
                        lines=1,
                        visible=False,
                    )

                    elim_input = gr.Textbox(
                        label="Elimination Reasoning (JSON: disease → why eliminated)",
                        placeholder='{"parkinsons disease": "The pathway involves cortisol, not dopamine..."}',
                        lines=4,
                        visible=False,
                    )

                    submit_btn = gr.Button("✅ Submit Action", variant="primary", size="lg")

            with gr.Accordion("📊 Grading Results", open=True):
                reward_display = gr.Markdown(value="*Submit an action to see results*")
                breakdown_display = gr.Code(label="Full Score Breakdown (JSON)", language="json", value="")

            # ── Wiring ──

            all_reset_outputs = [
                question_display,
                sequence_display,
                candidates_display,
                candidates_accordion,
                status_display,
                task_info,
                answer_input,
                reasoning_input,
                go_terms_input,
                location_input,
                ranked_input,
                elim_input,
                reward_display,
                breakdown_display,
            ]

            task_dropdown.change(on_task_change, inputs=[task_dropdown], outputs=all_reset_outputs)
            reset_btn.click(on_reset, inputs=[task_dropdown], outputs=all_reset_outputs)

            submit_btn.click(
                on_submit,
                inputs=[task_dropdown, answer_input, reasoning_input, go_terms_input, location_input, ranked_input, elim_input],
                outputs=[reward_display, breakdown_display, status_display],
            )

        # ──────────────────────────────────────────────────────────────
        # TAB 2: Dataset Explorer
        # ──────────────────────────────────────────────────────────────
        with gr.TabItem("📚 Dataset Explorer"):

            gr.Markdown("### DNA Reasoning Dataset\n*100 samples — DNA mutations linked to diseases via biological pathways*")

            with gr.Row():
                dna_slider = gr.Slider(0, data_loader.dna_count - 1, step=1, value=0, label="Sample Index")
                dna_btn = gr.Button("Load Sample", variant="secondary")

            with gr.Row():
                with gr.Column():
                    dna_question_out = gr.Textbox(label="Question", lines=6, interactive=False, buttons=["copy"])
                with gr.Column():
                    dna_answer_out = gr.Markdown(label="Gold Answer")
            dna_reasoning_out = gr.Textbox(label="Gold Reasoning Trace", lines=6, interactive=False, buttons=["copy"])
            dna_meta_out = gr.Markdown()

            dna_btn.click(browse_dna, inputs=[dna_slider], outputs=[dna_question_out, dna_answer_out, dna_reasoning_out, dna_meta_out])
            dna_slider.change(browse_dna, inputs=[dna_slider], outputs=[dna_question_out, dna_answer_out, dna_reasoning_out, dna_meta_out])

            gr.Markdown("---")
            gr.Markdown("### Protein Function Dataset\n*100 samples — Proteins with curated function, location, and GO annotations*")

            with gr.Row():
                prot_slider = gr.Slider(0, data_loader.protein_count - 1, step=1, value=0, label="Sample Index")
                prot_btn = gr.Button("Load Sample", variant="secondary")

            prot_header_out = gr.Markdown()
            with gr.Row():
                with gr.Column():
                    prot_func_out = gr.Textbox(label="Function", lines=4, interactive=False, buttons=["copy"])
                with gr.Column():
                    prot_loc_out = gr.Textbox(label="Subcellular Location", interactive=False)
            with gr.Row():
                with gr.Column():
                    prot_go_out = gr.Textbox(label="GO IDs", interactive=False, buttons=["copy"])
                with gr.Column():
                    prot_interpro_out = gr.Textbox(label="InterPro Domains", lines=3, interactive=False)

            prot_btn.click(browse_protein, inputs=[prot_slider], outputs=[prot_header_out, prot_func_out, prot_loc_out, prot_go_out, prot_interpro_out])
            prot_slider.change(browse_protein, inputs=[prot_slider], outputs=[prot_header_out, prot_func_out, prot_loc_out, prot_go_out, prot_interpro_out])

        # ──────────────────────────────────────────────────────────────
        # TAB 3: GRPO Reward Analysis
        # ──────────────────────────────────────────────────────────────
        with gr.TabItem("📈 GRPO Reward Analysis"):

            gr.Markdown(
                "### Compare multiple responses to the same problem\n\n"
                "GRPO computes advantages *relative to the group*. Enter 3 different responses "
                "to see how reward variance enables effective policy optimization.\n\n"
                "> **Tip**: Try one correct, one partially correct, and one wrong answer to see the spread."
            )

            with gr.Row():
                grpo_task = gr.Dropdown(choices=TASK_TYPES, value="dna_classification", label="Task Type")
                grpo_idx = gr.Number(value=0, label="Sample Index", precision=0)

            with gr.Row():
                grpo_r1 = gr.Textbox(label="Response 1 (best attempt)", lines=3, placeholder="e.g. cushing syndrome")
                grpo_r2 = gr.Textbox(label="Response 2 (mediocre)", lines=3, placeholder="e.g. adrenal disorder")
                grpo_r3 = gr.Textbox(label="Response 3 (poor)", lines=3, placeholder="e.g. diabetes")

            grpo_btn = gr.Button("🔍 Analyse Group", variant="primary")
            grpo_output = gr.Markdown(value="*Enter responses and click Analyse*")

            grpo_btn.click(
                run_grpo_analysis,
                inputs=[grpo_task, grpo_idx, grpo_r1, grpo_r2, grpo_r3],
                outputs=[grpo_output],
            )

        # ──────────────────────────────────────────────────────────────
        # TAB 4: Drug Discovery Lab (long-horizon, tool-calling)
        # ──────────────────────────────────────────────────────────────
        with gr.TabItem("🧪 Drug Discovery Lab"):

            gr.Markdown(
                "### Long-horizon lab episodes with tool calls\n\n"
                "Reset a lab task, then iteratively **Call Tools** to gather evidence into your "
                "notebook. When ready, click **Submit** to finalise and receive a terminal reward.\n\n"
                "> **Themes covered**: Long-Horizon Planning · World Modeling · Self-Improvement"
            )

            with gr.Row():
                lab_task = gr.Dropdown(choices=LAB_TASK_TYPES, value="target_discovery_lab", label="Lab Task")
                lab_reset_btn = gr.Button("🔄 Reset Episode", variant="primary")

            lab_status = gr.Markdown(value="⚪ **No Episode** — Click Reset to start.")
            lab_question = gr.Textbox(label="Opening Brief / Question", lines=10, interactive=False)

            gr.Markdown("### 🔧 Tool Call")
            with gr.Row():
                lab_tool = gr.Dropdown(choices=LAB_TOOLS, value="get_pathway", label="Tool")
                lab_args = gr.Textbox(
                    label="Tool Args (JSON)",
                    value='{"gene": "TP53"}',
                    lines=2,
                    placeholder='e.g. {"gene":"TP53"} or {"protein_id":"P12345"}',
                )
                lab_call_btn = gr.Button("📞 Call Tool", variant="secondary")

            lab_step_reward = gr.Markdown(value="")
            lab_tool_result = gr.Markdown(value="*No tool called yet.*")

            gr.Markdown("### 📓 Notebook (rolling evidence)")
            lab_notebook = gr.Markdown(value="*Notebook empty — call a tool to populate evidence.*")

            gr.Markdown("---")
            gr.Markdown("### 📤 Submit Final Answer")

            with gr.Row():
                lab_answer = gr.Textbox(label="Answer (disease / function)", lines=2)
                lab_location = gr.Textbox(label="Subcellular Location (optional)")
            lab_reasoning = gr.Textbox(label="Reasoning Chain", lines=4)
            with gr.Row():
                lab_go = gr.Textbox(label="GO Terms (comma-separated IDs)")
                lab_intervention = gr.Textbox(
                    label="Proposed Intervention (JSON, optional)",
                    value='{"mode":"inhibit","target":"TP53"}',
                )
            with gr.Row():
                lab_predicted_ligand = gr.Textbox(
                    label="Predicted Ligand (SMILES or drug name — DRUG_DESIGN / ligand_design)",
                    placeholder="e.g. CS(=O)(=O)N1CCC(Nc2ncccc2-c2cnc3[nH]ccc3n2)C1",
                )
                lab_differential = gr.Textbox(
                    label="Differential Ranking (comma-separated — clinical_diagnosis_lab)",
                    placeholder="most_likely, next, next",
                )
            lab_submit_btn = gr.Button("✅ Submit Episode", variant="primary")

            lab_reward_out = gr.Markdown(value="*Submit an action to see the terminal reward.*")
            lab_breakdown_out = gr.Code(label="Score Breakdown (JSON)", language="json", value="")

            lab_reset_btn.click(
                on_lab_reset,
                inputs=[lab_task],
                outputs=[lab_question, lab_status, lab_notebook, lab_tool_result, lab_reward_out, lab_breakdown_out],
            )
            lab_call_btn.click(
                on_lab_tool,
                inputs=[lab_task, lab_tool, lab_args],
                outputs=[lab_status, lab_notebook, lab_tool_result, lab_step_reward],
            )
            lab_submit_btn.click(
                on_lab_submit,
                inputs=[
                    lab_task,
                    lab_answer,
                    lab_reasoning,
                    lab_go,
                    lab_location,
                    lab_intervention,
                    lab_predicted_ligand,
                    lab_differential,
                ],
                outputs=[lab_reward_out, lab_breakdown_out, lab_status],
            )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, theme=gr.themes.Soft(), css=CSS)
