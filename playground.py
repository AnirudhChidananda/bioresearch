"""
Bioresearch Gradio Playground

Interactive UI for testing the Bioresearch OpenEnv environment.
Runs directly against the environment (no server needed).

    python playground.py
"""

import json

import gradio as gr

from server import actors as actors_module
from server import tools as tools_module
from server.data_loader import DataLoader
from server.bioresearch_environment import BioresearchEnvironment
from models import BioresearchAction

data_loader = DataLoader()
env = BioresearchEnvironment()

# Dedicated environment instance for the Virtual Tumor Board tab so its
# session state can live alongside the single-turn tab without collisions.
vtb_env = BioresearchEnvironment()

TASK_TYPES = ["dna_classification", "dna_reasoning", "evidence_ranking", "protein_function"]

TASK_LABELS = {
    "dna_classification": "Task 1 — DNA Mutation Disease Classification (Easy)",
    "dna_reasoning": "Task 2 — DNA Mutation Biological Reasoning (Medium)",
    "evidence_ranking": "Task 3 — Variant Pathogenicity Evidence Ranking (Medium-Hard)",
    "protein_function": "Task 4 — Protein Function Hypothesis Generation (Hard)",
}

TASK_DESCRIPTIONS = {
    "dna_classification": "Identify the disease caused by a DNA mutation. Reply with **only the disease name**.",
    "dna_reasoning": "Identify the disease **and** explain the biological mechanism step-by-step.",
    "evidence_ranking": "Rank 4 candidate diseases. Eliminate wrong ones with reasoning. Support your top pick.",
    "protein_function": "Predict protein function, subcellular location, and GO terms from sequence data.",
}

ANSWER_LABELS = {
    "dna_classification": "Disease Name",
    "dna_reasoning": "Disease Name",
    "evidence_ranking": "Selected Disease (your top pick)",
    "protein_function": "Function Description",
}

ANSWER_PLACEHOLDERS = {
    "dna_classification": "e.g. cushing syndrome",
    "dna_reasoning": "e.g. cushing syndrome",
    "evidence_ranking": "e.g. cushing syndrome",
    "protein_function": "e.g. Forms voltage-independent pH-gated sodium channels...",
}

# ── Session state ────────────────────────────────────────────────────────

_session = {"task_id": "", "task_type": "", "active": False}


# ── Interactive Environment callbacks ────────────────────────────────────

def on_task_change(task_type):
    """When task type dropdown changes: update field visibility, labels, and auto-reset."""
    show_reasoning = task_type != "dna_classification"
    show_protein = task_type == "protein_function"
    show_ranking = task_type == "evidence_ranking"

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
        # Ranked diseases field
        gr.update(visible=show_ranking, value=""),
        # Elimination reasoning field
        gr.update(visible=show_ranking, value=""),
        # Clear results
        "",
        "",
    )


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

    ranked_diseases = None
    if ranked_str and ranked_str.strip():
        ranked_diseases = [d.strip() for d in ranked_str.split(",") if d.strip()]

    elimination_reasoning = None
    if elim_str and elim_str.strip():
        try:
            elimination_reasoning = json.loads(elim_str)
        except json.JSONDecodeError:
            pass

    action = BioresearchAction(
        task_id=_session["task_id"],
        answer=answer or "",
        reasoning=reasoning if reasoning else None,
        go_terms=go_terms,
        subcellular_location=location if location else None,
        ranked_diseases=ranked_diseases,
        elimination_reasoning=elimination_reasoning,
    )

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
    return q


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

                    with gr.Accordion("🎯 Candidate Diseases (Task 3 only)", open=True, visible=False) as candidates_accordion:
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
        # TAB 4: Virtual Tumor Board (multi-turn)
        # ──────────────────────────────────────────────────────────────
        with gr.TabItem("🧑‍⚕️ Virtual Tumor Board"):
            gr.Markdown(
                "### Multi-turn, multi-agent biomedical orchestration\n\n"
                "You are the **orchestrator** of a diagnostic panel. Consult specialists, run tools, "
                "then submit a consensus diagnosis. You have up to **8 turns**. "
                "Every tool and specialist is a deterministic pure function of the case — so you can "
                "replay the same scenario any number of times for GRPO."
            )

            _vtb_session = {"task_id": None, "turn": 0, "done": True, "gold": "", "last_output": ""}

            def _vtb_render_status():
                if _vtb_session["done"]:
                    return "⚪ **No active episode** — click Start new case."
                return (
                    f"🟢 **Active** — Case `{_vtb_session['task_id']}` — "
                    f"Turn {_vtb_session['turn']}/8"
                )

            def _vtb_render_history(history_summary):
                if not history_summary:
                    return "*(no turns yet)*"
                lines = []
                for h in history_summary:
                    args_str = json.dumps(h.get("args", {}), default=str)[:80]
                    out = (h.get("output_preview") or "")[:120].replace("\n", " ")
                    lines.append(f"**Turn {h['turn']}** — `{h['tool']}` {args_str}\n> {out}")
                return "\n\n".join(lines)

            def _vtb_candidates_md(obs):
                if not obs.candidate_diseases:
                    return ""
                items = "\n".join(f"- {d}" for d in obs.candidate_diseases)
                return f"**Candidate diagnoses (pick one):**\n{items}"

            def vtb_start(case_idx):
                idx = int(case_idx)
                task_id = f"dna_{idx:03d}"
                try:
                    data_loader.get_dna_sample_by_id(task_id)
                except KeyError:
                    return (
                        "Invalid case index.", "", "", "", _vtb_render_status(), "", "",
                        gr.update(interactive=False),
                    )

                obs = vtb_env.reset(task_type="virtual_tumor_board", task_id=task_id)
                _vtb_session.update({
                    "task_id": obs.task_id, "turn": 0, "done": False,
                    "gold": data_loader.get_dna_sample_by_id(task_id).answer,
                    "last_output": "",
                })
                return (
                    obs.question,
                    _vtb_candidates_md(obs),
                    _vtb_render_history(obs.history_summary or []),
                    "*(no tool called yet)*",
                    _vtb_render_status(),
                    "",
                    "",
                    gr.update(interactive=True),
                )

            def vtb_call_tool(tool_name, tool_args_json, gene, disease, role, specialist_question):
                if _vtb_session["done"] or not _vtb_session["task_id"]:
                    return (
                        "*(no active episode — start one first)*",
                        "*(no turns yet)*",
                        _vtb_render_status(),
                        "",
                        "",
                    )

                args = {}
                if tool_args_json and tool_args_json.strip():
                    try:
                        args = json.loads(tool_args_json)
                    except json.JSONDecodeError:
                        return (
                            "**ERROR**: tool_args is not valid JSON.",
                            _vtb_render_history([]),
                            _vtb_render_status(),
                            "",
                            "",
                        )

                if tool_name == "pathway_expand" and gene:
                    args.setdefault("gene", gene)
                if tool_name == "literature_snippet" and disease:
                    args.setdefault("disease", disease)
                if tool_name == "ask_specialist":
                    if role:
                        args.setdefault("role", role)
                    if specialist_question:
                        args.setdefault("question", specialist_question)

                action = BioresearchAction(
                    task_id=_vtb_session["task_id"],
                    tool_name=tool_name,
                    tool_args=args,
                )
                obs = vtb_env.step(action)
                _vtb_session["turn"] = obs.turn_count
                _vtb_session["last_output"] = obs.tool_output or ""

                if obs.done:
                    _vtb_session["done"] = True
                    breakdown = (obs.metadata or {}).get("score_breakdown", {}) or {}
                    return (
                        f"### Episode complete — reward {obs.reward:.4f}\n\n{_format_reward(obs.reward or 0.0, breakdown)}",
                        _vtb_render_history(obs.history_summary or []),
                        _vtb_render_status(),
                        json.dumps(breakdown, indent=2, default=str),
                        f"**Gold answer was:** `{_vtb_session['gold']}`",
                    )

                return (
                    obs.tool_output or "*(empty output)*",
                    _vtb_render_history(obs.history_summary or []),
                    _vtb_render_status(),
                    "",
                    "",
                )

            def vtb_submit_consensus(final_answer, final_reasoning):
                if _vtb_session["done"] or not _vtb_session["task_id"]:
                    return (
                        "*(no active episode — start one first)*",
                        "*(no turns yet)*",
                        _vtb_render_status(),
                        "",
                        "",
                    )
                action = BioresearchAction(
                    task_id=_vtb_session["task_id"],
                    tool_name="submit_consensus",
                    tool_args={"answer": final_answer or "", "reasoning": final_reasoning or ""},
                )
                obs = vtb_env.step(action)
                _vtb_session["done"] = True
                breakdown = (obs.metadata or {}).get("score_breakdown", {}) or {}
                return (
                    _format_reward(obs.reward or 0.0, breakdown),
                    _vtb_render_history(obs.history_summary or []),
                    _vtb_render_status(),
                    json.dumps(breakdown, indent=2, default=str),
                    f"**Gold answer was:** `{_vtb_session['gold']}`",
                )

            with gr.Row():
                vtb_case_idx = gr.Number(value=7, label="DNA case index (0–99)", precision=0)
                vtb_start_btn = gr.Button("🎬 Start new case", variant="primary")

            with gr.Row():
                with gr.Column(scale=3):
                    vtb_status = gr.Markdown(value=_vtb_render_status())
                    with gr.Accordion("📋 Case brief", open=True):
                        vtb_case = gr.Markdown("*(no case loaded — click Start)*")
                    vtb_candidates = gr.Markdown("")
                with gr.Column(scale=2):
                    gr.Markdown("### 🛠️ Tool controls")
                    vtb_tool = gr.Dropdown(
                        choices=sorted(tools_module.TOOL_NAMES),
                        value="ask_specialist",
                        label="Tool",
                    )
                    vtb_role = gr.Dropdown(
                        choices=actors_module.list_roles(),
                        value="geneticist",
                        label="role (for ask_specialist)",
                    )
                    vtb_spec_q = gr.Textbox(
                        label="question (for ask_specialist)",
                        placeholder="e.g. What is the variant's mechanistic impact?",
                        lines=2,
                    )
                    vtb_gene = gr.Textbox(label="gene (for pathway_expand)", placeholder="e.g. PDE11A")
                    vtb_disease = gr.Textbox(label="disease (for literature_snippet)", placeholder="e.g. cushing syndrome")
                    vtb_raw_args = gr.Textbox(
                        label="raw tool_args JSON (optional, merges with fields above)",
                        placeholder='{"role": "clinician"}',
                        lines=2,
                    )
                    vtb_call_btn = gr.Button("▶ Call tool (advance 1 turn)", interactive=False)

                    gr.Markdown("---")
                    gr.Markdown("### 🏁 Submit consensus")
                    vtb_final_answer = gr.Textbox(label="Final diagnosis", placeholder="e.g. cushing syndrome")
                    vtb_final_reasoning = gr.Textbox(
                        label="Synthesised reasoning",
                        lines=4,
                        placeholder="Based on specialist inputs...",
                    )
                    vtb_submit_btn = gr.Button("🏁 Submit consensus & grade", variant="primary")

            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 📨 Last tool output")
                    vtb_last = gr.Markdown("*(no tool called yet)*")
                with gr.Column():
                    gr.Markdown("### 📜 Trajectory so far")
                    vtb_history = gr.Markdown("*(no turns yet)*")

            gr.Markdown("### 🎯 Final reward & breakdown")
            vtb_gold = gr.Markdown("")
            vtb_breakdown = gr.Code(language="json", value="", label="Score breakdown")

            vtb_start_btn.click(
                vtb_start,
                inputs=[vtb_case_idx],
                outputs=[vtb_case, vtb_candidates, vtb_history, vtb_last, vtb_status, vtb_breakdown, vtb_gold, vtb_call_btn],
            )
            vtb_call_btn.click(
                vtb_call_tool,
                inputs=[vtb_tool, vtb_raw_args, vtb_gene, vtb_disease, vtb_role, vtb_spec_q],
                outputs=[vtb_last, vtb_history, vtb_status, vtb_breakdown, vtb_gold],
            )
            vtb_submit_btn.click(
                vtb_submit_consensus,
                inputs=[vtb_final_answer, vtb_final_reasoning],
                outputs=[vtb_last, vtb_history, vtb_status, vtb_breakdown, vtb_gold],
            )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, theme=gr.themes.Soft(), css=CSS)
