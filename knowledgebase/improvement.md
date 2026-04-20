# Bioresearch → Hackathon Winning Improvement Plan

This document is a ruthlessly prioritised, phase-by-phase plan to take the current Bioresearch OpenEnv from a solid single-agent baseline to a **top-15 hackathon submission** that scores well on every judging criterion.

---

## 1. Current State vs. Hackathon Requirements

### What we already have (keep and reuse)

| Asset | Status | Notes |
|-------|--------|-------|
| OpenEnv-compliant server | ✅ Done | `server/app.py`, `server/bioresearch_environment.py` |
| 4 tasks, 200 curated samples | ✅ Done | DNA + Protein, increasing difficulty |
| GRPO-compatible deterministic grading | ✅ Done | `server/graders.py`, same-prompt replay works |
| `BioresearchEnv` client (HTTP + WebSocket) | ✅ Done | `client.py` |
| Baseline inference script (OpenAI API) | ✅ Done | `inference.py` |
| Gradio playground (3 tabs) | ✅ Done | `playground.py` |
| Docker + HF Spaces manifest | ✅ Done | `Dockerfile`, `openenv.yaml` |
| Unit + integration tests | ✅ Done | 34 tests passing |

### What we are missing (blocks hackathon submission)

| Gap | Impact | Priority |
|-----|--------|----------|
| **Training script (Unsloth or TRL) in Colab** | Hard minimum requirement — disqualification if absent | P0 |
| **Reward curves / before-after metrics** | 20% of score | P0 |
| **Mini-blog on HF OR <2-min video** | Hard minimum requirement | P0 |
| **Environment innovation beyond single-shot Q&A** | 40% of score — single-turn graders are table stakes | P0 |
| **Pitch / storytelling assets** | 30% of score | P1 |
| **Public HF Spaces deployment** | Demo polish | P1 |

---

## 2. Strategic Theme Choice

The current environment is single-agent, single-turn. That scores low on **Environment Innovation (40%)**. We will pivot the narrative and add one decisive innovation layer that reuses every line of existing code.

**Chosen primary theme:** **Theme #1 — Multi-Agent Interactions** (Halluminate sub-theme: *multi-actor environments where an agent manages multiple actors to discover and achieve a task*).

**Secondary theme:** **Theme #2 — Long-Horizon Planning & Instruction Following** (multi-turn case workup with tool calls).

**Tertiary theme (free bonus):** **Theme #3.1 — Professional Tasks, scientific workflow loops** — this is literally the example given in the requirements ("papers → code → experiments").

### The pitch in one sentence

> **"Virtual Tumor Board"** — a multi-agent OpenEnv where an **Orchestrator LLM** delegates sub-questions to specialist **actors** (Geneticist, Pathway Analyst, Structural Biologist, Clinician), **debates evidence across turns**, and must reach a **diagnostic consensus** — trained with GRPO on real biomedical cases.

This story is:

- **Novel** (multi-agent biology tumor boards are not common RL environments)
- **Believable** (tumor boards are how real diagnoses happen)
- **Emotional** (drug discovery / rare diseases / patient impact)
- **Buildable on top of existing code** (our 4 tasks become cases; our graders become specialist graders)

### Why this wins on each criterion

| Criterion | Weight | How we win |
|-----------|--------|-----------|
| Environment Innovation | 40% | Multi-agent consensus + long-horizon tool use + deterministic bio grading |
| Storytelling | 30% | "AI Tumor Board" narrative, real disease cases (Cushing, ALS, Parkinson's), patient-impact framing |
| Reward Improvement | 20% | GRPO training curves on Qwen2.5-1.5B showing measurable lift on consensus reward |
| Pipeline Setup | 10% | Colab notebook end-to-end: env → TRL GRPOTrainer → reward curve → inference |

---

## 3. Phased Roadmap

Phases are ordered by **criticality to winning**. Each phase has a clear deliverable and a concrete acceptance test.

### Phase A — Minimum Requirements Unlock (MUST HAVE)

> **Goal:** pass the disqualification bar. No ambitious changes yet.

#### A1. Create training Colab notebook
- **File:** `notebooks/train_grpo.ipynb`
- **Stack:** Hugging Face TRL `GRPOTrainer` + Unsloth-optimised `Qwen/Qwen2.5-1.5B-Instruct`
- **Flow:**
  1. `pip install openenv trl unsloth datasets wandb`
  2. Clone this repo, install, start `BioresearchEnvironment` in-process
  3. Wrap env as a TRL reward function: for each generated completion, call `env.reset(task_id=...)` + `env.step(action)` and return `reward`
  4. Run GRPO for 200–500 steps on `dna_classification` (fastest, highest variance grader)
  5. Log `reward/mean`, `reward/std`, `kl_divergence` per step
  6. Save final checkpoint to HF Hub
- **Acceptance:** notebook runs end-to-end on a free Colab T4, produces a reward curve PNG

#### A2. Generate reward curves and before/after metrics
- **File:** `notebooks/evaluation.ipynb`
- Run the **baseline** (`Qwen2.5-1.5B-Instruct`) and **GRPO-trained** checkpoint over all 4 tasks × 25 samples = 100 episodes
- Produce table: `{task → baseline_mean, trained_mean, delta}`
- Produce plot: reward-over-training-steps
- Save `baselines.json` with full numbers
- **Acceptance:** measurable positive delta on at least one task (target: +0.05 mean reward)

#### A3. Mini-blog on Hugging Face
- **File:** `knowledgebase/blog_draft.md` → publish as HF Space README or model card
- **Sections:** Problem → Environment → Reward Design → Training → Results → Try it yourself
- Include 3 images: tumor-board diagram, reward curve, sample trajectory
- **Acceptance:** published at `huggingface.co/blog/<user>/bioresearch-tumor-board`

#### A4. 90-second demo video
- **File:** `knowledgebase/video_script.md`
- **Script:**
  1. 0–15s — hook ("A 10-year diagnosis in 30 seconds")
  2. 15–45s — show Gradio playground with a DNA reasoning case
  3. 45–75s — show the reward curve / training result
  4. 75–90s — call to action
- Record with OBS, upload unlisted to YouTube
- **Acceptance:** < 2 min, uploaded, link in `README.md`

### Phase B — Environment Innovation (THE 40% PHASE)

> **Goal:** turn the single-turn graders into a multi-agent, multi-turn, tool-using environment without breaking existing tests.

#### B1. Add multi-turn state machine to environment
- **File:** `server/bioresearch_environment.py`
- Introduce `episode_turn_count` in `State` and a `max_turns` per task (e.g. 8)
- `done=False` until either (a) agent submits `final_answer`, (b) max_turns hit, or (c) a "submit_consensus" action is received
- Each intermediate step returns a **small shaping reward** (≥ 0 for useful tool calls, negative for redundant ones)
- **Acceptance:** `test_long_horizon.py` — agent can run 5 tool calls, then submit, and get cumulative reward

#### B2. Add tool actions (long-horizon + world modelling)
- **File:** `server/tools.py` (new)
- Tools the agent can call as actions:
  - `blast_lookup(sequence)` → nearest known protein (from local cache of UniProt hits)
  - `pathway_expand(gene)` → returns Reactome/KEGG neighbours
  - `go_term_lookup(protein_id)` → returns annotated GO terms
  - `literature_snippet(disease)` → returns a canned abstract (from curated `data/literature.json`)
  - `ask_specialist(role, question)` → delegates to a specialist actor (see B3)
  - `submit_consensus(answer, reasoning)` → terminates the episode
- Tool outputs are deterministic from a seed (GRPO replay requirement)
- **Acceptance:** `test_tools.py` — all tool calls are pure functions of (task_id, args)

#### B3. Add multi-agent "specialist actor" layer
- **File:** `server/actors.py` (new)
- Define 4 specialist personas with scripted behaviours (cheap, no extra LLM cost for training):
  - **Geneticist** — answers questions about variant → phenotype using `DNA_reasoning.json` reasoning
  - **Pathway Analyst** — answers using pathway gene lists
  - **Structural Biologist** — answers using `interpro_formatted`
  - **Clinician** — grades clinical plausibility
- Each specialist has its own small grader (`grade_geneticist_response`, etc.)
- The **main agent is the Orchestrator** — it decides *which* specialist to query, in what order
- **Acceptance:** `test_actors.py` — orchestrator that asks all 4 specialists gets a higher consensus reward than one that asks none

#### B4. Consensus grader
- **File:** `server/graders.py` → add `grade_consensus`
- Composite reward:
  - 40% — final answer correctness (reuse existing graders)
  - 25% — **specialist consultation coverage** (did the orchestrator consult the right experts for the case?)
  - 15% — **reasoning synthesis** (does the final answer integrate multiple specialist inputs?)
  - 10% — **efficiency penalty** (fewer redundant tool calls is better)
  - 10% — **process consistency** (final answer matches intermediate specialist inputs)
- **Acceptance:** score spread ≥ 0.5 across a set of 10 deliberately varied rollouts

#### B5. New task: `virtual_tumor_board` (headline task)
- Uses a DNA case with 4 candidate diseases (reuse `evidence_ranking` infrastructure)
- Multi-turn orchestration required to win
- Adds a 5th entry to `TASK_TYPES` in `openenv.yaml`
- **Acceptance:** end-to-end episode via `BioresearchEnv` client completes in ≤ 8 turns with reward > 0.6

### Phase C — Storytelling & Demo Polish (30%)

#### C1. Rewrite `README.md` around the tumor-board story
- New opening: patient vignette ("27-year-old with unexplained cortisol levels…")
- Architecture diagram (Mermaid) showing orchestrator ↔ specialists ↔ tools
- "Why this matters" section with real diagnostic statistics
- Link to blog + video + Colab + HF Space

#### C2. Upgrade Gradio playground for the tumor-board tab
- **File:** `playground.py` → add **Tab 4: Virtual Tumor Board**
- Live visualisation:
  - Sidebar: case facts (sequences, pathway)
  - Centre: chat-like turn log showing orchestrator → specialist exchanges
  - Right: running reward, specialist-coverage radar chart, turn budget
- Record this tab in action for the demo video
- **Acceptance:** a human user can step through a case turn-by-turn and see cumulative reward build up

#### C3. Deploy to Hugging Face Spaces
- Push server via `openenv push`
- Push Gradio playground as a second Space (CPU-only, points at the server Space)
- Verify public URLs in README
- **Acceptance:** both Spaces reachable by anonymous users

#### C4. Pitch deck (3 min, 5 slides)
- **File:** `knowledgebase/pitch.md` (outline) + slides on Google Slides
- Slide 1: The problem (diagnostic delay in rare disease)
- Slide 2: Our environment (architecture + 5 tasks)
- Slide 3: Innovation (multi-agent consensus + long-horizon + bio-specific grading)
- Slide 4: Results (reward curves, before/after table)
- Slide 5: Try it (Spaces URL, Colab URL, GitHub)
- **Acceptance:** 3-minute rehearsal clocks ≤ 180s

### Phase D — Reward Improvement Evidence (20%)

#### D1. Run two-stage training
- **Stage 1:** GRPO on `dna_classification` (fast signal, high variance)
- **Stage 2:** GRPO on `virtual_tumor_board` using the Stage 1 checkpoint
- Both stages in `notebooks/train_grpo.ipynb`
- Target: ≥ +0.10 reward delta on virtual_tumor_board after Stage 2

#### D2. Before/after qualitative trajectories
- **File:** `notebooks/before_after.ipynb`
- Pick 3 cases, show full orchestrator trajectories pre- and post-training
- Highlight:
  - Pre-training: asks random specialists, misses the diagnosis
  - Post-training: calls geneticist + pathway analyst, reaches consensus in fewer turns
- Save as an HTML report, embed in blog

#### D3. Evaluation harness
- **File:** `evaluate.py` (new, top level)
- CLI: `python evaluate.py --model <hf-repo> --tasks all --episodes 25`
- Emits `eval_results.json` + console table
- **Acceptance:** running against the baseline and trained checkpoint reproduces the numbers in the blog

### Phase E — Pipeline Coherence (10%)

#### E1. Ensure GRPO training loop is reproducible
- Pin versions in `pyproject.toml`: `trl>=0.14`, `openenv>=<latest>`, `unsloth>=<latest>`, `transformers<=<match>`
- Pin seed, Python version, hardware (Colab T4 + A100 variant)

#### E2. Add CI smoke test
- **File:** `.github/workflows/ci.yml`
- Run `pytest`, import `playground`, spin up server in background, hit `/health`
- **Acceptance:** green badge in README

#### E3. `make` targets for the judges
- **File:** `Makefile` (new)
- `make install` → uv sync
- `make server` → uvicorn server
- `make playground` → Gradio UI
- `make train` → open Colab link
- `make eval` → run `evaluate.py`
- `make demo` → full round-trip on one case, prints reward

---

## 4. File-Change Summary

| Phase | New files | Modified files |
|-------|-----------|----------------|
| A | `notebooks/train_grpo.ipynb`, `notebooks/evaluation.ipynb`, `knowledgebase/blog_draft.md`, `knowledgebase/video_script.md` | `README.md` |
| B | `server/tools.py`, `server/actors.py`, `data/literature.json`, `tests/test_long_horizon.py`, `tests/test_tools.py`, `tests/test_actors.py` | `server/bioresearch_environment.py`, `server/graders.py`, `models.py`, `client.py`, `openenv.yaml`, `server/__init__.py` |
| C | `knowledgebase/pitch.md` | `playground.py`, `README.md` |
| D | `evaluate.py`, `notebooks/before_after.ipynb` | `notebooks/train_grpo.ipynb` |
| E | `.github/workflows/ci.yml`, `Makefile` | `pyproject.toml` |

---

## 5. Time Budget (assuming one builder, ~3 working days onsite)

| Phase | Estimated hours | When |
|-------|-----------------|------|
| A — Minimum unlock (notebook + curves + blog + video) | 10h | Day 1 morning + evening |
| B — Multi-agent environment | 14h | Day 1 afternoon → Day 2 morning |
| C — Storytelling + playground + HF Space + pitch | 8h | Day 2 afternoon |
| D — Training evidence | 8h (mostly GPU wait) | Day 2 evening + Day 3 morning |
| E — Polish, CI, Makefile | 4h | Day 3 afternoon |
| **Total active** | **~44h** | |

If time is short, **cut Phase E entirely** and **cut B2 tools to just 2 (`pathway_expand`, `ask_specialist`)** — they are the only two needed to justify the multi-agent story.

---

## 6. Hackathon Submission Checklist

### Minimum requirements (disqualification bar)

- [ ] OpenEnv latest release (`openenv.yaml` valid, `openenv push` succeeds)
- [ ] Training script in Colab using **Unsloth or HF TRL** (`notebooks/train_grpo.ipynb`)
- [ ] Mini-blog on HF **or** <2 min YouTube video (prefer **both** — they are cheap)

### Scoring deliverables

- [ ] Environment is multi-agent AND long-horizon AND bio-specific (uniqueness)
- [ ] Reward curve image showing training lift (≥ +0.05 on at least one task)
- [ ] Before/after trajectories on 3 cases
- [ ] Deployed HF Space for server
- [ ] Deployed HF Space for Gradio playground
- [ ] Public GitHub repo with README, Makefile, Dockerfile
- [ ] 3-minute pitch ready, rehearsed under time
- [ ] Q&A anticipated questions prepared (see Appendix)

### Judging-criterion coverage

- [ ] **Innovation (40%)** — multi-agent tumor board + tool use + deterministic bio grading
- [ ] **Storytelling (30%)** — patient narrative + architecture diagram + live demo
- [ ] **Improvement (20%)** — GRPO curve + before/after table + qualitative trajectories
- [ ] **Pipeline (10%)** — reproducible Colab, pinned deps, CI green

---

## 7. Risk Register

| Risk | Likelihood | Mitigation |
|------|-----------|-----------|
| Multi-agent refactor breaks existing GRPO replay | Medium | Keep single-turn `reset(task_id)` path working; multi-turn is an additional mode gated by `task_type="virtual_tumor_board"` |
| GRPO training on Colab T4 is too slow for reward curve | High | Use Unsloth 4-bit Qwen 1.5B, limit to 100 steps, or rent an A100 for 1h |
| Specialist actor scripts feel "fake" to judges | Medium | Ground each specialist response in real data from `DNA_reasoning.json` / `Protein_reasoning.json` — no hallucinated strings |
| Video recording eats 3+ hours | Medium | Use the polished Gradio tab for a one-take screencast; script first, record second |
| OpenEnv latest has breaking API changes | Low | Pin exact version and test `openenv push` on Day 1 |

---

## 8. Execution Order (TL;DR)

1. **Day 1 AM** — A1 (training notebook) + A2 (reward curve) — unblocks disqualification bar
2. **Day 1 PM** — B1 + B2 (multi-turn + 2 tools) — minimum innovation lift
3. **Day 2 AM** — B3 + B4 + B5 (specialists + consensus grader + tumor-board task)
4. **Day 2 PM** — C1 + C2 + C3 (README + playground tab + HF Space deploy)
5. **Day 2 EVE** — D1 (training run, let it cook overnight if possible)
6. **Day 3 AM** — A3 + A4 (blog + video) — now with real numbers
7. **Day 3 PM** — D2 + D3 + C4 + E3 (before/after, evaluate.py, pitch, Makefile), rehearse pitch
8. **Submit**

---

## Appendix — Expected Judge Q&A

- **"Is the multi-agent behaviour actually emergent or scripted?"** — The *orchestrator* is trained (learned); the *specialists* are deterministic graders that reward good delegation. This is the same pattern as Gemini-MedPaLM and Google Med-AI multi-agent papers.
- **"How do you ensure reproducibility for GRPO?"** — `reset(task_id=X)` returns identical observations; tool outputs are pure functions; specialist responses are deterministic per `(task_id, question)`.
- **"Why not just fine-tune a larger model?"** — We show that GRPO on a 1.5B model with structured rewards beats zero-shot 72B inference on reasoning quality metrics.
- **"What is the real-world impact?"** — Rare-disease diagnostic odyssey (7-year average); automated triage could save lives and $Bn in misdiagnosis costs.
