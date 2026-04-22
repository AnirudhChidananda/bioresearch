---
title: Bioresearch Environment Server
emoji: 🧬
colorFrom: green
colorTo: blue
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# Bioresearch Environment — Drug Discovery Lab

A biological reasoning environment for training and evaluating AI agents on real-world genomics and proteomics tasks. Designed for **GRPO (Group Relative Policy Optimization)** compatibility and inspired by the [BioReason](https://arxiv.org/abs/2505.14028) model series from Arc Institute.

This environment pairs **fast single-step tasks** with a **long-horizon, tool-calling "Drug Discovery Lab"** that trains frontier models to reason about disease mechanisms, aging biology, and druggable targets — and gives GRPO a dense per-step reward signal derived from gold `<think>` reasoning traces.

**Hackathon themes covered**:

- *World Modeling / Professional Tasks* — full target→evidence→hypothesis→intervention loop.
- *Long-Horizon Planning & Instruction Following* — 8–20 tool-call steps per episode.
- *Self-Improvement* — curriculum self-play that progressively hides tool hints.

## Motivation

Understanding genetic variants and protein function is central to modern biomedical research and drug discovery. This environment evaluates whether AI agents can:

- **Classify** the pathogenic effects of DNA mutations using pathway and sequence context
- **Reason** through step-by-step biological mechanisms linking mutations to diseases
- **Predict** protein function, subcellular location, and Gene Ontology annotations from sequence data
- **Compare** and **eliminate** candidate diseases using structured evidence ranking
- **Operate** a multi-step lab workflow: pick a target, characterise it via tool calls, hypothesise a mechanism, and propose a therapeutic intervention

These are tasks that human experts routinely perform, making this a genuine real-world evaluation benchmark.

## Tasks

| Task | Mode | Difficulty | Source Data | Description |
|------|------|-----------|-------------|-------------|
| `dna_classification` | single-step | Easy | `DNA_reasoning.json` | Identify the disease caused by a DNA mutation given pathway context |
| `dna_reasoning` | single-step | Medium | `DNA_reasoning.json` | Identify disease AND explain the step-by-step biological mechanism |
| `evidence_ranking` | single-step | Medium-Hard | `DNA_reasoning.json` | Rank 4 candidate diseases with elimination reasoning and supporting evidence |
| `protein_function` | single-step | Hard | `Protien_sft_reasoning.json` | Predict protein function, subcellular location, and GO terms from sequence |
| `target_discovery_lab` | **long-horizon** | Very Hard | `DNA_reasoning.json` + `Protien_sft_reasoning.json` | From a mutation brief, iteratively call tools to identify a druggable target and propose an intervention |
| `protein_hypothesis_lab` | **long-horizon** | Very Hard | `Protien_sft_reasoning.json` + `Protien_catalogue.json` | From a protein brief, build a mechanistic hypothesis with dense per-step reward from gold `<think>` traces |
| `curriculum_self_play` | **long-horizon** | Adaptive | `Protien_catalogue.json` | Self-play curriculum that progressively hides tool outputs as the agent improves |

### Task 1: DNA Mutation Disease Classification (Easy)

The agent receives a DNA mutation context (chromosome, pathway network, gene list, reference/variant sequences) and must identify the resulting disease. A classification task with ~26 possible diseases.

### Task 2: DNA Mutation Biological Reasoning (Medium)

Same input as Task 1, but the agent must also articulate the step-by-step biological mechanism (e.g., "PDE11A loss-of-function → elevated cAMP → PKA activation → cortisol overproduction → Cushing syndrome"). Reasoning is graded at the step level.

### Task 3: Evidence Ranking (Medium-Hard)

The agent receives the same mutation context plus 4 candidate diseases (1 correct, 3 distractors). It must rank candidates, explain why each wrong disease was eliminated, and provide supporting evidence for the selected disease.

### Task 4: Protein Function Hypothesis Generation (Hard)

Given a protein sequence, name, organism, and InterPro domain annotations, the agent predicts biological function, subcellular location, and Gene Ontology terms with supporting reasoning.

### Task 5: Drug Discovery Lab (Long-Horizon)

Three new tasks run inside a **phased state machine** that gives agents up to 20 steps to:

1. **TARGET** — read an opening brief (DNA mutation or protein) and pick a candidate gene/protein.
2. **CHARACTERIZE** — call tools (`get_interpro`, `get_ppi`, `get_go`, `get_sequence`, `get_subcellular_location`, `search_catalogue`, `get_pathway`) to pull evidence into a rolling **notebook**.
3. **HYPOTHESIZE** — reason from the notebook toward a mechanism that explains the phenotype.
4. **INTERVENE** — propose a druggable modality: `{"mode": "inhibit" | "activate" | "degrade" | ..., "target": "..."}`.
5. **SUBMIT** — receive a terminal reward blending answer accuracy, GO-term F1 (leaf level), intervention plausibility, tool efficiency, and reasoning-trace coherence. **Dense per-step rewards** come from step-wise similarity of the agent's reasoning to gold `<think>` traces.

Each step, the observation exposes a `phase`, `remaining_steps`, `notebook`, `tool_result`, and `available_tools` so the agent can plan deliberately without the conversation context exploding.

## Action Space

```python
class BioresearchAction(Action):
    task_id: str                                    # ID of the task instance
    answer: str                                     # Disease name or function description (submit)
    reasoning: Optional[str]                        # Biological reasoning chain (dense reward signal)
    go_terms: Optional[List[str]]                   # Predicted GO terms
    subcellular_location: Optional[str]             # Predicted location
    ranked_diseases: Optional[List[str]]            # Ordered ranking (evidence_ranking)
    elimination_reasoning: Optional[Dict[str, str]] # Disease → why eliminated (evidence_ranking)
    # Long-horizon lab mode ──────────────────────────
    tool_name: Optional[str]                        # Name of the tool to invoke
    tool_args: Optional[Dict[str, Any]]             # Arguments passed to the tool
    submit: bool                                    # If True the episode is finalised and graded
    proposed_intervention: Optional[Dict[str, str]] # e.g. {"mode": "inhibit", "target": "PDE11A"}
```

## Observation Space

```python
class BioresearchObservation(Observation):
    task_id: str                                    # Unique problem instance ID
    task_type: str                                  # Task type identifier
    question: str                                   # The prompt for the agent
    sequence_data: Dict[str, str]                   # DNA or protein sequences
    context: Dict[str, Any]                         # Pathway genes, organism, domains, etc.
    candidate_diseases: Optional[List[str]]         # 4 candidates for evidence_ranking
    # Long-horizon lab mode ──────────────────────────
    phase: str                                      # TARGET | CHARACTERIZE | HYPOTHESIZE | INTERVENE | SUBMIT
    tool_result: Optional[Dict[str, Any]]           # Response to the most recent tool call
    remaining_steps: int                            # Max additional steps before forced submit
    notebook: List[Dict[str, Any]]                  # Rolling evidence log from prior tool calls
    available_tools: List[str]                      # Tool names currently available to the agent
```

## Reward Design

All scores are in **[0.01, 0.99]** with continuous partial credit.

### Single-step tasks

| Component | T1 (Easy) | T2 (Medium) | T3 (Med-Hard) | T4 (Hard) |
|-----------|-----------|-------------|----------------|-----------|
| Answer accuracy | 100% | 40% | 30% (ranking) | 25% (function) |
| Reasoning quality | — | 60% | 25% | 20% |
| Elimination reasoning | — | — | 35% | — |
| Subcellular location | — | — | — | 20% |
| GO term prediction (leaf F1 when available) | — | — | — | 35% |
| Logical consistency | — | — | 10% | — |

### Long-horizon lab tasks

Episodes combine a **terminal reward** (on submit) with **dense per-step process rewards** during CHARACTERIZE/HYPOTHESIZE:

| Component | `target_discovery_lab` | `protein_hypothesis_lab` | `curriculum_self_play` |
|-----------|------------------------|--------------------------|------------------------|
| Disease / function accuracy | 30% | 25% | 20% |
| Reasoning quality | 15% | 15% | 15% |
| Leaf-level GO F1 | 20% | 25% | 20% |
| Intervention plausibility | 15% | 10% | 10% |
| Tool efficiency (useful/redundant) | 10% | 10% | 10% |
| Reasoning-trace coherence w/ notebook | 10% | 10% | 10% |
| **Per-step** process reward (`<think>` step similarity) | ✓ | ✓ (primary) | ✓ (difficulty-weighted) |

The per-step reward is the best **`difflib.SequenceMatcher`** ratio between the agent's latest `reasoning` and any unseen gold `<think>` step from `Protien_catalogue.json`. This gives GRPO a visible reward gradient within minutes rather than waiting for terminal rollouts.

## GRPO Compatibility

This environment is designed for GRPO training loops:

- **Same-prompt replay**: `reset(task_id="dna_042")` always returns the identical observation
- **Deterministic grading**: Same (input, response) always produces the same reward (tools return deterministic data)
- **High reward variance**: Different quality responses yield meaningfully different scores
- **Process reward**: Tasks 2–4 weight reasoning quality at 40–70% of total score; lab tasks add per-step dense reward
- **Dense signal**: Continuous metrics (Jaccard, F1, leaf-F1, SequenceMatcher) — no binary pass/fail

## Quick Start

### Running Locally

```bash
# Install dependencies
uv sync

# Start the server
uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

# In another terminal, run inference
python inference.py
```

### Using Docker

```bash
docker build -t bioresearch-env:latest .
docker run -p 8000:8000 bioresearch-env:latest
```

### Running the Gradio Playground

```bash
# Install playground dependencies
uv sync --extra playground

# Start server first, then:
python playground.py
```

### Environment Variables

```bash
API_BASE_URL=https://router.huggingface.co/v1
MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
HF_TOKEN=your_token_here
```

## Project Structure

```
bioresearch/
├── __init__.py                  # Module exports
├── models.py                    # BioresearchAction & BioresearchObservation (incl. lab fields)
├── client.py                    # BioresearchEnv client (supports tool-calling schema)
├── inference.py                 # Baseline inference (single-step + long-horizon drivers)
├── playground.py                # Gradio UI (4 tabs: Interactive / Explorer / GRPO / Lab)
├── openenv.yaml                 # OpenEnv manifest listing all task types
├── pyproject.toml               # Dependencies
├── Dockerfile                   # Container definition
├── README.md                    # This file
├── data/
│   ├── DNA_reasoning.json       # 100 DNA mutation samples
│   ├── Protien_sft_reasoning.json # 100 protein samples w/ reasoning + go_pred_leaf
│   └── Protien_catalogue.json   # 100 rows with gold <think> traces for process reward
├── server/
│   ├── __init__.py
│   ├── app.py                   # FastAPI application
│   ├── bioresearch_environment.py # Legacy + lab loop state machine
│   ├── data_loader.py           # Dataset loading, sampling, tool dispatch
│   ├── graders.py               # Graders (leaf-GO F1, process trace, intervention, tool efficiency)
│   └── requirements.txt
├── notebooks/
│   └── train_grpo_colab.ipynb   # End-to-end GRPO training + reward plot
└── tests/
    ├── test_graders.py          # Unit tests for grading functions
    └── test_environment.py      # Integration tests
```

## Baseline Scores

| Task | Mean Score | Episodes |
|------|-----------|----------|
| dna_classification | TBD | 5 |
| dna_reasoning | TBD | 5 |
| evidence_ranking | TBD | 5 |
| protein_function | TBD | 5 |
| target_discovery_lab | TBD | 3 |
| protein_hypothesis_lab | TBD | 3 |
| curriculum_self_play | TBD | 3 |
| **Overall** | **TBD** | **29** |

*Scores will be filled after running baseline evaluation. The Colab at `notebooks/train_grpo_colab.ipynb` produces a before/after reward curve on the long-horizon tasks — the headline deliverable for judging.*

## Deploying to Hugging Face Spaces

```bash
openenv push
```
