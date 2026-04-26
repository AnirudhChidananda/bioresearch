---

## title: Bioresearch Environment Server
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

A biological reasoning environment for training and evaluating AI agents on real-world genomics and proteomics tasks. This environment pairs **fast single-step tasks** with a **long-horizon, tool-calling "Drug Discovery Lab"** that trains frontier models to reason about disease mechanisms, aging biology, and druggable targets — and gives GRPO a dense per-step reward signal derived from gold `<think>` reasoning traces.

- **Blog.** ([Blog.md](./blog.md))
- **Hugging Face Space** ([Bioresearch](https://huggingface.co/spaces/anirudhchida/bioresearch))
- **Training Script.** Unsloth + TRL GRPO Colab (T4) ([Google Colab Notebook](https://colab.research.google.com/drive/1CQjgDTPaPkEqWe8vA0xNt9TyxWa8fwAx#scrollTo=HeQZU9DHPbtE)) [notebooks/train_grpo_t4.ipynb]
- **Metrics.** ([Trackio Link](https://huggingface.co/spaces/anirudhchida/trackio))

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

The 14 tasks are grouped into five narrative scenes that walk the agent up the biological abstraction stack (variant → protein → pathway / perturbation → clinical) and then into long-horizon, tool-calling labs.

### Scene 1 — Variant reasoning


| Task                 | Mode        | Difficulty  | Source Data          | Description                                                                  |
| -------------------- | ----------- | ----------- | -------------------- | ---------------------------------------------------------------------------- |
| `dna_classification` | single-step | Easy        | `DNA_reasoning.json` | Identify the disease caused by a DNA mutation given pathway context          |
| `dna_reasoning`      | single-step | Medium      | `DNA_reasoning.json` | Identify disease AND explain the step-by-step biological mechanism           |
| `evidence_ranking`   | single-step | Medium-Hard | `DNA_reasoning.json` | Rank 4 candidate diseases with elimination reasoning and supporting evidence |


### Scene 2 — Protein function


| Task               | Mode        | Difficulty | Source Data                  | Description                                                                |
| ------------------ | ----------- | ---------- | ---------------------------- | -------------------------------------------------------------------------- |
| `protein_function` | single-step | Hard       | `Protien_sft_reasoning.json` | Predict protein function, subcellular location, and GO terms from sequence |


### Scene 3 — Systems biology (pathway + perturbation)


| Task                        | Mode              | Difficulty | Source Data                                                                                 | Description                                                                                                                                   |
| --------------------------- | ----------------- | ---------- | ------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| `kegg_pathway_reasoning`    | single-step       | Hard       | `kegg_reasoning.json` + `kegg_reasoning_2.json`                                             | KEGG-style declarative pathway graph (`TARDBP* -                                                                                              |
| `perturbation_qa`           | single-step batch | Hard       | `PertubationQA_language_pert_de.json`                                                       | Batched CRISPRi world-modeling: predict whether knocking down query_gene changes target_gene in a given cell line                             |
| `perturbation_direction_qa` | single-step batch | Hard       | `PertubationQA_Language_pert_dir.json`                                                      | 3-class directional CRISPRi world-modeling (`Increase` / `Decrease` / `Unknown`) — denser reward signal than the binary perturbation task     |
| `perturbation_benchmark`    | single-step batch | Very Hard  | `PertubationQA_Language_pert_dir.json` + `pert_de.json` + `gse_pert.json` + `gse_gene.json` | Umbrella CRISPRi benchmark across four variants with a weighted mean (25% per variant) so one score compares directional reasoning end-to-end |


### Scene 4 — Clinical


| Task                 | Mode        | Difficulty  | Source Data                    | Description                                                                                                        |
| -------------------- | ----------- | ----------- | ------------------------------ | ------------------------------------------------------------------------------------------------------------------ |
| `clinical_diagnosis` | single-step | Medium-Hard | `diagnosis_training_data.json` | Rank radiology differentials, commit to a final diagnosis, and mirror the gold `gptoss120b_reasoning` step-by-step |


### Scene 5 — Long-horizon labs (tool-calling)


| Task                     | Mode             | Difficulty | Source Data                                                                                    | Description                                                                                                                               |
| ------------------------ | ---------------- | ---------- | ---------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| `protein_hypothesis_lab` | **long-horizon** | Very Hard  | `Protien_sft_reasoning.json` + `Protien_catalogue.json` + `SMILES_top1000_drug_discovery.json` | Build a mechanistic hypothesis with dense per-step reward from gold `<think>` traces, plus optional DRUG_DESIGN closing move              |
| `target_discovery_lab`   | **long-horizon** | Very Hard  | `DNA_reasoning.json` + `Protien_sft_reasoning.json` + `SMILES_top1000_drug_discovery.json`     | From a mutation brief, call tools to identify a druggable target, propose an intervention, and (DRUG_DESIGN phase) emit a concrete ligand |
| `clinical_diagnosis_lab` | **long-horizon** | Very Hard  | `diagnosis_training_data.json` + `Protien_catalogue.json`                                      | Diagnostic lab with tool access (search_catalogue / get_pathway / get_go) and dense per-step process reward                               |
| `ligand_design`          | short-horizon    | Very Hard  | `drug_discovery_hetionet.json` + `SMILES_top1000_drug_discovery.json`                          | Propose a high-pIC50 molecule (SMILES or drug name) for a gene; graded by token-set Jaccard + property proximity + catalogue membership   |
| `curriculum_self_play`   | **long-horizon** | Adaptive   | `Protien_catalogue.json`                                                                       | Self-play capstone that progressively hides tool outputs as the agent improves                                                            |


#### Intentional nested-grader scaffolding

Two tasks deliberately share graders with richer downstream tasks — kept on purpose as curriculum anchors and ablation baselines, not pruned as duplicates:

- `**dna_classification` ⊂ `dna_reasoning*`*. `grade_dna_reasoning` calls `grade_dna_classification` internally (~40% of its score). Classification is the only easy-difficulty task in the registry and anchors the GRPO curriculum floor — it lets reward curves start from a non-trivial baseline without requiring the agent to emit reasoning chains.
- `**perturbation_direction_qa` ⊂ `perturbation_benchmark**`. `grade_perturbation_benchmark` reuses `grade_perturbation_direction` per variant (25% each). The standalone task uses only the clean `pert_dir` pool (low variance, one of the three headline GRPO reward curves), whereas the benchmark mixes four files at 2 samples/variant for a noisier "are we good across the whole CRISPRi axis?" score.

### Task details (all 14)

The tables above are the canonical list; the bullets below are one-paragraph each-task summaries. Together they are the full registry.

**Scene 1 — Variant reasoning**

- `**dna_classification` (Easy).** The agent receives a DNA mutation context (chromosome, pathway network, gene list, reference/variant sequences) and must identify the resulting disease. A label-only task with a finite disease set.
- `**dna_reasoning` (Medium).** Same input as `dna_classification`, but the agent must also articulate the step-by-step biological mechanism (e.g. PDE11A loss-of-function → elevated cAMP → Cushing). Reasoning is graded at the step level.
- `**evidence_ranking` (Medium–Hard).** Same mutation context plus 4 candidate diseases (1 correct, 3 distractors). Rank candidates, explain why each wrong disease was eliminated, and support the chosen label.

**Scene 2 — Protein function**

- `**protein_function` (Hard).** Given a protein sequence, name, organism, and InterPro domain annotations, the agent predicts biological function, subcellular location, and Gene Ontology terms with supporting reasoning.

**Scene 3 — Systems biology (pathway + perturbation)**

- `**kegg_pathway_reasoning` (Hard).** The observation includes a KEGG-style declarative pathway graph; the agent must identify the correct disease, quote pathway edges in its reasoning, and link cited genes to the graph.
- `**perturbation_qa` (Hard, batch).** A batch of CRISPRi pairs (query_gene, target_gene, cell line); predict yes/no for whether knock-down of the query changes the target. Reward uses macro–F1 and balanced accuracy.
- `**perturbation_direction_qa` (Hard, batch).** Same pairing setup as `perturbation_qa` but 3-class labels: `Increase` / `Decrease` / `Unknown` for directional effect. Sharper signal than the binary task for GRPO.
- `**perturbation_benchmark` (Very Hard, batch).** Reuses the directional grader on four data variants (pert_dir, pert_de, gse_pert, gse_gene) with 25% weight each for one umbrella end-to-end score.

**Scene 4 — Clinical**

- `**clinical_diagnosis` (Medium–Hard).** Radiology-style case: rank differentials, commit to a final diagnosis, and align step-by-step reasoning with the gold `gptoss120b_reasoning` trace.

**Scene 5 — Long-horizon and short-horizon labs (tool-calling)**

The five scene-5 task types are:

- `**protein_hypothesis_lab` (long-horizon, Very Hard).** Start from a protein brief; use the lab state machine to characterise, hypothesise, intervene, optionally complete **DRUG_DESIGN**, and submit. Terminal reward plus dense per-step process reward from gold `<think>` traces in `Protien_catalogue.json`.
- `**target_discovery_lab` (long-horizon, Very Hard).** From a **DNA mutation** brief, use tools to pick a druggable target, build a mechanistic case, propose an intervention, and optionally emit a concrete ligand in **DRUG_DESIGN**.
- `**clinical_diagnosis_lab` (long-horizon, Very Hard).** Diagnostic lab with the same tool stack as the other labs; per-step process reward is aligned to gold clinical reasoning in `diagnosis_training_data.json` where available.
- `**ligand_design` (short-horizon, Very Hard).** Propose a high-pIC50 molecule (SMILES or drug name) for a given gene; graded without requiring a full multi-turn lab episode.
- `**curriculum_self_play` (long-horizon, Adaptive).** Same family of lab loop as the other long-horizon tasks, with **curriculum** that progressively hides tool outputs as the agent improves.

**Phased state machine (long-horizon lab tasks: `protein_hypothesis_lab`, `target_discovery_lab`, `clinical_diagnosis_lab`, `curriculum_self_play`)**

These run inside a **phased state machine** (up to ~20 steps) with optional **DRUG_DESIGN** before submit:

1. **TARGET** — read the opening brief (DNA mutation or protein) and pick a candidate gene or protein to focus on.
2. **CHARACTERIZE** — call tools (`get_interpro`, `get_ppi`, `get_go`, `get_sequence`, `get_subcellular_location`, `search_catalogue`, `get_pathway`, `get_structure`, and in DRUG_DESIGN: `get_candidate_ligands`, `get_drug_properties`) to fill a rolling **notebook**.
3. **HYPOTHESIZE** — reason from the notebook toward a mechanism that explains the phenotype.
4. **INTERVENE** — propose a druggable modality, e.g. `{"mode": "inhibit" | "activate" | "degrade" | …, "target": "…"}`.
5. **DRUG_DESIGN** (optional window) — pick a concrete ligand where the task spec allows.
6. **SUBMIT** — terminal reward from answer quality, GO F1, intervention plausibility, tool efficiency, reasoning–notebook coherence, and (where applicable) ligand match; long-horizon tasks also get **dense per-step** reward from step-wise match to gold `<think>` traces.

Each step, the observation exposes `phase`, `remaining_steps`, `notebook`, `tool_result`, and `available_tools` so the agent can plan without the conversation context exploding.

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
    # v2 fields ──────────────────────────────────────
    predicted_ligand: Optional[str]                 # SMILES or drug name (DRUG_DESIGN / ligand_design)
    perturbation_answers: Optional[Dict[str, bool]] # {pair_id: yes/no} for perturbation_qa batches
    differential_ranking: Optional[List[str]]       # Ordered diff-dx list for clinical_diagnosis
    # v3 fields ──────────────────────────────────────
    direction_answers: Optional[Dict[str, str]]     # {pair_id: "Increase"|"Decrease"|"Unknown"} for directional CRISPRi tasks
    mentioned_genes: Optional[List[str]]            # Genes cited from the pathway graph (kegg_pathway_reasoning)
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
    phase: str                                      # TARGET | CHARACTERIZE | HYPOTHESIZE | INTERVENE | DRUG_DESIGN | SUBMIT
    tool_result: Optional[Dict[str, Any]]           # Response to the most recent tool call
    remaining_steps: int                            # Max additional steps before forced submit
    notebook: List[Dict[str, Any]]                  # Rolling evidence log from prior tool calls
    available_tools: List[str]                      # Tool names currently available to the agent
    # v2 fields ──────────────────────────────────────
    ligand_candidates: Optional[List[Dict[str, Any]]]       # pre-computed candidates for ligand_design
    perturbation_batch: Optional[List[Dict[str, str]]]      # batch of (pair_id, query, target, cell_line)
    differentials: Optional[List[str]]                      # differential candidates for clinical tasks
    # v3 fields ──────────────────────────────────────
    pathway_graph: Optional[str]                            # KEGG declarative graph string (kegg_pathway_reasoning)
    genes_in_pathway: Optional[List[str]]                   # Gene symbols parsed from the pathway context
    structure_path: Optional[str]                           # AlphaFold structure file hint (surfaced by get_structure)
    direction_batch: Optional[List[Dict[str, str]]]         # Directional CRISPRi batch with variant tag
    benchmark_variants: Optional[List[str]]                 # Variant labels for perturbation_benchmark
```

## Reward Design

All scores are in **[0.01, 0.99]** with continuous partial credit.

### Single-step tasks


| Component                                   | T1 (Easy) | T2 (Medium) | T3 (Med-Hard) | T4 (Hard)      |
| ------------------------------------------- | --------- | ----------- | ------------- | -------------- |
| Answer accuracy                             | 100%      | 40%         | 30% (ranking) | 25% (function) |
| Reasoning quality                           | —         | 60%         | 25%           | 20%            |
| Elimination reasoning                       | —         | —           | 35%           | —              |
| Subcellular location                        | —         | —           | —             | 20%            |
| GO term prediction (leaf F1 when available) | —         | —           | —             | 35%            |
| Logical consistency                         | —         | —           | 10%           | —              |


### Long-horizon lab tasks

Episodes combine a **terminal reward** (on submit) with **dense per-step process rewards** during CHARACTERIZE/HYPOTHESIZE:


| Component                                               | `target_discovery_lab` | `protein_hypothesis_lab` | `curriculum_self_play`  |
| ------------------------------------------------------- | ---------------------- | ------------------------ | ----------------------- |
| Disease / function accuracy                             | 30%                    | 25%                      | 20%                     |
| Reasoning quality                                       | 15%                    | 15%                      | 15%                     |
| Leaf-level GO F1                                        | 20%                    | 25%                      | 20%                     |
| Intervention plausibility                               | 15%                    | 10%                      | 10%                     |
| Tool efficiency (useful/redundant)                      | 10%                    | 10%                      | 10%                     |
| Reasoning-trace coherence w/ notebook                   | 10%                    | 10%                      | 10%                     |
| **Per-step** process reward (`<think>` step similarity) | ✓                      | ✓ (primary)              | ✓ (difficulty-weighted) |


The per-step reward is the best `**difflib.SequenceMatcher`** ratio between the agent's latest `reasoning` and any unseen gold `<think>` step from `Protien_catalogue.json`. This gives GRPO a visible reward gradient within minutes rather than waiting for terminal rollouts.

#### v2 task weights


| Component                                           | `clinical_diagnosis` | `clinical_diagnosis_lab` | `perturbation_qa` | `ligand_design` |
| --------------------------------------------------- | -------------------- | ------------------------ | ----------------- | --------------- |
| Final diagnosis match                               | 30%                  | 70% × 30%                | —                 | —               |
| Differential ranking                                | 25%                  | 70% × 25%                | —                 | —               |
| Gold CoT process trace                              | 25%                  | 70% × 25%                | —                 | —               |
| Reasoning quality                                   | 20%                  | 70% × 20%                | —                 | —               |
| Tool efficiency                                     | —                    | 30%                      | —                 | 20%             |
| Macro-F1 + balanced accuracy                        | —                    | —                        | 100%              | —               |
| SMILES token Jaccard                                | —                    | —                        | —                 | 80% × 40%       |
| Named-drug / SMILES equality bonus                  | —                    | —                        | —                 | 80% × 25%       |
| Top-1000 catalogue membership (drug_score weighted) | —                    | —                        | —                 | 80% × 25%       |
| Property proximity (logP, num_atoms)                | —                    | —                        | —                 | 80% × 10%       |


#### v3 task weights


| Component                                                 | `kegg_pathway_reasoning` | `perturbation_direction_qa` | `perturbation_benchmark` |
| --------------------------------------------------------- | ------------------------ | --------------------------- | ------------------------ |
| Disease / final-answer accuracy                           | 30%                      | —                           | —                        |
| Pathway-graph fidelity (Jaccard on edge tokens)           | 25%                      | —                           | —                        |
| Process-trace similarity to gold CoT                      | 25%                      | —                           | —                        |
| Pathway-gene coverage F1                                  | 20%                      | —                           | —                        |
| 3-class balanced accuracy (Increase / Decrease / Unknown) | —                        | 50%                         | —                        |
| 3-class macro-F1                                          | —                        | 50%                         | —                        |
| Per-variant directional score × 25% weight                | —                        | —                           | 25% × 4 variants         |


The directional grader normalises agent output aggressively: `up`, `increase`, `+`, `yes` → `Increase`; `down`, `decrease`, `-`, `no` → `Decrease`; anything else → `Unknown` and is scored neutrally (0.33). The `perturbation_benchmark` grader is just `grade_perturbation_direction` run four times (one per variant: `pert_dir`, `pert_de`, `gse_pert`, `gse_gene`) and averaged with equal weights — the per-variant sub-scores are exposed in `metadata.score_breakdown.per_variant` for plotting.

#### `get_structure` tool

`target_discovery_lab`, `protein_hypothesis_lab`, `clinical_diagnosis_lab`, and `curriculum_self_play` can now call `get_structure(protein_id=...)` to fetch the AlphaFold `structure_path` plus a deterministic 16-character signature. Proteins loaded from `Protien_data_2.json` / `Protien_sft_reasoning_2.json` expose this field; older entries return `{"error": "not_in_catalogue"}` so the agent learns to gracefully fall back. This lets the final hypothesis quote a concrete structure id (e.g. `AF-A0A0B4K6E2-F1-model_v4.cif`) — tightening the "mutation → structure → molecule" storytelling beat without adding any heavy deps.

#### Drug Design Phase

`target_discovery_lab` and `protein_hypothesis_lab` now schedule a **DRUG_DESIGN** window between `INTERVENE` and `SUBMIT`. During this phase the agent can call `get_candidate_ligands(gene, k=5)` and `get_drug_properties(smiles)` to pick a concrete molecule. On submit, if the agent populates `predicted_ligand`, a blended ligand-match + drug-tool-efficiency addon (≤ 15% weight) is folded into the existing terminal reward — turning the previously abstract `{"mode":"inhibit","target":"X"}` output into a real SMILES with a measurable pIC50.

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

## Project Structure

```
bioresearch/
├── __init__.py                  # Module exports
├── models.py                    # BioresearchAction & BioresearchObservation (incl. lab fields)
├── client.py                    # BioresearchEnv client (supports tool-calling schema)
├── inference.py                 # Baseline inference (single-step + long-horizon drivers)
├── training_core.py             # GRPO: prompts, parsers, rollouts, task-gated reward fns
├── training_a100.py             # Trackio + eval shims, paired demos (uses training_core)
├── playground.py                # Gradio UI (4 tabs: Interactive / Explorer / GRPO / Lab)
├── blog.md                      # Long-form project writeup (HF Community, etc.)
├── openenv.yaml                 # OpenEnv manifest listing all task types
├── pyproject.toml               # Dependencies
├── uv.lock                      # Pinned lockfile (uv)
├── Dockerfile                   # Container definition
├── README.md                    # This file
├── data/
│   ├── DNA_reasoning.json                  # DNA mutation / variant-brief pool (Scenes 1 & 5)
│   ├── Protien_sft_reasoning.json          # Primary protein SFT: reasoning + go_pred_leaf
│   ├── Protien_sft_reasoning_2.json        # Extended protein SFT (v2; may include structure_path)
│   ├── Protein_data.json                   # Alternate name for the primary SFT (DataLoader picks one)
│   ├── Protien_data_2.json                 # Extended protein pool (structure hints for get_structure, etc.)
│   ├── Protien_catalogue.json              # Gold <think> traces for per-step process reward
│   ├── kegg_reasoning.json                 # KEGG declarative-graph pathway cases
│   ├── kegg_reasoning_2.json               # Second KEGG graph pool
│   ├── diagnosis_training_data.json        # Radiology cases + gold gptoss120b step-wise CoT
│   ├── PertubationQA_language_pert_de.json  # CRISPRi binary Q&A (pert_de)
│   ├── PertubationQA_language_pert_dir.json # 3-class directional CRISPRi (pert_dir)
│   ├── PertubationQA_language_gse_pert.json # GSE-derived pert variant (benchmark slice)
│   ├── PertubationQA_language_gse_gene.json # GSE-derived gene variant (benchmark slice)
│   ├── drug_discovery_hetionet.json        # gene → SMILES / drug-name supervision
│   ├── SMILES_top1000_drug_discovery.json  # High-pIC50 catalogue (drug tools / ligand_design)
│   └── protein_catalogue_bridge.json        # Bridge-only records (not added to training pools)
├── knowledgebase/               # Design notes and hackathon copy
│   ├── brief.md
│   └── blog.md
├── server/
│   ├── __init__.py
│   ├── app.py                   # FastAPI application
│   ├── bioresearch_environment.py # Legacy + lab loop state machine
│   ├── data_loader.py           # Dataset loading, sampling, tool dispatch
│   ├── graders.py               # Graders (leaf-GO F1, process trace, intervention, tool efficiency)
│   └── requirements.txt
├── notebooks/
│   └── train_grpo_t4.ipynb      # Unsloth + TRL GRPO on Colab (T4)
└── tests/
    ├── __init__.py
    ├── test_graders.py          # Unit tests for grading functions
    ├── test_environment.py      # Integration tests
    ├── test_training_core.py    # Offline smoke tests for training_core
    └── test_training_a100.py   # training_a100 helpers
```


## Deploying to Hugging Face Spaces

```bash
openenv push
```
