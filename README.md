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

# Bioresearch Environment

A biological reasoning environment for training and evaluating AI agents on real-world genomics and proteomics tasks. Designed for **GRPO (Group Relative Policy Optimization)** compatibility, inspired by the [BioReason](https://arxiv.org/abs/2505.14028) model series from Arc Institute.

## Motivation

Understanding genetic variants and protein function is central to modern biomedical research and drug discovery. This environment evaluates whether AI agents can:

- **Classify** the pathogenic effects of DNA mutations using pathway and sequence context
- **Reason** through step-by-step biological mechanisms linking mutations to diseases
- **Predict** protein function, subcellular location, and Gene Ontology annotations from sequence data
- **Compare** and **eliminate** candidate diseases using structured evidence ranking

These are tasks that human experts routinely perform, making this a genuine real-world evaluation benchmark.

## Tasks

| Task | Difficulty | Source Data | Description |
|------|-----------|-------------|-------------|
| `dna_classification` | Easy | DNA_reasoning.json | Identify the disease caused by a DNA mutation given pathway context |
| `dna_reasoning` | Medium | DNA_reasoning.json | Identify disease AND explain the step-by-step biological mechanism |
| `evidence_ranking` | Medium-Hard | DNA_reasoning.json | Rank 4 candidate diseases with elimination reasoning and supporting evidence |
| `protein_function` | Hard | Protein_reasoning.json | Predict protein function, subcellular location, and GO terms from sequence |

### Task 1: DNA Mutation Disease Classification (Easy)

The agent receives a DNA mutation context (chromosome, pathway network, gene list, reference/variant sequences) and must identify the resulting disease. A classification task with ~26 possible diseases.

### Task 2: DNA Mutation Biological Reasoning (Medium)

Same input as Task 1, but the agent must also articulate the step-by-step biological mechanism (e.g., "PDE11A loss-of-function → elevated cAMP → PKA activation → cortisol overproduction → Cushing syndrome"). Reasoning is graded at the step level.

### Task 3: Evidence Ranking (Medium-Hard)

The agent receives the same mutation context plus 4 candidate diseases (1 correct, 3 distractors). It must rank candidates, explain why each wrong disease was eliminated, and provide supporting evidence for the selected disease.

### Task 4: Protein Function Hypothesis Generation (Hard)

Given a protein sequence, name, organism, and InterPro domain annotations, the agent predicts biological function, subcellular location, and Gene Ontology terms with supporting reasoning.

## Action Space

```python
class BioresearchAction(Action):
    task_id: str                                    # ID of the task instance
    answer: str                                     # Disease name or function description
    reasoning: Optional[str]                        # Biological reasoning chain
    go_terms: Optional[List[str]]                   # Predicted GO terms (T3 only)
    subcellular_location: Optional[str]             # Predicted location (T3 only)
    ranked_diseases: Optional[List[str]]            # Ordered ranking (T4 only)
    elimination_reasoning: Optional[Dict[str, str]] # Disease → why eliminated (T4 only)
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
```

## Reward Design

All scores are in **[0.01, 0.99]** with continuous partial credit:

| Component | T1 (Easy) | T2 (Medium) | T3 (Med-Hard) | T4 (Hard) |
|-----------|-----------|-------------|----------------|-----------|
| Answer accuracy | 100% | 40% | 30% (ranking) | 25% (function) |
| Reasoning quality | — | 60% | 25% | 20% |
| Elimination reasoning | — | — | 35% | — |
| Subcellular location | — | — | — | 20% |
| GO term prediction | — | — | — | 35% |
| Logical consistency | — | — | 10% | — |

## GRPO Compatibility

This environment is designed for GRPO training loops:

- **Same-prompt replay**: `reset(task_id="dna_042")` always returns the identical observation
- **Deterministic grading**: Same (input, response) always produces the same reward
- **High reward variance**: Different quality responses yield meaningfully different scores
- **Process reward**: Tasks 2–4 weight reasoning quality at 40–70% of total score
- **Dense signal**: Continuous metrics (Jaccard, F1, coverage ratios) — no binary pass/fail

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
├── __init__.py              # Module exports
├── models.py                # BioresearchAction & BioresearchObservation
├── client.py                # BioresearchEnv client
├── inference.py             # Baseline inference script
├── playground.py            # Gradio UI playground
├── openenv.yaml             # OpenEnv manifest with 4 tasks
├── pyproject.toml           # Dependencies
├── Dockerfile               # Container definition
├── README.md                # This file
├── data/
│   ├── DNA_reasoning.json   # 100 DNA mutation samples
│   └── Protein_reasoning.json # 100 protein function samples
├── server/
│   ├── __init__.py
│   ├── app.py               # FastAPI application
│   ├── bioresearch_environment.py  # Core environment logic
│   ├── data_loader.py       # Dataset loading and sampling
│   ├── graders.py           # Grading functions for all 4 tasks
│   └── requirements.txt
└── tests/
    ├── test_graders.py      # Unit tests for grading functions
    └── test_environment.py  # Integration tests
```

## Baseline Scores

| Task | Mean Score | Episodes |
|------|-----------|----------|
| dna_classification | TBD | 5 |
| dna_reasoning | TBD | 5 |
| evidence_ranking | TBD | 5 |
| protein_function | TBD | 5 |
| **Overall** | **TBD** | **20** |

*Scores will be filled after running baseline evaluation.*

## Deploying to Hugging Face Spaces

```bash
openenv push
```
