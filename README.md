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
  - grpo
  - multi-agent
  - bioinformatics
---

# Bioresearch — a multi-agent GRPO environment for biomedical diagnosis

A real hospital diagnoses a rare disease with a **multidisciplinary tumor board**: a geneticist, a pathway analyst, a structural biologist and a clinician sit down with a case, delegate sub-questions, pull in literature, and converge over multiple turns.

**Bioresearch** is an OpenEnv that recreates that workflow for an LLM agent. It ships five biomedical tasks of increasing difficulty — four single-turn benchmarks and one multi-turn, multi-agent, tool-using **Virtual Tumor Board** — all designed from the ground up for GRPO training.

- 🎯 **5 tasks**, easy → expert
- 🧠 **4 deterministic specialist actors** the agent can consult
- 🛠️ **6 tools** (BLAST lookup, pathway expansion, GO lookup, literature, ask specialist, submit consensus)
- ⚖️ **5-component consensus grader** with ~0.6 reward spread per case — ideal for GRPO
- 📓 **Colab GRPO training notebook** (Qwen 2.5 1.5B + Unsloth + TRL)
- 🖥️ **Gradio playground** for interactive case review

## Quick start

```bash
uv sync --extra dev --extra playground

# Run the full test suite
uv run pytest

# Baseline evaluation (random vs heuristic vs gold ceiling)
uv run python evaluate.py --policy heuristic --episodes 5
uv run python evaluate.py --policy gold --episodes 5

# Interactive Gradio playground
uv run python playground.py

# HTTP server
uv run python -m server.app
```

## The five tasks

| # | Task                  | Turns | Difficulty  | Gold-policy reward | Random-policy reward |
|---|-----------------------|-------|-------------|--------------------|----------------------|
| 1 | `dna_classification`  | 1     | easy        | 0.90               | ~0.04                |
| 2 | `dna_reasoning`       | 1     | medium      | 0.80               | ~0.06                |
| 3 | `evidence_ranking`    | 1     | medium-hard | 0.85               | ~0.08                |
| 4 | `protein_function`    | 1     | hard        | ~0.65              | ~0.05                |
| 5 | `virtual_tumor_board` | 8     | expert      | 0.84               | ~0.08                |

The reward gap between `random` and `gold` is your training signal — Bioresearch keeps it wide on every task.

### Task 1–4: single-turn biomedical reasoning

Classic "one prompt, one answer" tasks with process-aware graders:

- **Task 1 (`dna_classification`)** — name the disease from a variant + pathway context.
- **Task 2 (`dna_reasoning`)** — same input, but also articulate the step-by-step mechanism. Graded on step structure, concept coverage, causal connectors, and hallucination penalty.
- **Task 3 (`evidence_ranking`)** — rank 4 candidate diseases, justify eliminations, and supply supporting evidence.
- **Task 4 (`protein_function`)** — predict function, subcellular location and GO terms from the raw protein sequence + InterPro domains.

### Task 5: the Virtual Tumor Board (multi-turn, multi-agent)

**Reset** loads a real curated case (sequences + pathway + 4 candidate diagnoses).

**Each turn** the orchestrator calls one tool:

```
blast_lookup        → nearest-known-gene summary for the variant
pathway_expand(gene)→ pathway neighbours of a named gene
go_term_lookup      → GO annotations (real, for proteins; inferred, for DNA cases)
literature_snippet  → curated abstract for a candidate disease
ask_specialist(role)→ delegate to geneticist / pathway_analyst / structural_biologist / clinician
submit_consensus    → terminate with (answer, reasoning)
```

Every tool and every specialist is a **deterministic pure function of the case** — no external API calls, no hallucinated biology, no RNG. Rollouts are bit-for-bit reproducible, which is a hard requirement for GRPO same-prompt replay.

Budget: up to 8 turns. After `submit_consensus` (or when the budget runs out), the episode is graded and the reward is decomposed into five components:

| Component              | Weight | What it measures                                                |
|------------------------|-------:|-----------------------------------------------------------------|
| Answer accuracy        |   40%  | final disease matches the gold (Jaccard-smoothed)               |
| Specialist coverage    |   25%  | did the agent consult the right specialists for the case type?  |
| Reasoning synthesis    |   15%  | final reasoning integrates specialist outputs + pathway genes   |
| Efficiency             |   10%  | fewer redundant calls, shorter trajectory                       |
| Process consistency    |   10%  | final answer appears in intermediate tool outputs               |

## Action and observation space

```python
class BioresearchAction(Action):
    task_id: str
    answer: str = ""                       # single-turn final answer
    reasoning: str | None = None
    go_terms: list[str] | None = None              # T4
    subcellular_location: str | None = None        # T4
    ranked_diseases: list[str] | None = None       # T3
    elimination_reasoning: dict[str, str] | None = None  # T3
    tool_name: str | None = None                   # T5
    tool_args: dict | None = None                  # T5

class BioresearchObservation(Observation):
    task_id: str
    task_type: str
    question: str
    sequence_data: dict[str, str]
    context: dict
    candidate_diseases: list[str] | None           # T3, T5
    turn_count: int                                # T5
    max_turns: int                                 # T5
    tool_output: str | None                        # T5
    available_tools: list[str] | None              # T5
    available_specialists: list[str] | None        # T5
    history_summary: list[dict] | None             # T5
```

## GRPO compatibility

Bioresearch is designed for GRPO-style training out of the box:

1. **Same-prompt replay.** `env.reset(task_id="dna_007")` returns bit-for-bit identical observations. Required for the advantage estimator.
2. **Process reward.** The tumor-board grader scores the *trajectory*, not just the answer — so a lucky guess can never score higher than a correctly orchestrated rollout.
3. **Reward variance inside a group.** Our held-out split shows ~0.6 spread between good and bad rollouts on the same case — plenty of signal for the GRPO advantage.
4. **Smooth reward landscape.** Jaccard-smoothed correctness + multiple reward components = no flat regions that kill the policy gradient.
5. **Cheap rollouts.** Specialists and tools are Python functions, not LLM calls. A full 8-turn episode runs in milliseconds on CPU.

See `notebooks/train_grpo.ipynb` for a working Qwen 2.5 1.5B + Unsloth + TRL training loop that fits in a free Colab T4.

## Evaluate a policy

The `evaluate.py` CLI ships with four built-in policies:

```bash
uv run python evaluate.py --policy random --episodes 5
uv run python evaluate.py --policy heuristic --episodes 5
uv run python evaluate.py --policy gold --episodes 5

# Or point at any OpenAI-compatible API (e.g. HF router):
export HF_TOKEN=...
uv run python evaluate.py --policy openai --model Qwen/Qwen2.5-7B-Instruct --episodes 10
```

Results are written to `eval_results.json` (per-episode breakdown) and a table is printed to stdout.

## Project structure

```
bioresearch/
├── models.py                         # Pydantic action/observation types
├── client.py                         # OpenEnv client (WebSocket)
├── inference.py                      # OpenAI-router inference harness
├── evaluate.py                       # Policy evaluation CLI
├── playground.py                     # Gradio interactive playground
├── openenv.yaml                      # OpenEnv manifest
├── Dockerfile                        # HF Space deployment
├── server/
│   ├── app.py                        # FastAPI app (create_app)
│   ├── bioresearch_environment.py    # 5-task Environment + multi-turn state machine
│   ├── data_loader.py                # Curated dataset access + distractor sampling
│   ├── graders.py                    # All 5 deterministic graders
│   ├── tools.py                      # 6 tool implementations
│   └── actors.py                     # 4 specialist actors
├── data/
│   ├── DNA_reasoning.json            # DNA variant + mechanism dataset
│   ├── Protein_reasoning.json        # Protein function dataset
│   └── literature.json               # Curated abstracts for the literature tool
├── notebooks/
│   ├── train_grpo.ipynb              # Qwen + GRPO training notebook
│   └── evaluation.ipynb              # Baseline + before/after comparison
├── tests/                            # 60+ unit tests (graders, tools, actors, env)
└── knowledgebase/
    ├── plan.md                       # Initial design doc
    ├── improvement.md                # Hackathon roadmap
    ├── blog_draft.md                 # 1000-word technical writeup
    ├── pitch.md                      # 30-second elevator pitch
    └── video_script.md               # 3-minute demo video script
```

## Environment variables

| Variable         | Default                                      | Purpose                                 |
|------------------|----------------------------------------------|-----------------------------------------|
| `API_BASE_URL`   | `https://router.huggingface.co/v1`           | OpenAI-compatible endpoint for inference|
| `HF_TOKEN`       | —                                            | Auth token for the HF inference router  |
| `MODEL_NAME`     | `Qwen/Qwen2.5-72B-Instruct`                  | Default model for `inference.py`        |
| `EPISODES_PER_TASK` | `5`                                       | Rollouts per task in `inference.py`     |

## License and acknowledgements

- Curated DNA and protein datasets derive from publicly available genomics resources.
- Environment interface and client types from [OpenEnv](https://github.com/open-source-ai/openenv).
- Training recipe inspired by [BioReason](https://arxiv.org/abs/2505.14028) (Arc Institute) and [DeepSeek-R1](https://arxiv.org/abs/2501.12948)'s use of GRPO for reasoning chains.

MIT License.
