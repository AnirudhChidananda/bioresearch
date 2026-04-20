# Bioresearch OpenEnv — Implementation Plan

---

## 1. Current State Assessment

The project is an **OpenEnv scaffold** with a placeholder echo environment. Nothing biology-related is wired in yet:

| Component | Current State | Status |
|-----------|--------------|--------|
| `models.py` | `BioresearchAction` has a single `message: str` field; `BioresearchObservation` has `echoed_message` + `message_length` | Needs full rewrite |
| `server/bioresearch_environment.py` | Echoes messages back; `reward = len(message) * 0.1` | Needs full rewrite |
| `inference.py` | Copy-paste from a BrowserGym project — imports `MyEnvV4Env`, `MyEnvV4Action`; doesn't reference this package | Needs full rewrite |
| `client.py` | Serialises/deserialises the echo payload | Needs update |
| `data/DNA_reasoning.json` | 100 rows with fields: `question`, `answer`, `reasoning`, `reference_sequence`, `variant_sequence` | Ready to use |
| `data/Protein_reasoning.json` | 100 rows with fields: `protein_id`, `protein_names`, `protein_function`, `organism`, `length`, `subcellular_location`, `sequence`, `go_ids`, `go_bp`, `go_mf`, `go_cc`, `interpro_ids`, `interpro_formatted`, `ppi_formatted`, `go_pred` | Ready to use |
| `openenv.yaml` | Minimal — no task definitions | Needs update |
| `Dockerfile` | Generic OpenEnv Dockerfile | Needs verification + minor tweaks |
| Tests / Graders | None exist | Needs creation |
| Gradio UI Playground | Does not exist | Needs creation |

**What must change**: Models, environment, graders, data loader, client, inference, README, `openenv.yaml` tasks, Dockerfile deps, and a Gradio playground — essentially the entire domain layer plus a testing UI.

---

## 2. GRPO Alignment & Reward Design Philosophy

This environment is designed to be compatible with **GRPO (Group Relative Policy Optimization)**, the reinforcement learning method used by BioReason to train biological reasoning traces. The BioReason team uses GRPO to fine-tune models on high-quality step-by-step reasoning traces verified by expert biologists.

### How GRPO Works (and what the environment must support)

GRPO samples **K completions** for the **same prompt**, scores each with a reward, then computes advantages *relative to the group*:

```
advantage_i = (reward_i - mean(group_rewards)) / std(group_rewards)
```

The policy is updated using these relative advantages — no value function needed.

### Environment Design Principles for GRPO

| GRPO Requirement | How This Environment Satisfies It |
|-----------------|----------------------------------|
| **Same-prompt replay** | `reset(task_id=<specific_id>)` returns the exact same problem instance every time. GRPO can sample K completions for the same observation. |
| **High reward variance** | Grading produces a smooth, continuous score in [0.01, 0.99]. Different quality responses to the same prompt yield meaningfully different scores (see variance analysis below). |
| **Process reward over outcome reward** | Tasks 2, 3, and 4 all weight *reasoning quality* at 40–60% of the total score. Reasoning is graded at the **step level** — each step in the chain is evaluated for biological validity, not just the final answer. |
| **Deterministic grading** | Same (prompt, response) pair always produces the same reward. No stochastic grading components. |
| **Dense signal** | Partial credit at every level: partial disease name match, partial concept coverage, partial GO term overlap, partial reasoning chain quality. No binary pass/fail. |
| **Smooth reward landscape** | Small improvements in reasoning quality → small improvements in score. The grading functions use continuous similarity metrics (Jaccard, F1, coverage ratios), not hard thresholds. |

### Expected Reward Variance (per task)

For GRPO to compute useful advantages, different quality completions must produce sufficiently different scores. Expected distributions:

| Task | Poor Response | Mediocre Response | Good Response | Excellent Response |
|------|--------------|-------------------|---------------|-------------------|
| T1 (Classification) | 0.01 (wrong disease) | 0.25 (related disease) | 0.70 (partial match) | 0.90 (exact match) |
| T2 (Reasoning) | 0.05 (wrong + no reasoning) | 0.30 (right disease, weak reasoning) | 0.60 (right disease, decent reasoning) | 0.90 (right + full mechanism chain) |
| T3 (Protein Function) | 0.05 (wrong function) | 0.25 (vague function, no GO terms) | 0.55 (correct function, some GO terms) | 0.85 (full function + location + GO + reasoning) |
| T4 (Evidence Ranking) | 0.05 (wrong ranking, no elimination) | 0.30 (correct #1, poor elimination) | 0.60 (correct ranking, decent elimination) | 0.88 (perfect ranking + full evidence chain) |

This variance profile ensures GRPO can discriminate between response qualities within each group.

### Step-Level Reasoning Grading (GRPO-Critical)

Unlike simple concept-counting, the reasoning grader decomposes the agent's chain into individual steps and evaluates each:

1. **Step detection**: Split reasoning on "Step N:", numbered bullets, or sentence boundaries
2. **Per-step biological validity**: Does each step contain a real biological mechanism? (gene → protein → pathway → phenotype)
3. **Step-to-step coherence**: Does step N logically follow from step N-1? (causal chain, not random facts)
4. **Pathway coverage**: Do the steps collectively trace through the genes in the network definition?
5. **Terminal conclusion**: Does the final step connect to the correct disease/function?

This decomposition means two responses that both get the right answer can still receive very different reasoning scores — exactly what GRPO needs to optimise the reasoning *process*.

---

## 3. Environment Design (Four Tasks)

The four tasks use the two curated datasets and increase in difficulty. They are designed to produce the reward variance profile that GRPO requires.

### Task 1 — DNA Mutation Disease Classification (Easy)

| Aspect | Detail |
|--------|--------|
| **Source** | `data/DNA_reasoning.json` |
| **Maps to** | "DNA Mutation Effect Predictor" (from `environments.md`) |
| **Input (observation)** | Chromosome number, pathway network definition, gene list, question prompt, truncated reference & variant sequences |
| **Expected output (action)** | A disease name string (e.g. `"cushing syndrome"`) |
| **Episode length** | Single-step: agent sees observation, submits one action, episode ends |
| **Why "easy"** | The question explicitly asks "what disease does this contribute to?" — a classification task with a finite label set (~26 diseases in the dataset). The pathway context and gene names are strong hints. |
| **GRPO role** | Baseline task. Reward variance comes from exact vs. partial vs. wrong disease match. Useful for initial policy warm-up before harder tasks. |

**Grading (`grade_dna_classification`)**:
- Case-insensitive, strip punctuation, normalise whitespace
- Exact match → `0.90`
- Jaccard token overlap > 0.5 → `0.50 + (overlap * 0.40)`
- Partial keyword match (any disease token found) → `0.20–0.40`
- No match → `0.01`

### Task 2 — DNA Mutation Biological Reasoning (Medium)

| Aspect | Detail |
|--------|--------|
| **Source** | `data/DNA_reasoning.json` |
| **Maps to** | "Reasoning biological model" (from `environments.md`). Directly inspired by BioReason 1, which uses GRPO to train reasoning traces about DNA sequences. |
| **Input (observation)** | Same as Task 1 |
| **Expected output (action)** | A JSON object with two fields: `answer` (disease name) and `reasoning` (multi-step biological reasoning chain) |
| **Episode length** | Single-step |
| **Why "medium"** | The agent must not only identify the disease but articulate the biological mechanism (e.g. "PDE11A loss-of-function → elevated cAMP → PKA activation → cortisol overproduction → Cushing syndrome"). This requires genuine biological reasoning, the core skill BioReason's GRPO trains. |
| **GRPO role** | Primary reasoning-trace training task. The 60% reasoning weight means GRPO can optimise the step-by-step chain quality even when the answer is correct. Two correct-answer responses with different reasoning quality will get very different rewards. |

**Grading (`grade_dna_reasoning`)** — Composite score:
- **Answer accuracy (40%)**: Same logic as Task 1 grader
- **Reasoning quality (60%)** — Step-level decomposition:
  - **Step count and structure (10%)**: Are there identifiable numbered steps? Minimum 3 steps expected.
  - **Biological concept coverage (20%)**: Extract key concepts from gold reasoning (gene names, enzyme names, pathway terms like "cAMP", "PKA", "cortisol"). Score = `concepts_found / total_gold_concepts`.
  - **Pathway gene coverage (15%)**: What fraction of genes from the network definition does the reasoning mention? These are the specific genes the agent should trace through.
  - **Causal chain coherence (10%)**: Does the reasoning follow a gene → protein → pathway → phenotype → disease structure? Check for causal connectors ("leads to", "results in", "causes", "activates", "inhibits").
  - **Penalty for hallucination (−5%)**: Mentions gene names not present in the pathway context → deduct up to 5%.

### Task 3 — Protein Function Hypothesis Generation (Hard)

| Aspect | Detail |
|--------|--------|
| **Source** | `data/Protein_reasoning.json` |
| **Maps to** | "Protein Function Hypothesis Generator" (from `environments.md`). Inspired by BioReason-Pro (BioReason 2), which bridges the sequence-to-function gap. |
| **Input (observation)** | Protein sequence, protein name, organism, sequence length. Optionally: InterPro domain hints (can be withheld for harder variant) |
| **Expected output (action)** | A JSON object with: `function_description` (free text), `subcellular_location` (predicted location), `go_terms` (list of predicted GO term IDs or names), `reasoning` (supporting evidence chain) |
| **Episode length** | Single-step |
| **Why "hard"** | Predicting protein function from sequence is an open research problem. The agent must synthesise structural, evolutionary, and functional cues. The GO term space is enormous (>40k terms), making random guessing ineffective. Even frontier models struggle with precise GO annotations. |
| **GRPO role** | Multi-dimensional scoring creates a rich reward landscape. GRPO can independently optimise function description quality, GO term precision, location accuracy, and reasoning depth — all within a single reward signal. |

**Grading (`grade_protein_function`)** — Multi-dimensional composite:
- **Function description (25%)**: Semantic overlap with gold `protein_function` using keyword extraction + Jaccard similarity on biological terms
- **Subcellular location (20%)**: Exact or partial match against gold `subcellular_location` (hierarchical: "Cell membrane" ⊂ "Membrane")
- **GO term prediction (35%)**:
  - Compare predicted GO IDs/names against gold `go_ids`, `go_bp`, `go_mf`, `go_cc`
  - Accept both GO IDs (`GO:0005737`) and term names (`cytoplasm`)
  - Precision-recall F1 on matched terms
  - Partial credit for parent/child terms in the GO hierarchy (if feasible, otherwise exact match only)
- **Reasoning quality (20%)**: Concept coverage from gold function, domain mentions from `interpro_formatted`, interaction partner awareness from `ppi_formatted`

### Task 4 — Variant Pathogenicity Reasoning with Evidence Ranking (Medium-Hard) [NEW]

| Aspect | Detail |
|--------|--------|
| **Source** | `data/DNA_reasoning.json` (same data, different task framing) |
| **Maps to** | Advanced form of BioReason 1's variant effect prediction. Combines classification with structured elimination reasoning. |
| **Input (observation)** | Same pathway/mutation context as Tasks 1/2, PLUS a list of 4 candidate diseases: the correct one + 3 distractors sampled from other entries in the dataset |
| **Expected output (action)** | A JSON object with: `ranked_diseases` (ordered list, most likely first), `selected_disease` (the top pick), `elimination_reasoning` (dict mapping each rejected disease → why it was eliminated), `supporting_evidence` (reasoning chain for the selected disease) |
| **Episode length** | Single-step |
| **Why "medium-hard"** | The agent must do more than identify the correct disease — it must reason about *why each alternative is wrong* for this specific pathway. This requires comparing pathway mechanisms across diseases, a skill that even strong models find challenging. The distractor diseases are drawn from the same dataset, so they are biologically plausible (not random labels). |
| **GRPO role** | This is the **most GRPO-aligned task**. The structured elimination + ranking format produces maximal reward variance: correct ranking gives partial credit even if elimination reasoning is weak, good elimination reasoning gives partial credit even if ranking is imperfect, and the multi-disease comparison creates many independent scoring axes. GRPO can optimise each reasoning dimension independently. |

**Grading (`grade_evidence_ranking`)** — Composite score:
- **Ranking accuracy (30%)**:
  - Correct disease ranked #1 → `0.30`
  - Correct disease ranked #2 → `0.15`
  - Correct disease ranked #3 → `0.05`
  - Correct disease ranked #4 or absent → `0.00`
- **Elimination reasoning quality (35%)**:
  - For each of the 3 distractors: does the elimination reasoning correctly identify *why this disease doesn't match the given pathway*?
  - Score per distractor: mention of pathway mismatch (`0.04`), mention of gene mismatch (`0.04`), mention of mechanism difference (`0.04`) = max `0.12` per distractor × 3 = `0.35` (rounded)
  - Bonus: if the elimination references specific genes from the actual pathway that contradict the distractor
- **Supporting evidence quality (25%)**:
  - Same step-level reasoning grading as Task 2, applied to the supporting evidence for the selected disease
- **Logical consistency (10%)**:
  - No contradictions between elimination reasoning and supporting evidence
  - Reasoning for selected disease doesn't accidentally support a rejected disease
  - All 4 candidates are addressed (no omissions)

**Distractor Selection Strategy**:
- Distractors are sampled from other entries in `DNA_reasoning.json` to ensure they are real diseases with real pathways
- Prefer diseases from different pathway families (e.g., don't pair two cAMP-pathway diseases as distractor and target)
- Deterministic distractor selection per `task_id` (hash-based) so same problem always has same distractors — required for GRPO same-prompt replay

---

## 4. Action / Observation / Reward Models

### `BioresearchAction` (new)

```python
class BioresearchAction(Action):
    task_id: str                                    # which task instance is being answered
    answer: str                                     # disease name (T1/T2/T4) or function description (T3)
    reasoning: Optional[str] = None                 # biological reasoning chain (T2/T3/T4)
    go_terms: Optional[List[str]] = None            # predicted GO terms (T3 only)
    subcellular_location: Optional[str] = None      # predicted location (T3 only)
    ranked_diseases: Optional[List[str]] = None     # ordered disease ranking (T4 only)
    elimination_reasoning: Optional[Dict[str, str]] = None  # disease → why eliminated (T4 only)
```

### `BioresearchObservation` (new)

```python
class BioresearchObservation(Observation):
    task_id: str                        # unique ID for this problem instance
    task_type: str                      # "dna_classification" | "dna_reasoning" | "protein_function" | "evidence_ranking"
    question: str                       # the question/prompt for the agent
    sequence_data: Dict[str, str]       # reference_sequence, variant_sequence (DNA) or sequence (protein)
    context: Dict[str, Any]             # pathway info, gene list, organism, interpro hints, etc.
    candidate_diseases: Optional[List[str]] = None  # 4 candidate diseases for T4 (correct + 3 distractors)
```

### Reward Design

All tasks return a composite float in **[0.01, 0.99]**. Partial credit is always available. Process (reasoning) is weighted more heavily than outcome (answer) in Tasks 2–4, aligned with GRPO's strength in optimising reasoning traces.

| Component | Task 1 (Easy) | Task 2 (Medium) | Task 3 (Hard) | Task 4 (Medium-Hard) |
|-----------|--------------|-----------------|---------------|---------------------|
| Answer / ranking accuracy | 100% | 40% | 25% (function) | 30% (ranking) |
| Reasoning quality | — | 60% (step-level) | 20% | 25% (supporting evidence) |
| Elimination reasoning | — | — | — | 35% |
| Subcellular location | — | — | 20% | — |
| GO term prediction | — | — | 35% | — |
| Logical consistency | — | — | — | 10% |
| **Process reward %** | **0%** | **60%** | **40%** | **70%** |
| **Clamp range** | [0.01, 0.99] | [0.01, 0.99] | [0.01, 0.99] | [0.01, 0.99] |

Note: The **process reward %** row shows what fraction of the total score comes from reasoning quality rather than factual accuracy. Tasks 2 and 4 are the most GRPO-friendly, as they strongly reward the reasoning process.

---

## 5. GRPO Training Loop Compatibility

While this OpenEnv environment is primarily an *evaluation* harness, it is designed so a GRPO training loop can use it directly:

### Same-Prompt Replay Protocol

```python
# GRPO sampling: K completions for same problem
observation = env.reset(task_id="dna_042")  # always returns the same problem
for k in range(K):
    completion_k = model.generate(observation)
    reward_k = env.step(completion_k).reward
    env.reset(task_id="dna_042")  # replay same problem

# Compute group advantage
advantages = (rewards - rewards.mean()) / rewards.std()
```

The `reset(task_id=...)` method guarantees identical observations, and the deterministic grader guarantees identical scoring for identical responses.

### Batch Evaluation Support

The data loader exposes `get_all_sample_ids(task_type)` so a training loop can iterate over all problem instances systematically rather than random sampling:

```python
for task_id in data_loader.get_all_sample_ids("dna_reasoning"):
    obs = env.reset(task_id=task_id)
    # ... sample K completions and grade them ...
```

### Reward Properties for Stable GRPO Training

- **No reward hacking**: Graders check biological validity, not just keyword density. Repeating the same gene name 100 times won't increase the concept coverage score.
- **Anti-hallucination penalty**: Mentioning genes not in the pathway context deducts score, preventing reward gaming through fabricated reasoning.
- **Score ceiling < 1.0**: Maximum achievable score is 0.99 (clamped), preventing overconfident advantage estimates.
- **Score floor > 0.0**: Minimum score is 0.01 (clamped), ensuring every response contributes to the group statistics.

---

## 6. Implementation Phases

### Phase 1: Data Layer & Models
**Files**: `models.py`, `server/data_loader.py` (new), `server/graders.py` (new)

1. **Rewrite `models.py`** with the new `BioresearchAction` and `BioresearchObservation` Pydantic models as described above, including Task 4 fields (`ranked_diseases`, `elimination_reasoning`, `candidate_diseases`).

2. **Create `server/data_loader.py`**:
   - Load `DNA_reasoning.json` and `Protein_reasoning.json` at server startup
   - Parse into typed dataclass lists (`DNASample`, `ProteinSample`)
   - Provide `get_random_sample(task_type)` and `get_sample_by_id(task_id)` methods
   - Provide `get_all_sample_ids(task_type)` for GRPO batch iteration
   - **Distractor sampling for Task 4**: `get_distractors(task_id, n=3)` returns deterministic distractors (hash-based selection from other disease entries, preferring different pathway families)
   - Split data into pools: use 80 samples for episodes, hold out 20 for reproducible baseline evaluation
   - Handle the HuggingFace dataset JSON format (features + rows structure)

3. **Create `server/graders.py`**:
   - `grade_dna_classification(predicted: str, gold: str) -> Tuple[float, Dict]`
   - `grade_dna_reasoning(predicted_answer: str, predicted_reasoning: str, gold_answer: str, gold_reasoning: str, pathway_genes: List[str]) -> Tuple[float, Dict]`
     - Includes step-level reasoning decomposition as described in Section 2
   - `grade_protein_function(action: BioresearchAction, gold: ProteinSample) -> Tuple[float, Dict]`
   - `grade_evidence_ranking(action: BioresearchAction, gold_disease: str, gold_reasoning: str, distractors: List[str], pathway_genes: List[str]) -> Tuple[float, Dict]` [NEW]
   - All graders return both a score and a breakdown dict for transparency
   - All scores clamped to `[0.01, 0.99]`

### Phase 2: Environment Core
**Files**: `server/bioresearch_environment.py`

4. **Rewrite `BioresearchEnvironment`**:
   - Constructor loads data via `DataLoader` and initialises empty state
   - `reset(task_type=None, task_id=None)` → If `task_id` is provided, loads that exact problem (GRPO same-prompt replay). If only `task_type`, samples randomly. If neither, samples random task type and problem.
   - `step(action: BioresearchAction)` → call the appropriate grader, compute reward, return observation with `done=True` (single-step episodes)
   - `state` property → return `State` with `episode_id`, `step_count`, `task_type`, `task_id`

5. **Task registry**: maintain a dict mapping `task_type → (dataset, grader_fn, observation_builder)` so adding future tasks is trivial.

6. **Deterministic mode**: `reset(task_id=...)` always returns the exact same observation. Essential for GRPO group sampling.

7. **Sequence truncation**: DNA sequences can be >3000 chars. Truncate to first/last 500 chars with `[...]` marker in the observation. Full sequence available in `context["full_reference_sequence"]` and `context["full_variant_sequence"]` if needed.

8. **Task 4 observation builder**: On reset for `evidence_ranking`, call `data_loader.get_distractors(task_id, n=3)` and include the 4 candidate diseases (shuffled, deterministic order per task_id) in `observation.candidate_diseases`.

### Phase 3: Client Update
**Files**: `client.py`, `__init__.py`

9. **Update `BioresearchEnv._step_payload`** to serialise the new `BioresearchAction` fields, including Task 4 fields (`ranked_diseases`, `elimination_reasoning`).

10. **Update `BioresearchEnv._parse_result`** to deserialise the new `BioresearchObservation` fields, including `candidate_diseases`.

11. **Update `__init__.py`** exports if any new public types are added.

### Phase 4: Inference Script
**Files**: `inference.py`

12. **Rewrite from scratch** for the bioresearch domain:
    - Use `BioresearchEnv` client (not BrowserGym)
    - Read `API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN` from environment variables
    - Use OpenAI client for LLM calls
    - Run all **4 tasks** sequentially (configurable subset via env var)
    - For each task: `reset()` → build a prompt from the observation → call LLM → parse response into `BioresearchAction` → `step()` → log result
    - Follow the mandatory stdout protocol: `[START]`, `[STEP]`, `[END]`
    - Run N episodes per task (default 5) and report mean scores
    - Total runtime must be < 20 minutes; target < 10 minutes

13. **Prompt engineering**:
    - **Task 1** system prompt: "You are a genomics expert. Given the DNA variant and pathway context, identify the disease. Reply with only the disease name."
    - **Task 2** system prompt: "You are a genomics expert. Identify the disease AND explain the biological mechanism step-by-step. Reply with JSON: {\"answer\": \"...\", \"reasoning\": \"Step 1: ... Step 2: ...\"}"
    - **Task 3** system prompt: "You are a protein biologist. Given the protein sequence and metadata, predict its function, subcellular location, and GO terms with reasoning. Reply with JSON: {\"function_description\": \"...\", \"subcellular_location\": \"...\", \"go_terms\": [...], \"reasoning\": \"...\"}"
    - **Task 4** system prompt: "You are a genomics expert. Given the DNA variant, pathway context, and 4 candidate diseases, rank the candidates from most to least likely. For each rejected disease, explain why this pathway does NOT lead to it. For your top pick, provide a step-by-step biological mechanism. Reply with JSON: {\"ranked_diseases\": [...], \"selected_disease\": \"...\", \"elimination_reasoning\": {\"disease_name\": \"why eliminated\", ...}, \"supporting_evidence\": \"Step 1: ...\"}"

14. **Response parsing**: extract structured JSON from LLM output, with fallback handling for malformed responses (try JSON → regex extraction → score 0.01 with error logged).

### Phase 5: Configuration & Deployment
**Files**: `openenv.yaml`, `pyproject.toml`, `Dockerfile`, `README.md`, `server/requirements.txt`

15. **Update `openenv.yaml`**: add task definitions with names and descriptions for all 4 tasks:
    ```yaml
    tasks:
      - name: dna_classification
        description: "Classify the disease caused by a DNA mutation given pathway context"
        difficulty: easy
      - name: dna_reasoning
        description: "Identify disease and explain the biological mechanism step-by-step"
        difficulty: medium
      - name: evidence_ranking
        description: "Rank 4 candidate diseases with elimination reasoning and supporting evidence"
        difficulty: medium-hard
      - name: protein_function
        description: "Predict protein function, subcellular location, and GO terms from sequence"
        difficulty: hard
    ```

16. **Update `pyproject.toml`**: add `openai` as a dependency. Remove stale comments about numpy/torch.

17. **Update Dockerfile**: ensure `data/` directory is copied, all deps are installed. Verify the multi-stage build still works. No additional system-level dependencies needed.

18. **Update `server/requirements.txt`**: add `openai` and keep in sync with `pyproject.toml`.

19. **Rewrite `README.md`**:
    - Environment description and motivation (biological reasoning evaluation for drug discovery and genomics)
    - GRPO compatibility section explaining how the environment supports RL training
    - Action and observation space definitions (with field descriptions and examples)
    - Task descriptions with expected difficulty, sample inputs, and grading criteria
    - Setup instructions (local via `uv`, Docker, HF Spaces)
    - Baseline scores table (filled in after Phase 7)

### Phase 6: Gradio UI Playground
**Files**: `playground.py` (new)

This is a **Gradio-based interactive testing UI** that lets users manually interact with the environment and visualise results. It runs **after** the OpenEnv server is up and connects to it as a client.

20. **Create `playground.py`** with the following layout:

    **Tab 1 — Interactive Environment**
    - Dropdown to select task type (`dna_classification` / `dna_reasoning` / `evidence_ranking` / `protein_function`)
    - "Reset" button → calls `env.reset()` → displays the observation (question, sequence data, context, candidate diseases for T4) in formatted panels
    - Input fields for the agent's response (dynamic based on task type)
    - "Submit Action" button → calls `env.step()` → displays reward, score breakdown, done status
    - Colour-coded reward indicator (red < 0.3, yellow 0.3–0.7, green > 0.7)
    - **Score breakdown panel**: shows per-component scores from the grader's breakdown dict (e.g., "Answer accuracy: 0.36 | Reasoning steps: 0.42 | Pathway coverage: 0.15")

    **Tab 2 — Automated Inference**
    - Text input for OpenAI API key (or auto-read from env)
    - Model name input
    - Number of episodes per task
    - "Run Inference" button → runs the inference loop in a background thread
    - Live-updating log output (streaming the `[START]`, `[STEP]`, `[END]` lines)
    - Final scores table with per-task breakdown

    **Tab 3 — Dataset Explorer**
    - Browse DNA and Protein samples
    - View full question, gold answer, gold reasoning
    - View raw sequences (with copy button)
    - Filter by disease / organism / protein name

    **Tab 4 — GRPO Reward Analysis** [NEW]
    - Select a problem instance by task_id
    - Submit multiple responses and compare their scores side-by-side
    - Visualise score components as stacked bar charts
    - Shows how GRPO would compute advantages from the group of responses
    - Useful for debugging reward design and verifying sufficient variance

21. **Add `gradio` to `pyproject.toml`** as an optional dependency:
    ```toml
    [project.optional-dependencies]
    playground = ["gradio>=4.0.0"]
    ```

22. **Startup flow**: The playground connects to a running OpenEnv server via `BioresearchEnv(base_url="http://localhost:8000")`. The user must start the server first (`uvicorn server.app:app`), then run `python playground.py`.

### Phase 7: Testing & Validation

23. **Unit tests** (`tests/test_graders.py`):
    - Test each grader (all 4) with known inputs and expected scores
    - Test edge cases: empty input, extremely long input, malformed JSON
    - Verify all scores are in `[0.01, 0.99]`
    - **GRPO variance test**: for each task, submit 5 responses of varying quality to the same problem and verify score spread > 0.4

24. **Integration tests** (`tests/test_environment.py`):
    - Test `reset()` returns valid observation for each task type
    - Test `reset(task_id=X)` returns identical observation on repeated calls (GRPO same-prompt replay)
    - Test `step()` with correct answer returns high reward (> 0.7)
    - Test `step()` with wrong answer returns low reward (< 0.3)
    - Test Task 4 observation includes `candidate_diseases` with 4 entries
    - Test state management across episodes

25. **End-to-end validation**:
    - `openenv validate` must pass
    - `docker build && docker run` must work
    - `inference.py` must produce scores on all 4 tasks
    - Verify stdout format matches `[START]` / `[STEP]` / `[END]` spec exactly

26. **Baseline scoring**: run inference against a frontier model (e.g. `Qwen/Qwen2.5-72B-Instruct`), record scores in README.

---

## 7. File Change Summary

| File | Action | Description |
|------|--------|-------------|
| `models.py` | **Rewrite** | New `BioresearchAction` and `BioresearchObservation` with biology-specific fields + T4 fields |
| `server/bioresearch_environment.py` | **Rewrite** | Task-aware environment with data loading, grading, GRPO same-prompt replay |
| `server/data_loader.py` | **Create** | Data loading, parsing, sampling, distractor selection for T4 |
| `server/graders.py` | **Create** | Grading functions for all 4 tasks with step-level reasoning scoring |
| `client.py` | **Update** | Serialise/deserialise new action/observation models including T4 |
| `__init__.py` | **Update** | Export new types if needed |
| `inference.py` | **Rewrite** | Bioresearch-specific inference with OpenAI client, 4 tasks, stdout protocol |
| `playground.py` | **Create** | Gradio UI playground with GRPO reward analysis tab |
| `server/app.py` | **Minor update** | Should work as-is since it references models/env by class name |
| `openenv.yaml` | **Update** | Add task definitions for 4 tasks |
| `pyproject.toml` | **Update** | Add `openai`, `gradio` dependencies |
| `Dockerfile` | **Verify** | Ensure `data/` is copied; may need minor tweaks |
| `README.md` | **Rewrite** | Full documentation including GRPO compatibility |
| `server/requirements.txt` | **Update** | Sync with `pyproject.toml` |
| `tests/test_graders.py` | **Create** | Unit tests for 4 grading functions + GRPO variance tests |
| `tests/test_environment.py` | **Create** | Integration tests including same-prompt replay |

---

## 8. Data Strategy

### Existing Data (sufficient for all 4 tasks)
- **DNA_reasoning.json**: 100 samples covering ~26 diseases (Cushing syndrome, Parkinson's, ALS, etc.)
  - Fields: `question`, `answer`, `reasoning`, `reference_sequence`, `variant_sequence`
  - Used by: Task 1 (classification), Task 2 (reasoning), Task 4 (evidence ranking — same data, different framing + distractors)

- **Protein_reasoning.json**: 100 samples covering diverse proteins across species
  - Fields: `protein_id`, `protein_names`, `protein_function`, `organism`, `length`, `subcellular_location`, `sequence`, `go_ids`, `go_bp`, `go_mf`, `go_cc`, `interpro_ids`, `interpro_formatted`, `ppi_formatted`, `go_pred`
  - Used by: Task 3 (protein function)

**No new data files needed for Task 4** — it reuses `DNA_reasoning.json` with a different observation framing (adds distractor diseases from other entries in the same dataset).

### Data Split
- **80 samples** for training/episodes (random sampling during reset)
- **20 samples** held out for reproducible baseline evaluation (deterministic seed)
- Split is deterministic based on `row_idx` (0–79 for episodes, 80–99 for baseline)

### Potential Additional Data (if needed)
If we need to strengthen any task, we can add verified datasets from:
- **ClinVar** — for more DNA variant → disease mappings (public domain, NCBI)
- **UniProt/Swiss-Prot** — for more curated protein function annotations (CC BY 4.0)
- **KEGG DISEASE** — for pathway-to-disease mappings (academic use)

For MVP, the existing 200 samples (100 DNA + 100 Protein) are sufficient for all 4 tasks.

---

## 9. Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| LLM responses don't parse as valid JSON | Inference fails silently | Robust fallback parsing: try JSON, then regex extraction, then score 0.01 with error logged |
| DNA sequences are very long (>3000 chars) | Token limits exceeded | Truncate sequences in observation to first/last 500 chars with `[...]` marker; full sequence available in context |
| GO term matching is too strict | Scores artificially low | Accept both GO IDs (`GO:0005737`) and term names (`cytoplasm`); normalise before comparison |
| Single-step episodes feel simplistic | Lower "environment design" score | Justified by domain: biological analysis is inherently single-query. The complexity is in the reasoning, not the interaction length. Document this rationale in README. |
| Dataset has only 100 samples per task | Overfitting risk during evaluation | Hold out 20 samples for baseline; shuffle with seed for reproducibility; document limitation |
| GRPO reward variance too low | Training signal collapses | Step-level reasoning grading + multi-component scoring ensures wide variance. Verified in unit tests (spread > 0.4). |
| Task 4 distractor selection creates trivially easy eliminations | Reduces task difficulty | Prefer distractors from related pathway families; avoid diseases that are obviously unrelated to the chromosome/gene context |
| Gradio playground blocks the main server | Poor UX | Run playground as a separate process that connects to the OpenEnv server via HTTP |
| Inference runtime exceeds 20 min limit | Disqualification | Limit to 5 episodes per task (now 4 tasks × 5 = 20 episodes), set conservative `max_tokens`, use efficient model |
| Resource constraints (2 vCPU, 8GB RAM) | OOM or slow response | No heavy dependencies (no torch, numpy); pure Python grading; lazy data loading |

---

## 10. Implementation Order (Critical Path)

```
Phase 1 (Data & Models)         → Can be fully tested standalone
    ↓
Phase 2 (Environment Core)      → Depends on Phase 1
    ↓
Phase 3 (Client Update)         → Depends on Phase 1 models
    ↓                               ↓
Phase 4 (Inference Script)      Phase 5 (Config & Deploy)    ← Can run in parallel
    ↓                               ↓
Phase 6 (Gradio Playground)     → Depends on Phases 2, 3, 4 (needs working env + inference)
    ↓
Phase 7 (Testing & Validation)  → Runs alongside Phases 2–6
```

**Critical path**: Phase 1 → Phase 2 → Phase 3 → (Phase 4 + Phase 5 in parallel) → Phase 6 → Phase 7

**Estimated effort per phase**:
- Phase 1: ~2.5 hours (models + data loader + 4 graders including step-level reasoning)
- Phase 2: ~2.5 hours (environment rewrite + GRPO same-prompt replay + T4 observation builder)
- Phase 3: ~30 min (client update)
- Phase 4: ~2.5 hours (inference rewrite + 4 task prompts + response parsing)
- Phase 5: ~1 hour (config, Dockerfile, README)
- Phase 6: ~2.5 hours (Gradio playground + GRPO analysis tab)
- Phase 7: ~2 hours (tests + GRPO variance validation)

**Total estimated**: ~13.5 hours

---

## 11. Pre-Submission Checklist

- [ ] `openenv validate` passes
- [ ] `docker build && docker run` works
- [ ] HF Space deploys and responds to `reset()`
- [ ] `inference.py` produces scores on all **4 tasks**
- [ ] 4 tasks with graders, scores in `[0.01, 0.99]`
- [ ] Graders are deterministic and reproducible
- [ ] Hard task genuinely challenges frontier models
- [ ] Stdout format matches `[START]` / `[STEP]` / `[END]` spec exactly
- [ ] `API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN` env vars are used correctly
- [ ] Runtime of inference < 20 minutes
- [ ] Environment runs on 2 vCPU, 8GB RAM
- [ ] `reset(task_id=X)` returns identical observation on repeated calls (GRPO replay)
- [ ] Reward variance > 0.4 across quality levels for each task (GRPO compatibility)
- [ ] Step-level reasoning grading correctly decomposes chains for Tasks 2, 3, 4
- [ ] Task 4 distractors are deterministic and biologically plausible
- [ ] README has environment description, GRPO compatibility, action/observation spaces, task descriptions, setup instructions, baseline scores
- [ ] Gradio playground works and connects to running environment
