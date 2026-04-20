Bioresearch OpenEnv — Implementation Plan
Current State Assessment
The project is an OpenEnv scaffold with a placeholder echo environment. Nothing biology-related is wired in yet:

- BioresearchEnvironment echoes messages back; reward = len(message) 0.99
- BioresearchAction has a single message: str field; BioresearchObservation has echoed_message + message_length
- inference.py is a copy-paste from a BrowserGym project — imports BrowserGymEnv, PIL, numpy, and doesn't reference this package at all
- client.py serialises/deserialises the echo payload
- data/DNA_reasoning.json (100 rows) and data/Protien_reasoning.json (100 rows) exist but are unused
- No tests, no graders, no task definitions
  What must change: Models, environment, client, inference, README, openenv.yaml tasks, Dockerfile deps — essentially the entire domain layer.

Environment Design (Four Tasks)  
The three tasks map directly to environments.md and use the two curated datasets.  
Task 1 — DNA Mutation Disease Classification  
Source: DNA_reasoning.json · Maps to: "DNA Mutation Effect Predictor"  
Aspect Detail  
Input (observation) Chromosome number, pathway network definition, gene list, question prompt, reference sequence, variant sequence  
Expected output (action) A disease name string (e.g. "parkinson's disease")

Episode length Single-step: agent sees observation, submits one action, episode ends  
Why "easy" The question explicitly asks "what disease does this contribute to?" — a classification task with a finite label set (26 diseases in the dataset). The pathway context and gene names are strong hints.  
Task 2 — DNA Mutation Biological Reasoning  
Source: DNA_reasoning.json · Maps to: "Reasoning biological model"  
Aspect Detail  
Input (observation) Same as Task 1  
Expected output (action) A JSON object with two fields: answer (disease name) and reasoning (multi-step biological reasoning chain)

Episode length Single-step  
Why "medium" The agent must not only identify the disease but articulate the biological mechanism (e.g. "PDE11A loss-of-function → elevated cAMP → PKA activation → cortisol overproduction → Cushing syndrome"). This requires genuine biological reasoning, not just pattern matching.  
Task 3 — Protein Function Hypothesis Generation  
Source: Protien_reasoning.json · Maps to: "Protein Function Hypothesis Generator"  
Aspect Detail  
Input (observation) Protein sequence, protein name, organism, sequence length. Optionally: InterPro domain hints (can be withheld for harder variant)  
Expected output (action) A JSON object with: function_description (free text), subcellular_location (predicted location), go_terms (list of predicted GO term IDs or names), reasoning (supporting evidence chain)  
Episode length Single-step  
Why "hard" Predicting protein function from sequence is an open research problem. The agent must synthesise structural, evolutionary, and functional cues. The GO term space is enormous (>40k terms), making random guessing ineffective. Even frontier models struggle with precise GO annotations.  
Action / Observation / Reward Models  
BioresearchAction (new)  
class BioresearchAction(Action):  
task_id: str # which task is being answered  
answer: str # disease name (T1/T2) or function description (T3)  
reasoning: Optional[str] = None # biological reasoning chain (required for T2/T3)  
go_terms: Optional[List[str]] = None # predicted GO terms (T3)  
subcellular_location: Optional[str] = None # predicted location (T3)  
BioresearchObservation (new)  
class BioresearchObservation(Observation):  
task_id: str # unique ID for this problem instance  
task_type: str # "dna_classification" | "dna_reasoning" | "protein_function"  
question: str # the question/prompt for the agent  
sequence_data: Dict[str, str] # reference_sequence, variant_sequence (DNA) or sequence (protein)  
context: Dict[str, Any] # pathway info, gene list, organism, etc.  
Reward Design  
All tasks return a composite float in [0.01,0.99]. Partial credit is always available:

Implementation Phases
Phase 1: Data Layer & Models
Files: models.py, server/data_loader.py (new), server/graders.py (new)

1. Rewrite models.py with the new BioresearchAction and BioresearchObservation Pydantic models as described above.
2. Create server/data_loader.py:

- Load DNA_reasoning.json and Protien_reasoning.json at server startup.
- Parse into typed dataclass lists (DNASample, ProteinSample).
- Provide get_random_sample(task_type) and get_sample_by_id(task_id) methods.
- Split data into pools: use 80 samples for episodes, hold out 20 for reproducible baseline evaluation.

3. Create server/graders.py:
   _ grade_dna_classification(predicted: str, gold: str) -> float — case-insensitive, strip punctuation, Jaccard token overlap with exact-match bonus.
   _ grade\*dna_reasoning(predicted_answer: str, predicted_reasoning: str, gold_answer: str, gold_reasoning: str) -> Tuple[float, Dict] — answer match + concept extraction scoring.

- grade\*protein_function(action: BioresearchAction, gold: ProteinSample) -> Tuple[float, Dict] — multi-dimensional scoring as described.
- All graders return both a score and a breakdown dict for transparency.
  Phase 2: Environment Core
  Files: server/bioresearch_environment.py

4. Rewrite BioresearchEnvironment:

- Constructor accepts a task_type parameter (or defaults to random selection).
- reset(task_type=None) → samples a problem from the appropriate dataset, constructs the observation, returns it. Store the gold answer internally for grading.
- step(action: BioresearchAction) → call the appropriate grader, compute reward, return observation with done=True (single-step episodes).
- state property → return State with episode_id, step_count, task_type, task_id.

5. Task registry: maintain a dict mapping task_type → (dataset, grader_fn, observation_builder) so adding future tasks is trivial.
6. Deterministic mode: support a seed parameter on reset() for reproducible evaluation (always returns the same sample sequence given the same seed).
   Phase 3: Client Update
   Files: client.py, **init**.py
7. Update BioresearchEnv.step_payload to serialise the new BioresearchAction fields.
8. Update BioresearchEnv.parse_result to deserialise the new BioresearchObservation fields.
9. Update **init**.py exports if any new public types are added.
   Phase 4: Inference Script
   Files: inference.py
10. Rewrite from scratch for the bioresearch domain:

- Use BioresearchEnv client (not BrowserGym).
  - Read API_BASE_URL, MODEL_NAME, HF_TOKEN from environment variables.
  - Use OpenAI client for LLM calls.
  - Run all 3 tasks sequentially (configurable subset via env var).
  - For each task: reset() → build a prompt from the observation → call LLM → parse response into BioresearchAction → step() → log result.
  - Follow the mandatory stdout protocol: [START], [STEP], [END].
  - Run N episodes per task (default 5) and report mean scores.

11. Prompt engineering:

- Task 1 system prompt: "You are a genomics expert. Given the DNA variant and pathway context, identify the disease."
  - Task 2 system prompt: "You are a genomics expert. Identify the disease AND explain the biological mechanism step-by-step."
  - Task 3 system prompt: "You are a protein biologist. Given the protein sequence and metadata, predict its function, subcellular location, and GO terms with reasoning."

12. Response parsing: extract structured JSON from LLM output, with fallback handling for malformed responses.
    Phase 5: Configuration & Deployment
    Files: openenv.yaml, pyproject.toml, Dockerfile, README.md, server/requirements.txt
13. Update openenv.yaml: add task definitions with names and descriptions for all 3 tasks.
14. Update pyproject.toml: add openai as a dependency (needed for inference). Remove stale comments.
15. Update Dockerfile: ensure data files are copied, deps are installed. No additional system-level dependencies needed (no numpy/PIL since we're not doing image processing).
16. Update server/requirements.txt: keep in sync with pyproject.toml.
17. Rewrite README.md:
    _ Environment description and motivation (biological reasoning evaluation).
    _ Action and observation space definitions (with field descriptions).
    _ Task descriptions with expected difficulty and sample inputs.
    _ Setup instructions (local, Docker, HF Spaces). Baseline scores table.
    Phase 6: Testing & Validation
18. Unit tests (tests/test_graders.py):

- Test each grader with known inputs and expected scores.
  - Test edge cases: empty input, extremely long input, malformed JSON.

19. Integration tests (tests/test_environment.py):

- Test reset() returns valid observation for each task type.
  - Test step() with correct answer returns high reward.
  - Test step() with wrong answer returns low reward.
  - Test state management across episodes.

20. End-to-end validation:

- openenv validate must pass.
  - docker build && docker run must work.
  - inference.py must produce scores on all 3 tasks.

21. Baseline scoring: run inference against a frontier model, record scores in README.

File Change Summary
File Action Description
models.py Rewrite New BioresearchAction and BioresearchObservation with biology-specific fields
server/bioresearch_environment.py Rewrite Task-aware environment with data loading, grading, proper state management
server/data_loader.py Create Data loading, parsing, sampling from JSON datasets
server/graders.py Create Grading functions for all 3 tasks with partial credit
client.py Update Serialise/deserialise new action/observation models
**init**.py Update Export new types if needed
inference.py Rewrite Bioresearch-specific inference with OpenAI client, 3 tasks, stdout protocol
server/app.py Minor update Should work as-is since it references models/env by class name
openenv.yaml Update Add task definitions
pyproject.toml Update Add openai dependency
Dockerfile Verify Ensure data/ is copied; may need minor tweaks
README.md Rewrite Full documentation for the bio environment
server/requirements.txt Update Sync with pyproject.toml
tests/test_graders.py Create Unit tests for grading functions
tests/test_environment.py Create Integration tests for environment

Risks & Mitigations
Risk Impact Mitigation
LLM responses don't parse as valid JSON Inference fails silently Robust fallback parsing: try JSON, then regex extraction, then score 0.01 with error logged
DNA sequences are very long (>3000 chars) Token limits exceeded Truncate sequences in observation to first/last 500 chars with [...] marker; full sequence available on request
GO term matching is too strict Scores artificially low Accept both GO IDs (GO:0005737) and term names (cytoplasm); normalise before comparison
Single-step episodes feel simplistic Lower "environment design" score Justified by domain: biological analysis is inherently single-query. The complexity is in the reasoning, not the interaction length. Document this rationale in README.
Dataset has only 100 samples per task Overfitting risk during evaluation Hold out 20 samples for baseline; shuffle with seed for reproducibility; document limitation
Implementation Order
Phase 1 (Data & Models) → Can be fully tested standalone
Phase 2 (Environment Core) → Depends on Phase 1
Phase 3 (Client Update) → Depends on Phase 1 models
Phase 4 (Inference Script) → Depends on Phase 2 + 3
Phase 5 (Config & Deploy) → Depends on Phase 2
Phase 6 (Testing) → Runs alongside Phases 2-5
Phases 1 → 2 → (3 + 5 in parallel) → 4 → 6 is the critical path.
