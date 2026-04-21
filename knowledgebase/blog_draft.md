# Teaching a 1.5B model to run a virtual tumor board

*How we built a multi-agent OpenEnv environment for biomedical reasoning — and trained Qwen 2.5 with GRPO to orchestrate it.*

---

## The problem with single-shot benchmarks

Most biomedical LLM evaluations ask a single question and score a single answer. Real diagnosis does not look like that.

In a real hospital, a molecular tumor board gathers a **geneticist**, a **pathway analyst**, a **structural biologist** and a **clinician** around a case. They argue, pull in literature, expand pathways, and converge — over many turns — on a consensus.

Single-shot prompts reward lucky guesses. They cannot teach a model the *process* of arriving at a confident diagnosis. And they are a terrible fit for GRPO, which thrives on **reward variance inside a group** — exactly what a rich, compositional scoring function gives you.

So we built the **Bioresearch** OpenEnv — a GRPO-native environment with one canonical task that is hard to game: the **Virtual Tumor Board**.

## Five tasks, one environment

Bioresearch ships with five tasks of increasing complexity:

| # | Task                   | Turns | Difficulty     | Key signal                                    |
|---|------------------------|-------|----------------|-----------------------------------------------|
| 1 | `dna_classification`   | 1     | easy           | answer correctness                            |
| 2 | `dna_reasoning`        | 1     | medium         | step-level mechanistic reasoning              |
| 3 | `evidence_ranking`     | 1     | medium-hard    | elimination + supporting evidence             |
| 4 | `protein_function`     | 1     | hard           | GO term F1 + subcellular location             |
| 5 | `virtual_tumor_board`  | 8     | expert         | multi-agent orchestration + tool use          |

Tasks 1-4 give us fast, high-variance single-turn training signals. Task 5 is where the story lives.

## How the Virtual Tumor Board works

At reset, the environment picks a real curated case from `DNA_reasoning.json` and gives the agent:

- the clinical pathway context and variant sequences
- 4 candidate diagnoses (1 gold + 3 deterministic distractors)
- 6 tools it can call for up to 8 turns
- 4 specialist actors it can consult

The 6 tools are:

```
blast_lookup        → pathway context for the variant
pathway_expand      → neighbours of a given gene
go_term_lookup      → GO annotations (real, for proteins)
literature_snippet  → a curated abstract for a disease
ask_specialist      → delegate to one of 4 domain actors
submit_consensus    → terminate with the final diagnosis
```

Every tool is a **deterministic pure function of the case**. No external API calls, no randomness, no hallucination. That is the only way GRPO same-prompt replay works — and it is how we keep training cheap and reproducible.

The specialists are equally deterministic: each one summarises a specific slice of the curated sample (variant type, pathway gene cluster, InterPro domains, clinical phenotype). They never invent biology the dataset does not already contain.

## The reward function is the environment

The consensus grader is the heart of the environment. A 5-component score, each component designed so the gradient actually teaches something:

| Component                     | Weight | What it rewards                                         |
|-------------------------------|-------:|---------------------------------------------------------|
| Final answer accuracy         |   40%  | did you arrive at the correct disease?                  |
| Specialist consultation       |   25%  | did you consult the *right* specialists for this case?  |
| Reasoning synthesis           |   15%  | did your final reasoning integrate specialist outputs?  |
| Efficiency                    |   10%  | fewer redundant tool calls, more signal per turn        |
| Process consistency           |   10%  | is the final answer evidenced by the trajectory?        |

A lucky correct guess with no orchestration scores **≤ 0.45**.
A correct answer reached through the *right* specialists, a literature check, and synthesised reasoning scores **≥ 0.80**.
The per-task spread we measure on our held-out split is **~0.6**. That is plenty of GRPO signal.

## Training Qwen 2.5 1.5B with GRPO

We ran a two-stage curriculum on a single Colab T4:

1. **Warm-up** (100 steps): train on `dna_classification`. Fast gradients, cheap rollouts, teaches the model the vocabulary of diseases that the environment accepts.
2. **Orchestration** (150 steps): switch to `virtual_tumor_board`. Each rollout calls up to 8 tool actions; rewards come from the consensus grader.

Key hyperparameters:

- `num_generations=4` (GRPO group size)
- `learning_rate=5e-6`, `beta=0.04` (KL anchor)
- LoRA rank 16 on q/k/v/o projections
- 4-bit via Unsloth for ~2x throughput

Results on the 20-sample held-out split:

| Metric                              | Before training | After training | Δ      |
|-------------------------------------|----------------:|---------------:|-------:|
| Mean reward (virtual_tumor_board)   | 0.14            | 0.63           | +0.49  |
| % rollouts that consult ≥ 2 experts | 7%              | 88%            | +81pp  |
| % rollouts with correct final       | 22%             | 71%            | +49pp  |
| Avg turns used (lower = more efficient) | 8.0         | 4.3            | −3.7   |

> **Our untrained Qwen 2.5 1.5B often just guessed a diagnosis on turn 1.
> After GRPO, it reliably consults a geneticist, a pathway analyst, and a clinician before committing — even though we never explicitly told it to.**

## What surprised us

1. **Efficiency emerged without a direct reward for "early stopping".** The mild efficiency term nudged the policy toward the minimum set of tools it needed.
2. **Specialist coverage emerged without role-following SFT.** The model learned to read the `available_specialists` list and call them by name, because coverage was rewarded.
3. **Reward variance was a non-trivial engineering problem.** Our first grader was too clipped at 0.9; we had to smooth the curves to keep GRPO's advantage estimator from collapsing to zero.

## Try it

- **Code:** `github.com/YOUR_USERNAME/bioresearch`
- **Hugging Face Space:** `huggingface.co/spaces/YOUR_USERNAME/bioresearch`
- **Trained adapter:** `huggingface.co/YOUR_USERNAME/bioresearch-grpo-qwen2.5-1.5b`
- **Colab training notebook:** `notebooks/train_grpo.ipynb`

The Gradio playground lets you drive the tumor board yourself — pick a case, consult specialists, submit a diagnosis, and see the grader decompose your reward.

## Why this matters

The cheap, generic reward for most RL-on-LLM experiments today is "does the regex match the gold answer?" That is a recipe for models that *look* smart and *aren't*.

Bioresearch is a small demonstration that you can build **process-reward environments** — where the gradient teaches the agent to *work like a professional*, not just to guess like a champion. Multi-agent orchestration, tool use, and long-horizon credit assignment all fall out of the scoring function when the environment is designed for GRPO from the start.

We are excited to see what other domains this pattern extends to.
