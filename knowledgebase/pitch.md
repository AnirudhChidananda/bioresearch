# Bioresearch — Drug Discovery Lab · 3-Minute Hackathon Pitch

> **One-liner**: "An OpenEnv **drug-discovery lab** where agents chain tool calls — InterPro, PPI, GO, pathways — to go from a mutation to a druggable hypothesis, with dense `<think>`-trace rewards that teach frontier models to *reason* about disease, not just classify it."

Target: **winning** Cerebral Valley hackathon across Innovation (40%), Storytelling (30%), Reward Improvement (20%), Training Script (10%).

---

## 0:00 — 0:30  •  The "why"

- *Speaker on camera, single slide: a DNA mutation → ??? → a patient.*
- "Every drug we'll ever make starts the same way: someone reads a mutation brief, pulls evidence from five databases, and writes a hypothesis. Today, **that** is what frontier models fail at — not classification, not single-turn QA, but the **8-step workflow** a real scientist runs."
- "Aging and rare-disease research are bottlenecked on this exact loop. So we built an environment that trains for it."

Deliverable on screen: the opening brief for `DNA-042 — PDE11A → Cushing syndrome` with 4 candidate genes hidden.

---

## 0:30 — 1:15  •  What we shipped

Slide: side-by-side "before / after".

- **Before** (starter kit): 4 single-turn tasks (DNA classify / reason, evidence rank, protein function). A decent OpenEnv, but one-shot.
- **After**: **7 tasks**, three of which are long-horizon:
  - `target_discovery_lab` — pick a target from a DNA brief, characterise it via tool calls, propose a therapeutic intervention.
  - `protein_hypothesis_lab` — gold `<think>` chain-of-thought traces power a *dense per-step reward* that tracks how closely the agent's reasoning matches a senior scientist's.
  - `curriculum_self_play` — **Theme 4**: we progressively hide tool outputs as the model improves, forcing it to internalise biology.

Show the 7 tools (`get_interpro`, `get_ppi`, `get_go`, `get_sequence`, `get_subcellular_location`, `search_catalogue`, `get_pathway`) and the phased state machine `TARGET → CHARACTERIZE → HYPOTHESIZE → INTERVENE → SUBMIT`.

---

## 1:15 — 2:00  •  Live demo (the hero moment)

Open `playground.py`, click the **🧪 Drug Discovery Lab** tab.

1. Reset `target_discovery_lab`. The brief names a mutation on chromosome 2p16.3 with a cAMP pathway.
2. Agent calls `get_pathway(gene="PDE11A")` → notebook fills with the downstream PKA cascade.
3. Agent calls `get_interpro` → "Phosphodiesterase domain, 3',5'-cyclic AMP PDE".
4. Agent calls `get_go(protein_id=...)` → `GO:0004114` (cAMP phosphodiesterase activity, leaf).
5. Agent submits `answer="Cushing syndrome"`, `proposed_intervention={"mode":"activate","target":"PDE11A"}`.
6. Score panel unrolls: disease 0.95 · leaf-GO F1 0.80 · intervention plausible 0.70 · tool efficiency 0.90 · trace coherence 0.78 → **terminal reward 0.84**.

> **Punchline**: each tool call *moved* the reward; the per-step process reward from matching gold `<think>` steps is what makes this trainable with GRPO.

---

## 2:00 — 2:45  •  The reward curve (Theme 3, 20% of judging)

Show `notebooks/train_grpo_colab.ipynb` reward plot: **Qwen-2.5-1.5B** trained with TRL GRPO against the live OpenEnv server.

- X axis: training steps.
- Y axis: mean episode reward on `protein_hypothesis_lab`.
- Baseline zero-shot: ~0.29.
- After 150 GRPO steps (≈ 45 min on a free Colab T4): ~0.48.
- Smooth curve, not a step-function — **because** the per-step `<think>` similarity gives a dense signal that a sparse terminal reward can't.

Call out the *Self-Improvement* bonus: re-run with `curriculum_self_play` and the model keeps climbing as tool hints get progressively hidden.

---

## 2:45 — 3:00  •  The close

- "Seven tasks, seven tools, two reward channels, one real research loop. We think this is the template for teaching reasoning models the one thing they actually can't do yet: **science, iteratively**."
- "Everything is on HF Spaces: `openenv push`, `python playground.py`, open the Colab — you're training in five minutes."
- "Name: **Bioresearch — Drug Discovery Lab**. Thanks."

---

## Speaker notes / fallback cues

- If the live demo stalls, fall back to the recorded GIF `assets/lab_demo.gif`.
- Keep the Colab tab **pre-warmed** with the reward plot already rendered; don't re-run training on stage.
- If asked "why not a real ClinVar/UniProt call?": "For hackathon reproducibility + GRPO determinism. Tools are swappable — the OpenEnv schema means a live MCP adapter drops in behind the same action space."
- If asked about multi-agent (Theme 1): show `openenv.yaml` entry for the optional Principal ↔ Specialist dispatch mode — shipped as a stretch, not in the default demo path.
