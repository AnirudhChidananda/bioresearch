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
- **After**: **14 tasks in 5 narrative scenes** — variant reasoning → protein function → systems biology → clinical → long-horizon labs. The catalogue reads top-to-bottom like a PI's workflow.
  - **Scene 1 · Variant reasoning.** `dna_classification`, `dna_reasoning`, `evidence_ranking` — from raw variant brief to ranked diagnosis with elimination reasoning.
  - **Scene 2 · Protein function.** `protein_function` — sequence + domains → function / location / GO.
  - **Scene 3 · Systems biology (the v3 cluster).** Four new world-modeling primitives that push the model from "memorise labels" to "reason on declared topologies":
    - `kegg_pathway_reasoning` — reason on a declarative KEGG graph (`TARDBP* -| CxI -> Q`). Reward blends disease accuracy + edge-Jaccard + process trace + pathway-gene F1.
    - `perturbation_qa` — batched CRISPRi binary world modeling (the crispest GRPO curve in the Colab).
    - `perturbation_direction_qa` — 3-class directional CRISPRi (`Increase`/`Decrease`/`Unknown`). Denser GRPO signal than the binary variant.
    - `perturbation_benchmark` — umbrella over 4 CRISPRi variants with a weighted mean (25% each).
  - **Scene 4 · Clinical.** `clinical_diagnosis` — diff-dx ranking + gold GPT-OSS-120B CoT trace.
  - **Scene 5 · Long-horizon labs (the hackathon hero).**
    - `protein_hypothesis_lab` / `target_discovery_lab` — pick a target from a brief, characterise it via tool calls, propose an intervention, **and (v2) emit a concrete SMILES in the new DRUG_DESIGN phase**.
    - `clinical_diagnosis_lab` — radiology-style differentials with gold GPT-OSS-120B step-wise reasoning powering a dense per-step process reward.
    - `ligand_design` — given a gene plus a GO-neighborhood prompt, propose a high-pIC50 molecule. Graded by SMILES token Jaccard + property proximity + top-1000 catalogue membership.
    - `curriculum_self_play` — **Theme 4 capstone**: we progressively hide tool outputs as the model improves, forcing it to internalise biology.
- **New v3 tool:** `get_structure(protein_id)` — AlphaFold reference tool that lets the lab quote a concrete structure id in the closing hypothesis.

Show the 11 tools (`get_interpro`, `get_ppi`, `get_go`, `get_sequence`, `get_subcellular_location`, `search_catalogue`, `get_pathway`, **`get_drug_properties`, `get_candidate_ligands`, `get_perturbation_pair`, `get_structure`**) and the extended phased state machine `TARGET → CHARACTERIZE → HYPOTHESIZE → INTERVENE → DRUG_DESIGN → SUBMIT`.

---

## 1:15 — 2:00  •  Live demo (the hero moment)

Open `playground.py`, click the **🧪 Drug Discovery Lab** tab.

1. Reset `target_discovery_lab`. The brief names a mutation on chromosome 2p16.3 with a cAMP pathway.
2. Agent calls `get_pathway(gene="PDE11A")` → notebook fills with the downstream PKA cascade.
3. Agent calls `get_interpro` → "Phosphodiesterase domain, 3',5'-cyclic AMP PDE".
4. Agent calls `get_go(protein_id=...)` → `GO:0004114` (cAMP phosphodiesterase activity, leaf).
5. Agent submits `answer="Cushing syndrome"`, `proposed_intervention={"mode":"activate","target":"PDE11A"}`.
6. **NEW — DRUG_DESIGN beat (15s).** Before `SUBMIT` fires, the schedule hands the agent a `DRUG_DESIGN` window. It calls `get_candidate_ligands(gene="PDE11A", k=5)` → a ranked list of high-pIC50 molecules from the 1000-row SMILES catalogue. It picks one, calls `get_drug_properties(smiles=...)` → `pIC50=10.6, logP=1.47, drug_score=10.6, in_catalogue=True`, and submits `predicted_ligand=<SMILES>`. The pitch slide now shows **mutation → mechanism → protein → SMILES** on a single row.
7. Score panel unrolls: disease 0.95 · leaf-GO F1 0.80 · intervention plausible 0.70 · tool efficiency 0.90 · trace coherence 0.78 · drug_design addon 0.71 → **terminal reward 0.86**.

> **Punchline**: each tool call *moved* the reward; the per-step process reward from matching gold `<think>` steps plus the DRUG_DESIGN addon is what makes this trainable with GRPO *and* visually memorable.

---

## 2:00 — 2:45  •  The reward curve (Theme 3, 20% of judging)

Show `notebooks/train_grpo_colab.ipynb` reward plot: **Qwen-2.5-1.5B** trained with TRL GRPO against the live OpenEnv server.

- X axis: training steps.
- Y axis: mean episode reward on three curves — `protein_hypothesis_lab` (long-horizon), `perturbation_qa` (binary), and the **new** `perturbation_direction_qa` (3-class directional).
- The 3-class curve climbs fastest because the extra label entropy sharpens the GRPO advantage per step — exactly the v3 reward-improvement bet.
- Baseline zero-shot: ~0.29 on `protein_hypothesis_lab`; after 150 GRPO steps (≈ 45 min on a free Colab T4): ~0.48. Smooth curves, not step-functions — **because** the per-step `<think>` similarity + directional F1 give dense signal that a sparse terminal reward can't.
- Bonus slide: the KEGG `pathway_graph` Jaccard term shows up as a separate bar that moves independently of disease accuracy — evidence the agent is actually reading the graph.

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
