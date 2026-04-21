# Bioresearch — 3-minute demo video script

Target length: **~3:00** (hard cap 3:30). Target medium: screen recording with voiceover.

Use either the Gradio playground (`uv run --extra playground python playground.py`) or the evaluation notebook. All on-screen text should be legible at 720p.

---

## 0:00 — 0:15 — Hook

**Show:** Gradio playground, Virtual Tumor Board tab, fresh case loaded.

**Voiceover:**

> A real molecular tumor board takes hours, five specialists, and a stack of papers.
> We built an OpenEnv where a 1.5-billion-parameter model learns to run one — end-to-end — using GRPO.

**Cut to:** full-screen title card: `Bioresearch: a multi-agent GRPO environment`.

---

## 0:15 — 0:45 — The environment in 30 seconds

**Show:** side-by-side — `README.md` "five tasks" table on the left, Gradio playground on the right.

**Voiceover:**

> Bioresearch ships with five biomedical tasks in a single OpenEnv. Four of them are single-turn.
> The fifth — the Virtual Tumor Board — gives the agent six tools, four specialist actors, and eight turns to reach a confident diagnosis.
> Every tool and every specialist is a deterministic pure function of the case. That is the GRPO contract: identical inputs, identical rewards, every time.

**Cursor action:** click through `available_tools` and `available_specialists` in the playground.

---

## 0:45 — 1:30 — A live tumor board episode

**Show:** playground, step through a real case (e.g. `dna_007` whose gold is `creutzfeldt-jakob disease`).

**Voiceover, paced to actions on screen:**

> Here is a case the model has never seen. I click "ask geneticist" — the specialist responds with the variant type and mechanism.
> "Ask pathway analyst" — the specialist names the disrupted pathway.
> "Ask clinician" — the specialist bridges molecular evidence to a phenotype.
> "Literature snippet" — the environment hands back a curated abstract for the candidate disease.
> Now I submit consensus. The grader decomposes the reward into five parts: answer correctness, specialist coverage, reasoning synthesis, efficiency, and trajectory consistency. **The final score is 0.84** — inside the GRPO sweet spot.

---

## 1:30 — 2:10 — GRPO training

**Show:** Colab notebook (`train_grpo.ipynb`) running — reward curve being drawn live, or a pre-rendered PNG.

**Voiceover:**

> We train Qwen 2.5 1.5B on a free Colab T4. Two curriculum stages: warm up on single-turn classification, then switch to the tumor board. 150 GRPO steps, four generations per group, LoRA rank 16, Unsloth 4-bit.
> Watch the reward curve: mean reward on the held-out split climbs from 0.14 to 0.63 in under an hour.
> More importantly: the share of rollouts that consult at least two specialists jumps from 7 percent to 88 percent. **The model learned the *process* of diagnosis, not just the answer.**

---

## 2:10 — 2:40 — Before and after

**Show:** split screen. Left: the "before" rollout — the model guesses a diagnosis on turn 1. Right: the "after" rollout — the model consults three specialists, pulls a literature snippet, then commits.

**Voiceover:**

> Before: one turn, wrong guess, reward 0.08.
> After: four tool calls, correct consensus, reward 0.84.
> We never told the model to consult specialists. The environment did — through the shape of its reward function.

---

## 2:40 — 3:00 — Close

**Show:** README landing page, links list, QR code to the HF Space.

**Voiceover:**

> The full environment, Colab notebooks, evaluation harness and Gradio playground are open-source.
> Run your own model, design your own specialist, rewire the reward. Bioresearch is a blueprint for GRPO-native, multi-agent environments anywhere professionals make decisions in teams.

**End card:**
- `github.com/YOUR_USERNAME/bioresearch`
- `huggingface.co/spaces/YOUR_USERNAME/bioresearch`

---

## Production notes

- Use 1.5x speed on the specialist query steps in the tumor board demo so the whole episode fits in 45 seconds.
- Pre-warm the playground (first reset is cold) before recording.
- If you show Colab training live, start the run 3 minutes before the recording and resume from step 30 — the curve is more dramatic past the warm-up.
- Caption all on-screen text; reward numbers must be legible.
- Export as 1080p, H.264, max bitrate 12 Mbps.
