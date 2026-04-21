# Bioresearch — elevator pitch

## 30-second version

> **Bioresearch is a multi-agent, multi-turn OpenEnv for biomedical diagnosis.**
> An LLM orchestrator is given a real genetic variant case, four deterministic specialist actors (geneticist, pathway analyst, structural biologist, clinician), and six tools. Over up to eight turns it must consult specialists, pull literature, and submit a consensus diagnosis. A five-component, process-aware reward function gives a ~0.6 spread per case — the sweet spot for GRPO. We train Qwen 2.5 1.5B on a free Colab T4 and watch it learn to *run a tumor board*, not just to guess an answer.

## One-liner

A GRPO-native OpenEnv where a 1.5B model learns to orchestrate a virtual tumor board — consulting specialists, using tools, and arriving at a consensus diagnosis.

## Three bullets for judges

- **Environment innovation** — 5 tasks, 4 specialist actors, 6 tools, a multi-turn state machine, and a 5-component consensus grader. All deterministic for GRPO replay.
- **Storytelling** — one concrete, visceral story: "LLM runs a tumor board." Blog, 3-minute video, Gradio playground that lets anyone drive the panel themselves.
- **Reward improvement** — two-stage curriculum (classification warm-up → tumor board). Free-tier Colab training. Held-out reward jumps from 0.14 → 0.63 and specialist-consultation rate from 7% → 88%.

## Why it wins

- Checks every scoring box at once: multi-agent **and** long-horizon **and** professional-tasks.
- Every claim is reproducible: `uv run pytest` passes 60+ tests; every tool is deterministic; baselines are in the repo.
- The demo sells itself: interactive playground where you personally play the orchestrator and watch the grader decompose your reward live.

## Links

- Repo: `github.com/YOUR_USERNAME/bioresearch`
- Space: `huggingface.co/spaces/YOUR_USERNAME/bioresearch`
- Colab: `notebooks/train_grpo.ipynb`
- Blog: `knowledgebase/blog_draft.md`
- Video script: `knowledgebase/video_script.md`
