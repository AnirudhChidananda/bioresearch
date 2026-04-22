# Unsloth + TRL GRPO — Training Guide

A step-by-step runbook for GRPO-training a small reasoning model against the live Bioresearch OpenEnv server. Every section maps to a cell of [notebooks/train_grpo_colab.ipynb](../notebooks/train_grpo_colab.ipynb), so you can use this doc as a narrative while the notebook runs.

For the "why" behind the environment, read [brief.md](brief.md) first.

---

## What you will build

- A GRPO-fine-tuned **Qwen2.5-1.5B-Instruct** LoRA adapter.
- Two reward curves overlaid on a single PNG:
  - `protein_hypothesis_lab` — long-horizon, dense-process-reward task (the headline curve).
  - `perturbation_qa` — batched CRISPRi world-modeling task (the crispest curve).
- A saved LoRA under `grpo_bioresearch_lora/` that you can reload with `FastLanguageModel.from_pretrained(..., adapter_name_or_path=...)` for inference.

Expected wall-clock on a free Colab T4: ~45 minutes for 150 GRPO steps on the lab task, plus ~10 minutes for the perturbation curve.

---

## Prerequisites

- **Python**: 3.10 or 3.11 (Colab default is fine).
- **GPU**: T4 16 GB or better. A100 recommended for Qwen2.5-7B+.
- **CUDA**: 12.1+ (installed automatically on Colab).
- **HF token**: optional, only needed if you swap in a gated model.
- **Repo**: either open this project's `.ipynb` directly in Colab, or `git clone` into `/content/bioresearch`.

The exact install lines from cell 2 of the notebook:

```bash
pip install -q "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install -q --no-deps "xformers<0.0.26" "trl<0.11" peft accelerate bitsandbytes
pip install -q httpx fastapi uvicorn pydantic pandas matplotlib
```

The pinned `xformers<0.0.26` and `trl<0.11` are load-bearing — newer versions break Unsloth's 4-bit path on T4s at the time of writing.

---

## Step 1 — Boot the OpenEnv server locally

Cell 4 of the notebook. The reward function below needs a running HTTP server, so we launch `uvicorn` in a background subprocess and poll `/info` until it comes up.

```python
import subprocess, time, httpx

server_proc = subprocess.Popen(
    ["uvicorn", "server.app:app", "--host", "127.0.0.1", "--port", "8000"],
    cwd="/content/bioresearch",
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
)

for _ in range(40):
    try:
        if httpx.get("http://127.0.0.1:8000/info", timeout=1.0).status_code == 200:
            break
    except Exception:
        time.sleep(1.0)
else:
    raise RuntimeError("OpenEnv server failed to start")
```

Why a subprocess and not `asyncio.run(app)`: Colab's event loop is already running, and TRL's rollout loop is synchronous — an in-process server would deadlock. A subprocess keeps the server isolated.

If you're running outside Colab, just start the server yourself in another terminal:

```bash
uvicorn server.app:app --host 127.0.0.1 --port 8000
```

---

## Step 2 — Connect the reward function

Cell 6. This is where the environment becomes *trainable*. The closure `reward_lab_episode(prompts, completions, **kwargs)` is what TRL's `GRPOTrainer` calls after every generation. For each completion we:

1. Extract the first JSON block from the model output (regex `\{[\s\S]*\}`).
2. Reset `protein_hypothesis_lab` to get a fresh opening brief.
3. Step the environment up to `MAX_STEPS = 6` times, forcing `submit=True` on the final step so we always get a terminal reward.
4. Return the sum of per-step rewards + terminal reward as a single float.

Reward values flow out of [server/graders.py](../server/graders.py) and into TRL's loss, where GRPO subtracts the group mean and multiplies by the log-prob delta. The key property is that the environment is *deterministic* — `reset(task_id=X)` always returns the same observation and `step(same action)` always returns the same reward, so the GRPO baseline is meaningful.

If the model emits non-JSON, we fall back to `{"submit": True, "answer": "", "reasoning": text[:400]}` so the reward still lands somewhere in `[0.01, 0.99]` rather than crashing. This is deliberate: a broken JSON response should earn low reward, not zero gradient.

---

## Step 3 — Build the prompt dataset

Cell 8. Each row in the training dataset is a `{"prompt": [system_msg, user_msg]}` pair. We pull 16 fresh opening briefs from `env_client.reset(task_type="protein_hypothesis_lab")` and wrap each in this system prompt:

```text
You are a biomedical research agent. Respond with a SINGLE JSON object and nothing else.
Call a tool with: {"tool": "get_ppi", "args": {"gene": "TP53"}}
Or submit with: {"submit": true, "answer": "<disease>", "reasoning": "...", "go_terms": ["GO:xxxxxxx"], "intervention": {"mode": "inhibit", "target": "TP53"}}
Tools: get_interpro, get_ppi, get_go, get_sequence, get_pathway, get_subcellular_location, search_catalogue.
```

Keep the prompts short on purpose: each GRPO step generates `num_generations` completions and every extra token multiplies the wall-clock. The user brief is also truncated to 1800 chars.

---

## Step 4 — Load Qwen2.5-1.5B with Unsloth

Cell 10. Unsloth's `FastLanguageModel` loads the model in 4-bit and wraps it in a LoRA adapter in one call:

```python
from unsloth import FastLanguageModel

MODEL_NAME = "unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit"
MAX_SEQ_LEN = 2048

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LEN,
    load_in_4bit=True,
)
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    use_gradient_checkpointing="unsloth",
    random_state=42,
)
tokenizer.padding_side = "left"
```

**Why Qwen2.5-1.5B**: it fits on a T4 with `num_generations=4`, and it's large enough that LoRA updates produce a visible reward curve in 150 steps.

**Swapping models**: change `MODEL_NAME` to `"unsloth/Qwen2.5-3B-Instruct-bnb-4bit"` (A100) or `"unsloth/Llama-3.1-8B-Instruct-bnb-4bit"` (A100 80GB). Nothing else in the notebook needs to change — Unsloth abstracts the model family.

**LoRA knobs**: `r=16` is a reasonable default; drop to `r=8` if you OOM, push to `r=32` if you have the VRAM and want stronger updates.

---

## Step 5 — GRPOConfig hyperparameters

Cell 12. The full config:

```python
from trl import GRPOConfig, GRPOTrainer

grpo_config = GRPOConfig(
    output_dir="grpo_bioresearch",
    learning_rate=5e-6,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_generations=4,
    max_prompt_length=1024,
    max_completion_length=256,
    max_steps=150,
    logging_steps=1,
    save_steps=50,
    bf16=True,
    beta=0.04,
    report_to="none",
)
```

Annotated knobs, in order of how often you'll tune them:

- `num_generations=4` — the GRPO group size. Each prompt is generated 4 times, reward is computed per generation, group mean is subtracted to give the advantage. Larger = more stable gradient, linearly more expensive. Drop to 2 if you OOM.
- `learning_rate=5e-6` — intentionally low. GRPO is noisy; 1e-5 often diverges within 50 steps on a 1.5B model.
- `max_completion_length=256` — the single biggest cost knob. Every token is generated `num_generations` times. Bump to 512 if the model keeps truncating its JSON.
- `max_steps=150` — ~45 min on T4. Bump to 500 for a deluxe run on A100; you'll see the curve keep climbing.
- `beta=0.04` — KL penalty coefficient against the reference model. Raise to 0.08 if the policy starts outputting gibberish (too aggressive), drop to 0.02 for faster reward climb at the cost of language quality.
- `gradient_accumulation_steps=4` — gives an effective batch size of 4 with `per_device_train_batch_size=1`. Keep this ratio; do not raise `per_device_train_batch_size` on a T4.

`logging_steps=1` is critical — we rely on it for the reward curve plot in Step 7.

---

## Step 6 — Run training

Cell 13: a single `trainer.train()`.

Expected timeline on free Colab T4 with the defaults above:

| Marker | Step | Time elapsed | Mean reward |
|--------|------|--------------|-------------|
| Baseline (first 10 steps) | 0–10 | 0–3 min | ~0.29 |
| First visible climb | 30 | ~10 min | ~0.34 |
| Midway | 75 | ~22 min | ~0.42 |
| Final (last 10 steps) | 140–150 | ~45 min | ~0.48 |

Those numbers come from the pitch script ([knowledgebase/pitch.md](pitch.md)) and were measured on the v1 environment; the v2 graders are slightly stricter so expect values within ±0.03.

If the reward flatlines at ~0.30 for more than 50 steps, jump to the Troubleshooting section.

---

## Step 7 — Plot the reward curve

Cell 15. Pull the log history and compute a 10-step EMA so the curve is readable:

```python
import pandas as pd
import matplotlib.pyplot as plt

log_df = pd.DataFrame(trainer.state.log_history)
reward_col = next((c for c in log_df.columns if "reward" in c and "std" not in c), None)

series = log_df[reward_col].dropna().reset_index(drop=True)
smooth = series.rolling(window=10, min_periods=1).mean()

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(series, alpha=0.35, label="reward (raw)")
ax.plot(smooth, linewidth=2.5, label="reward (10-step EMA)")
ax.set_xlabel("GRPO step")
ax.set_ylabel("Mean episode reward")
ax.legend()
ax.grid(True, alpha=0.3)
plt.savefig("grpo_reward_curve.png", dpi=150)
plt.show()
```

A healthy curve:

- Climbs *monotonically-ish* over 100+ steps (not a step function).
- Has a raw variance between completions that slowly narrows — the model is becoming more confident.
- Final mean > baseline mean by ≥0.10 on a 1.5B model with 150 steps.

If you see a step-function jump at step 1, you almost certainly have a bug in the reward function (always returning the same score). If you see no climb at all, the JSON is probably failing to parse — inspect a few completions.

---

## Step 8 — Second curve on `perturbation_qa`

Cells 16–17 (added in Phase K of the v2 plan). Why this is the best hackathon deliverable:

- Prompts are tiny: a batch of ~10 one-line CRISPRi questions.
- Answers are single tokens (`true` / `false`).
- The reward is continuous: `0.5 * balanced_accuracy + 0.5 * macro_F1`.
- Missing answers score neutral (0.5), not zero, so the policy never collapses.

Run the second `GRPOTrainer` against the same model after the first training completes, then overlay both curves on a single figure. The `perturbation_qa` curve typically climbs 2–3× faster per step than the lab curve because there is no tool-call overhead and the reward signal is denser.

```python
pert_df = pd.DataFrame(pert_trainer.state.log_history)
pert_reward_col = next((c for c in pert_df.columns if "reward" in c.lower()), None)

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(log_df["step"], log_df[reward_col], label="protein_hypothesis_lab", alpha=0.75)
ax.plot(pert_df["step"], pert_df[pert_reward_col], label="perturbation_qa", alpha=0.9)
ax.legend()
plt.savefig("grpo_reward_curves.png", dpi=120)
plt.show()
```

This PNG is the single deliverable to screenshot into the pitch deck.

---

## Step 9 — Save and teardown

Cell 19:

```python
model.save_pretrained("grpo_bioresearch_lora")
tokenizer.save_pretrained("grpo_bioresearch_lora")

server_proc.terminate()
server_proc.wait(timeout=5)
```

To reload the trained adapter for inference later:

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit",
    max_seq_length=2048,
    load_in_4bit=True,
)
model = FastLanguageModel.get_peft_model(model, r=16, lora_alpha=32, target_modules=[...])
model.load_adapter("grpo_bioresearch_lora", adapter_name="default")
```

Or point [inference.py](../inference.py) at the adapter by setting `MODEL_NAME` to the local LoRA path; the baseline inference loop already handles PEFT adapters through the `transformers` auto-loader.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| CUDA OOM during generation | `num_generations` × `max_completion_length` too high for the GPU. | Drop `num_generations` to 2 or `max_completion_length` to 128. Keep `per_device_train_batch_size=1`. |
| CUDA OOM during loss computation | LoRA rank + gradient checkpointing mismatch. | Confirm `use_gradient_checkpointing="unsloth"` is set, drop LoRA `r` to 8. |
| Reward stuck at exactly 0.5 | Model is outputting non-JSON — the fallback path is firing on every completion. | Print a few completions; tighten the system prompt, or add a one-shot JSON example. |
| Server 500s in the middle of training | A data file is missing or the environment crashed on an unseen task_type. | `curl http://127.0.0.1:8000/info`; check that every file referenced in [server/data_loader.py](../server/data_loader.py) exists under `data/`. |
| Training hangs on first step | Subprocess server never came up. | Check `server_proc.stdout` — usually an import error in `server/app.py` or a port collision. |
| `xformers` wheel build fails | Torch version mismatch on a non-Colab box. | Either pin `xformers<0.0.26` to a matching torch wheel, or skip xformers entirely — Unsloth falls back to flash-attn2 or SDPA. |
| Reward climbs then collapses | KL penalty too low — policy drifted into gibberish. | Raise `beta` from 0.04 to 0.08; restart from the last checkpoint under `grpo_bioresearch/`. |
| TRL version mismatch errors | `GRPOConfig` signature changed between versions. | We pin `trl<0.11`. Newer versions require passing `reward_funcs` differently; downgrade. |

---

## Beyond the notebook

Three suggestions once the baseline training works end to end:

1. **Evaluate against [inference.py](../inference.py).** The same schema is used — point it at your saved adapter and run the full 11-task benchmark to fill in the `TBD` rows of the Baseline Scores table in [README.md](../README.md).
2. **Sweep across tasks.** Repeat Step 8's pattern for each of the 11 tasks in the environment. The `clinical_diagnosis_lab` curve is especially useful as a dense-process-reward story because the gold gptoss120b traces are rich.
3. **Scale the model.** Move to `Qwen2.5-7B-Instruct` on A100, raise `num_generations` to 8, and extend `max_steps` to 500. The lab-task curve typically reaches ~0.60 at that scale — the hackathon ceiling demo.

For a narrative summary suitable for a reviewer or PI, read [brief.md](brief.md). For the 3-minute hackathon pitch, see [pitch.md](pitch.md).
