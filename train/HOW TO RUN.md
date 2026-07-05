# How to Run — Wordle GRPO on Modal (Gemma-3 4B)

Train Gemma-3 to play Wordle using **custom DPO-style GRPO** on Modal cloud GPUs.

**Entry script:** `train/modal-run-grpo-Gemma3-4b.py`

Training runs in the cloud. Your laptop only launches and monitors the job.

---

## Prerequisites

### 1. Modal account

```sh
cd modal-grpo
uv sync
uv run modal setup
```

### 2. Hugging Face (Gemma is gated)

1. Accept the license for [google/gemma-3-4b-it](https://huggingface.co/google/gemma-3-4b-it) (and any other model you use).
2. Create a read token at https://huggingface.co/settings/tokens

### 3. Modal secrets

```sh
modal secret create huggingface-secret HF_TOKEN=<your_hf_token>

# optional — for W&B charts
modal secret create wandb-secret WANDB_API_KEY=<your_wandb_key>
```

List secrets:

```sh
modal secret list
```

If a secret already exists, update it with:

```sh
modal secret delete huggingface-secret
modal secret create huggingface-secret HF_TOKEN=<new_token>
```

> Local `.env` is **not** used by Modal automatically. Secrets must be created via `modal secret create`.

---

## Quick reference

| Run type | Command |
|----------|---------|
| **Recommended launch** | `modal run --detach ...::train_grpo` (Qwen-style) |
| Smoke test (5 steps, cheap) | see below |
| Medium run (100 steps) | `--iterations 100` |
| Full run (500 steps) | default, no flag |
| Detached (safe to close terminal) | `--detach` after `modal run` |
| Skip W&B | omit `--wandb-run-name` and use secrets off (see below) |

### Why `::train_grpo`? (Qwen3 reference pattern)

The Qwen script (`train/modal-run-grpo-Qwen3-1.7B.py`) has **no** `@app.local_entrypoint()`. It runs the cloud function directly:

```sh
modal run train/modal-run-grpo-Qwen3-1.7B.py::train
```

Our Gemma script supports the same pattern — **prefer this for long runs**:

```sh
uv run modal run --detach train/modal-run-grpo-Gemma3-4b.py::train_grpo \
  --iterations 100 \
  --wandb-run-name gemma3-4b-100steps
```

This avoids the local entrypoint wrapper that calls `.remote()` and waits for a return value (more fragile with detach/logs).

---

## Step 1 — Smoke test (recommended first)

Verifies image build, HF auth, game loop, and GRPO update (~5 min after first image build).

```sh
cd modal-grpo

uv run modal run --detach train/modal-run-grpo-Gemma3-4b.py::train_grpo \
  --iterations 5 \
  --model google/gemma-3-270m-it \
  --gpu L4
```

> For smoke tests without W&B, temporarily use a run that doesn't need `wandb-secret`, or create the secret. The `::train_grpo` function always enables W&B if `WANDB_API_KEY` is present.

Success looks like:

```
Training finished. Adapter saved to Modal volume at:
  /models/runs/<timestamp>_gemma-3-270m-it/adapters/adapter_final
```

---

## Step 2 — Medium run (validate 4B)

~2–3 hours on A10G. Good before a full 500-step run.

```sh
uv run modal run --detach train/modal-run-grpo-Gemma3-4b.py::train_grpo \
  --iterations 100 \
  --wandb-run-name gemma3-4b-100steps
```

---

## Step 3 — Full training (500 steps)

Matches the original MLX repo config (ported here). Expect **~10–12 hours** on A10G (in-training eval removed).

```sh
uv run modal run --detach train/modal-run-grpo-Gemma3-4b.py::train_grpo \
  --wandb-run-name gemma3-4b-full-500-v1
```

Default settings:

| Setting | Value |
|---------|--------|
| Model | `google/gemma-3-4b-it` |
| Steps | 500 |
| GPU | A10G |
| Config | `config/grpo_lora_config.json` |
| LoRA rank | 64 |
| Checkpoints | every 50 steps |
| Eval | external only — see `eval/` after training |
| Timeout | 24 hours |

> **Cost estimate:** ~$11–13 GPU (~$14–16 total with CPU/RAM). Check Modal billing dashboard.

---

## Always use `--detach` + `::train_grpo`

```sh
uv run modal run --detach train/modal-run-grpo-Gemma3-4b.py::train_grpo ...
```

| Mode | Close terminal / Ctrl+C |
|------|-------------------------|
| `modal run` (no detach) | **Kills** the job |
| `modal run --detach` + `::train_grpo` | Job **keeps running** |
| `modal app stop <app-id>` | **Kills** the job |

After detach, monitor via **dashboard + W&B only**. Avoid `modal app logs` until Completed.

---

## Monitor progress

### Modal dashboard

https://modal.com/apps/<your-username>/main

Look for app `wordle-grpo-gemma3-4b` and status **Running** → **Completed**.

### Weights & Biases

Project: `wordle-grpo-gemma`

Metrics logged:

- `train/loss`
- `train/avg_reward`
- `train/rolling_win_rate`

Wordle win rate vs base model: run external eval in `eval/` after training (or on checkpoints).

---

## Download results

Checkpoints live on Modal volume `wordle-grpo-checkpoints`:

```sh
cd modal-grpo

# Download one run (recommended — avoids path errors)
uv run modal volume get wordle-grpo-checkpoints \
  runs/20260704-060851_gemma-3-4b-it \
  ./local-runs/20260704-060851_gemma-3-4b-it

# Or only the final adapter (~190MB)
uv run modal volume get wordle-grpo-checkpoints \
  runs/20260704-060851_gemma-3-4b-it/adapters/adapter_final \
  ./local-runs/adapter_final
```

> If you get `Is a directory`, download a **specific run path** (not `runs/` → `./local-runs/`).

Output layout:

```
local-runs/<timestamp>_gemma-3-4b-it/
├── adapters/
│   ├── adapter_step_50/
│   ├── adapter_step_100/
│   └── adapter_final/          ← use this
│       ├── adapter_config.json
│       └── adapter_model.safetensors
├── training_metrics.jsonl
└── grpo_lora_config.json
```

Optional — download cached model weights:

```sh
uv run modal volume get wordle-grpo-hf-cache / ./local-hf-cache/
```

---

## CLI flags

```sh
uv run modal run --detach train/modal-run-grpo-Gemma3-4b.py::train_grpo [flags]
```

Note: `--gpu` and `--no-wandb` only work via the local entrypoint (`modal run file.py`). For `::train_grpo`, GPU is set in the script (default A10G) or edit `train_grpo.with_options(gpu=...)` in code.

| Flag | Default | Description |
|------|---------|-------------|
| `--iterations` | 500 | Training steps |
| `--model` | `google/gemma-3-4b-it` | Hugging Face model ID |
| `--gpu` | `A10G` | Modal GPU (`L4` for cheap tests) |
| `--config` | `config/grpo_lora_config.json` | Config path (relative to repo root) |
| `--wandb-run-name` | auto | W&B run name |
| `--no-wandb` | off | Skip W&B (no `wandb-secret` needed) |
| `--push-to-hub` | off | Push final LoRA to HF Hub |
| `--hub-model-id` | — | Hub repo ID when pushing |

---

## Files in this folder

| File | Purpose |
|------|---------|
| `train/modal-run-grpo-Gemma3-4b.py` | Modal app — launch training |
| `grpo_trainer.py` | PyTorch GRPO training loop |
| `wordle_core.py` | Wordle game + rewards (no MLX) |
| `train/modal-run-grpo-Qwen3-1.7B.py` | Reference only (TRL GRPO, not used here) |
| `pyproject.toml` / `uv.lock` | Local env (`uv sync`) |
| `config/` | Training hyperparameters (`grpo_lora_config.json`) |
| `data/` | Word lists, entropy map, RL training trajectories |

This repo is **self-contained** — `config/` and `data/` live in the repo root (no parent folder required).

---

## Troubleshooting

### W&B says "Finished" but Modal says "Cancelled"

W&B closes the run when the **process exits** — including cancellation. **Trust the Modal dashboard** for real status.

### `Received a cancellation signal`

Something stopped the job externally: `modal app stop`, dashboard Cancel, a new `modal run`, or Ctrl+C without detach. Not a training bug.

### Detached run still cancelled

Use **`::train_grpo`** (Qwen pattern), not bare `modal run file.py`. Do not run `modal app logs` until the job completes.

---

## Cost / time estimates

| Run | GPU | Steps | Approx. time |
|-----|-----|-------|----------------|
| Smoke test | L4 | 5 | ~5–10 min |
| Medium | A10G | 100 | ~2–3 hours |
| Full | A10G | 500 | ~10–12 hours |

~70–75 sec/step observed for Gemma-3 4B on A10G (varies with game length and eval).

---

## Evaluate adapter vs base (before full training)

Side-by-side on the **test set** (same games, base vs LoRA):

```sh
cd modal-grpo

# Quick eval — 25 games with history (~20–30 min on A10G)
uv run modal run eval/modal-eval-gemma3-4B.py::evaluate_sxs \
  --adapter-path /models/runs/20260704-060851_gemma-3-4b-it/adapters/adapter_final \
  --num-samples 25

# Fuller eval — 50 games
uv run modal run eval/modal-eval-gemma3-4B.py::evaluate_sxs \
  --adapter-path /models/runs/20260704-060851_gemma-3-4b-it/adapters/adapter_final \
  --num-samples 50

# Both with and without game history
uv run modal run eval/modal-eval-gemma3-4B.py::evaluate_sxs \
  --adapter-path /models/runs/20260704-060851_gemma-3-4b-it/adapters/adapter_final \
  --num-samples 50 \
  --also-without-history
```

Reads adapter from the **Modal volume** — no local download required.

Reports saved to volume:

```
runs/<run>/eval/results/base_and_lora_<N>games_<timestamp>/
├── summary.json      # aggregate stats + per_game list
└── per_game.jsonl    # one JSON object per game
```

See `eval/README.md` for field descriptions.

| Metric | Good sign | Bad sign (skip or fix before 500 steps) |
|--------|-----------|----------------------------------------|
| LoRA win% > Base | Training helped | LoRA ≤ Base |
| invalid_guess_rate dropping | Format learning | Still ~100% invalid |
| Delta +5% or more | Worth continuing | 0% or negative delta |

---
