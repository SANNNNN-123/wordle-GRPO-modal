"""
Modal entrypoint for Wordle GRPO training (Gemma-3 4B).

Uses the **Gemma repo's custom DPO-style GRPO** (winner/loser pairs + KL penalty)
from grpo_trainer.py — NOT TRL's GRPOTrainer used in modal-run-grpo-Qwen3-1.7B.py.

Modal infrastructure follows the Qwen3 reference:
  - separate checkpoints + HF-cache volumes
  - uv_pip_install, wandb, huggingface-secret / wandb-secret
  - volume.commit() after training

Run (recommended — Qwen-style, invoke function directly):
    cd modal-grpo
    uv run modal run --detach modal-run-grpo-Gemma3-4b.py::train_grpo \\
      --iterations 100 --wandb-run-name my-run

Run (alternate — via local entrypoint):
    uv run modal run --detach modal-run-grpo-Gemma3-4b.py --iterations 100

Prerequisites:
    pip install modal && modal setup
    modal secret create huggingface-secret HF_TOKEN=<token>
    modal secret create wandb-secret WANDB_API_KEY=<key>   # optional
"""

from __future__ import annotations

from pathlib import Path

import modal

MODAL_GRPO_DIR = Path(__file__).resolve().parent
REPO_ROOT = MODAL_GRPO_DIR.parent

# --- Volume layout (matches Qwen3 reference) ---
MODELS_DIR = Path("/models")
HF_HUB_CACHE_DIR = "/root/.cache/huggingface"

checkpoints_volume = modal.Volume.from_name("wordle-grpo-checkpoints", create_if_missing=True)
hf_cache_volume = modal.Volume.from_name("wordle-grpo-hf-cache", create_if_missing=True)

train_volumes: dict[str, modal.Volume] = {
    str(MODELS_DIR): checkpoints_volume,
    HF_HUB_CACHE_DIR: hf_cache_volume,
}

# --- Image ---
image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(
        "torch>=2.4.0",
        "transformers>=4.51.0",
        "peft>=0.14.0",
        "datasets>=3.0.0",
        "accelerate>=1.2.0",
        "safetensors>=0.4.0",
        "tqdm>=4.66.0",
        "requests>=2.31.0",
        "wandb>=0.18.0",
        "huggingface_hub>=0.26.0",
    )
    .env(
        {
            "HF_HOME": HF_HUB_CACHE_DIR,
            "TRANSFORMERS_CACHE": HF_HUB_CACHE_DIR,
            "PYTHONPATH": "/app",
            "WANDB_PROJECT": "wordle-grpo-gemma",
        }
    )
    .add_local_dir(
        str(REPO_ROOT),
        remote_path="/repo",
        ignore=[
            "venv",
            ".venv",
            ".git",
            "experiments",
            "__pycache__",
            "*.npz",
            "modal-grpo/.env",
        ],
    )
    .add_local_dir(str(MODAL_GRPO_DIR), remote_path="/app")
)

app = modal.App("wordle-grpo-gemma3-4b")

DEFAULT_GPU = "A10G"  # 24 GB — Gemma-3 4B + LoRA; use L4 for smoke tests
DEFAULT_MODEL = "google/gemma-3-4b-it"
DEFAULT_CONFIG = "config/grpo_lora_config.json"


def _training_secrets(use_wandb: bool) -> list[modal.Secret]:
    secrets = [modal.Secret.from_name("huggingface-secret")]
    if use_wandb:
        secrets.append(modal.Secret.from_name("wandb-secret"))
    return secrets


@app.function(
    image=image,
    gpu=DEFAULT_GPU,
    timeout=60 * 60 * 24,  # 24h — 500 steps ~10–12h on A10G (no in-training eval)
    volumes=train_volumes,
    secrets=_training_secrets(use_wandb=True),
)
def train_grpo(
    iterations: int = 500,
    model: str = DEFAULT_MODEL,
    config_rel_path: str = DEFAULT_CONFIG,
    gpu: str = DEFAULT_GPU,
    wandb_run_name: str | None = None,
    push_to_hub: bool = False,
    hub_model_id: str | None = None,
) -> str:
    """Run Gemma-style custom GRPO training on a Modal GPU."""
    import os
    import sys

    sys.path.insert(0, "/app")
    os.chdir("/repo")

    if wandb_run_name:
        os.environ["WANDB_NAME"] = wandb_run_name

    from grpo_trainer import run_training

    run_id = __import__("datetime").datetime.now().strftime("%Y%m%d-%H%M%S")
    model_short = model.split("/")[-1]
    output_dir = MODELS_DIR / "runs" / f"{run_id}_{model_short}"

    config_path = Path("/repo") / config_rel_path
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    print(f"GPU requested: {gpu}")
    print(f"Model: {model}")
    print(f"Config: {config_path}")
    print(f"Output: {output_dir}")
    print(f"GRPO: custom DPO-style (Gemma repo), not TRL GRPOTrainer")

    final_adapter = run_training(
        repo_root=Path("/repo"),
        output_dir=output_dir,
        config_path=config_path,
        model_override=model,
        iterations_override=iterations,
        use_wandb=bool(os.environ.get("WANDB_API_KEY")),
        push_to_hub=push_to_hub,
        hub_model_id=hub_model_id,
    )

    checkpoints_volume.commit()
    hf_cache_volume.commit()
    return str(final_adapter)


@app.local_entrypoint()
def main(
    iterations: int = 500,
    model: str = DEFAULT_MODEL,
    config: str = DEFAULT_CONFIG,
    gpu: str = DEFAULT_GPU,
    wandb_run_name: str | None = None,
    push_to_hub: bool = False,
    hub_model_id: str | None = None,
    no_wandb: bool = False,
):
    """
    Convenience wrapper — prefer invoking train_grpo directly (Qwen-style):

        uv run modal run --detach modal-run-grpo-Gemma3-4b.py::train_grpo \\
          --iterations 100 --wandb-run-name my-run
    """
    print(f"Launching GRPO (Gemma-3 4B) on Modal gpu={gpu}...")
    print(f"  model={model}")
    print(f"  iterations={iterations}")
    print(f"  config={config}")
    print("  Tip: use ::train_grpo directly for long runs (see HOW TO RUN.md)")

    train_fn = train_grpo.with_options(gpu=gpu)
    if no_wandb:
        train_fn = train_grpo.with_options(gpu=gpu, secrets=_training_secrets(use_wandb=False))

    adapter_path = train_fn.remote(
        iterations=iterations,
        model=model,
        config_rel_path=config,
        gpu=gpu,
        wandb_run_name=wandb_run_name,
        push_to_hub=push_to_hub,
        hub_model_id=hub_model_id,
    )

    print(f"\nTraining finished. Adapter saved to Modal volume at:\n  {adapter_path}")
    print("\nDownload checkpoints:")
    print("  modal volume get wordle-grpo-checkpoints runs/ ./local-runs/")
