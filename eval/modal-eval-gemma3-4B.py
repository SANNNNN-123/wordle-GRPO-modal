"""
Side-by-side eval on Modal: base Gemma-3 4B vs trained LoRA adapter.

Run:
    cd modal-grpo
    uv run modal run eval/modal-eval-gemma3-4B.py::evaluate_sxs \\
      --adapter-path /models/runs/20260704-060851_gemma-3-4b-it/adapters/adapter_final \\
      --num-samples 50

Quick eval (25 games, with history only):
    uv run modal run eval/modal-eval-gemma3-4B.py::evaluate_sxs \\
      --adapter-path /models/runs/20260704-060851_gemma-3-4b-it/adapters/adapter_final \\
      --num-samples 25
"""

from __future__ import annotations

from pathlib import Path

import modal

REPO_ROOT = Path(__file__).resolve().parent.parent

MODELS_DIR = Path("/models")
HF_HUB_CACHE_DIR = "/root/.cache/huggingface"

checkpoints_volume = modal.Volume.from_name("wordle-grpo-checkpoints", create_if_missing=True)
hf_cache_volume = modal.Volume.from_name("wordle-grpo-hf-cache", create_if_missing=True)

eval_volumes = {
    str(MODELS_DIR): checkpoints_volume,
    HF_HUB_CACHE_DIR: hf_cache_volume,
}

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
    )
    .env({"HF_HOME": HF_HUB_CACHE_DIR, "PYTHONPATH": "/repo"})
    .add_local_dir(
        str(REPO_ROOT),
        remote_path="/repo",
        ignore=["venv", ".venv", ".git", "experiments", "__pycache__", "*.npz", ".env"],
    )
)

app = modal.App("wordle-grpo-gemma3-eval")

DEFAULT_ADAPTER = "/models/runs/20260704-060851_gemma-3-4b-it/adapters/adapter_final"


@app.function(
    image=image,
    gpu="A10G",
    timeout=60 * 60 * 3,
    volumes=eval_volumes,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def evaluate_sxs(
    adapter_path: str = DEFAULT_ADAPTER,
    num_samples: int = 50,
    with_history: bool = True,
    also_without_history: bool = False,
    config_rel_path: str = "config/grpo_lora_config.json",
) -> dict:
    import os
    import sys

    sys.path.insert(0, "/repo")
    os.chdir("/repo")

    from eval.eval_base_and_lora import run_side_by_side_eval

    adapter = Path(adapter_path)
    if not adapter.exists():
        raise FileNotFoundError(f"Adapter not found on volume: {adapter_path}")

    report = run_side_by_side_eval(
        repo_root=Path("/repo"),
        config_path=Path("/repo") / config_rel_path,
        adapter_path=adapter,
        num_samples=num_samples,
        with_history=with_history,
        also_without_history=also_without_history,
    )

    print(f"\nEval artifacts: {report['output_dir']}")
    checkpoints_volume.commit()
    return report
