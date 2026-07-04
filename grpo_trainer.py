"""PyTorch GRPO trainer — port of src/ml/rl_trainer.py from MLX to CUDA."""

from __future__ import annotations

import itertools
import json
import math
import os
import shutil
from collections import deque
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from torch.optim import AdamW
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from wordle_core import (
    TrainConfig,
    build_grpo_pairs,
    load_train_config,
    load_word_entropy,
    load_word_list,
    play_wordle_game,
    prepare_data,
)


def cosine_decay_lr(step: int, initial_lr: float, min_lr: float, decay_steps: int) -> float:
    if step >= decay_steps:
        return min_lr
    decay_ratio = step / decay_steps
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (initial_lr - min_lr)


def pad_sequences(token_lists: list[list[int]], pad_value: int, device: torch.device) -> torch.Tensor:
    if not token_lists:
        return torch.empty(0, dtype=torch.long, device=device)
    max_len = max(len(tokens) for tokens in token_lists)
    padded = [tokens + [pad_value] * (max_len - len(tokens)) for tokens in token_lists]
    return torch.tensor(padded, dtype=torch.long, device=device)


def get_sequence_log_probs(
    model,
    prompt_ids: torch.Tensor,
    response_ids: torch.Tensor,
    pad_token_id: int,
    use_ref_adapter: bool = False,
) -> torch.Tensor:
    """Sum of token log-probabilities for each response sequence in the batch."""
    batch_size = prompt_ids.shape[0]
    full_sequence = torch.cat([prompt_ids, response_ids], dim=1)
    attention_mask = (full_sequence != pad_token_id).long()

    adapter_ctx = model.disable_adapter() if use_ref_adapter else nullcontext()
    with adapter_ctx:
        outputs = model(input_ids=full_sequence, attention_mask=attention_mask)
    logits = outputs.logits[:, :-1, :]
    target_ids = full_sequence[:, 1:]

    log_probs = F.log_softmax(logits, dim=-1)
    token_log_probs = log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)

    prompt_len = prompt_ids.shape[1]
    response_mask = (response_ids != pad_token_id).float()
    # Align token positions: response token t comes from logits at prompt_len - 1 + t
    response_token_log_probs = token_log_probs[:, prompt_len - 1 : prompt_len - 1 + response_ids.shape[1]]
    if response_token_log_probs.shape[1] > response_ids.shape[1]:
        response_token_log_probs = response_token_log_probs[:, : response_ids.shape[1]]
    elif response_token_log_probs.shape[1] < response_ids.shape[1]:
        pad_cols = response_ids.shape[1] - response_token_log_probs.shape[1]
        response_token_log_probs = F.pad(response_token_log_probs, (0, pad_cols), value=0.0)

    total_log_prob = (response_token_log_probs * response_mask).sum(dim=1)
    return total_log_prob


def grpo_loss(
    model,
    winner_toks: torch.Tensor,
    loser_toks: torch.Tensor,
    prompt_toks: torch.Tensor,
    config: TrainConfig,
    pad_token_id: int,
) -> torch.Tensor:
    log_probs_policy_winner = get_sequence_log_probs(model, prompt_toks, winner_toks, pad_token_id, use_ref_adapter=False)
    log_probs_policy_loser = get_sequence_log_probs(model, prompt_toks, loser_toks, pad_token_id, use_ref_adapter=False)
    with torch.no_grad():
        log_probs_ref_winner = get_sequence_log_probs(model, prompt_toks, winner_toks, pad_token_id, use_ref_adapter=True)
        log_probs_ref_loser = get_sequence_log_probs(model, prompt_toks, loser_toks, pad_token_id, use_ref_adapter=True)

    pi_log_ratios = log_probs_policy_winner - log_probs_policy_loser
    ref_log_ratios = log_probs_ref_winner - log_probs_ref_loser
    logits = pi_log_ratios - ref_log_ratios
    grpo = -F.logsigmoid(config.grpo_beta * logits).mean()

    kl_div = (log_probs_ref_winner - log_probs_policy_winner).mean()
    kl_penalty = torch.clamp(kl_div, min=0.0)
    return grpo + config.grpo_kl_coeff * kl_penalty


def _maybe_init_wandb(use_wandb: bool, run_name: str | None, config: TrainConfig, output_dir: Path):
    if not use_wandb:
        return None
    try:
        import wandb

        return wandb.init(
            project=os.environ.get("WANDB_PROJECT", "wordle-grpo-gemma"),
            name=run_name or output_dir.name,
            config={
                "model": config.model_name,
                "iterations": config.iterations,
                "learning_rate": config.learning_rate,
                "lora_rank": config.lora_rank,
                "grpo_beta": config.grpo_beta,
                "grpo_kl_coeff": config.grpo_kl_coeff,
                "num_generations": config.num_generations,
            },
        )
    except Exception as exc:
        print(f"Warning: wandb init failed ({exc}); continuing without logging.")
        return None


def run_training(
    repo_root: Path,
    output_dir: Path,
    config_path: Path,
    model_override: str | None = None,
    iterations_override: int | None = None,
    use_wandb: bool = False,
    push_to_hub: bool = False,
    hub_model_id: str | None = None,
) -> Path:
    config = load_train_config(config_path, model_override=model_override)
    if iterations_override is not None:
        config.iterations = iterations_override

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Model: {config.model_name}")

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "training_metrics.jsonl"
    adapter_dir = output_dir / "adapters"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, output_dir / config_path.name)

    wandb_run = _maybe_init_wandb(
        use_wandb,
        os.environ.get("WANDB_NAME"),
        config,
        output_dir,
    )

    allowed_words = load_word_list(repo_root / "data" / "nyt_possible_wordle_list.txt")
    word_entropy = load_word_entropy(repo_root)
    answers_words = sorted(load_word_list(repo_root / "data" / "nyt_answers_wordle_list.txt"))

    tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=dtype,
        device_map={"": device},
        trust_remote_code=True,
    )

    lora_config = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    model.train()

    optimizer = AdamW(model.parameters(), lr=config.learning_rate)
    train_dataset, _, _ = prepare_data(config, repo_root)
    print(f"Dataset: {len(train_dataset)} train samples")

    data_iterator = iter(itertools.cycle(train_dataset))
    win_tracker: deque[int] = deque(maxlen=config.iterations)
    train_metrics: list[dict] = []

    for step in tqdm(range(1, config.iterations + 1), desc="GRPO"):
        sample = next(data_iterator)
        rollout = play_wordle_game(
            model=model,
            tokenizer=tokenizer,
            secret_word=sample["secret"],
            config=config,
            allowed_words=allowed_words,
            word_entropy=word_entropy,
            answers_words=answers_words,
            device=device,
            initial_history=sample["messages"][1]["content"],
            print_debug=(step % config.log_steps == 0),
        )

        win_tracker.append(1 if rollout.solved else 0)
        rolling_win_rate = (sum(win_tracker) / len(win_tracker)) * 100 if win_tracker else 0.0
        pairs = build_grpo_pairs(rollout)

        avg_loss = -1.0
        if pairs:
            optimizer.zero_grad(set_to_none=True)
            accumulated_loss = 0.0
            valid_updates = 0

            for prompt_tokens, winner_tokens, loser_tokens in pairs:
                prompt_toks = pad_sequences([prompt_tokens], tokenizer.pad_token_id, device)
                winner_toks = pad_sequences([winner_tokens], tokenizer.pad_token_id, device)
                loser_toks = pad_sequences([loser_tokens], tokenizer.pad_token_id, device)

                loss = grpo_loss(model, winner_toks, loser_toks, prompt_toks, config, tokenizer.pad_token_id)
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"Step {step}: skipping pair due to NaN/Inf loss")
                    continue

                (loss / len(pairs)).backward()
                accumulated_loss += loss.item()
                valid_updates += 1

            if valid_updates > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grpo_clip_epsilon)
                if config.use_lr_scheduler:
                    new_lr = cosine_decay_lr(step, config.learning_rate, config.lr_min, config.lr_decay_steps)
                    for param_group in optimizer.param_groups:
                        param_group["lr"] = new_lr
                optimizer.step()
                avg_loss = accumulated_loss / valid_updates

        rewards = [att.training_reward for att in rollout.attempts]
        avg_reward = sum(rewards) / len(rewards) if rewards else 0.0
        record = {
            "step": step,
            "log_type": "train",
            "loss": avg_loss,
            "solved": rollout.solved,
            "secret_word": rollout.secret_word,
            "avg_reward": avg_reward,
            "rolling_win_rate": rolling_win_rate,
            "num_pairs": len(pairs),
        }
        train_metrics.append(record)
        with open(metrics_path, "a") as f:
            f.write(json.dumps(record) + "\n")

        if wandb_run is not None:
            import wandb

            wandb.log(
                {
                    "train/loss": avg_loss,
                    "train/avg_reward": avg_reward,
                    "train/rolling_win_rate": rolling_win_rate,
                    "train/solved": int(rollout.solved),
                    "train/num_pairs": len(pairs),
                },
                step=step,
            )

        if step % config.log_steps == 0:
            print(
                f"\nStep {step} | loss={avg_loss:.4f} | reward={avg_reward:.2f} | "
                f"train win%={rolling_win_rate:.1f} | pairs={len(pairs)}"
            )

        if step % config.checkpoint_steps == 0:
            ckpt_path = adapter_dir / f"adapter_step_{step}"
            model.save_pretrained(ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")

    final_path = adapter_dir / "adapter_final"
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"Training complete. Final adapter: {final_path}")

    if push_to_hub and hub_model_id:
        print(f"Pushing adapter to Hub: {hub_model_id}")
        model.push_to_hub(hub_model_id)
        tokenizer.push_to_hub(hub_model_id)

    if wandb_run is not None:
        import wandb

        wandb.finish()

    return final_path
