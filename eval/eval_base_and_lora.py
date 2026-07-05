"""Side-by-side Wordle eval: base Gemma-3 4B vs trained LoRA adapter."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from wordle_core import (
    TrainConfig,
    load_train_config,
    load_word_entropy,
    load_word_list,
    play_wordle_game,
    prepare_data,
)


@dataclass
class GameEvalResult:
    model_name: str
    secret_word: str
    solved: bool
    num_turns: int
    invalid_guesses: int
    guesses: list[str] = field(default_factory=list)
    with_history: bool = True


@dataclass
class GameComparison:
    game_index: int
    secret_word: str
    with_history: bool
    base: GameEvalResult
    lora: GameEvalResult

    @property
    def outcome(self) -> str:
        if self.base.solved and self.lora.solved:
            return "both_won"
        if self.lora.solved and not self.base.solved:
            return "lora_only"
        if self.base.solved and not self.lora.solved:
            return "base_only"
        return "neither"


def _load_base_model(config: TrainConfig, device: torch.device):
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=dtype,
        device_map={"": device},
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


def _load_lora_model(config: TrainConfig, adapter_path: Path, device: torch.device):
    base, tokenizer = _load_base_model(config, device)
    model = PeftModel.from_pretrained(base, str(adapter_path))
    model.eval()
    return model, tokenizer


def _guesses_from_rollout(rollout) -> list[str]:
    """Guesses that advanced the game, in turn order."""
    guesses = []
    for att in rollout.attempts:
        if att.feedback_given is not None and att.parsed_guess:
            guesses.append(att.parsed_guess)
    return guesses


def _play_single_game(
    model,
    tokenizer,
    model_name: str,
    sample: dict,
    config: TrainConfig,
    repo_root: Path,
    device: torch.device,
    with_history: bool,
) -> GameEvalResult:
    allowed_words = load_word_list(repo_root / "data" / "nyt_possible_wordle_list.txt")
    word_entropy = load_word_entropy(repo_root)
    answers_words = sorted(load_word_list(repo_root / "data" / "nyt_answers_wordle_list.txt"))
    history = sample["messages"][1]["content"] if with_history else ""

    rollout = play_wordle_game(
        model=model,
        tokenizer=tokenizer,
        secret_word=sample["secret"],
        config=config,
        allowed_words=allowed_words,
        word_entropy=word_entropy,
        answers_words=answers_words,
        device=device,
        initial_history=history,
        is_eval=True,
    )
    turns = len({att.prompt_string for att in rollout.attempts})
    invalid = sum(1 for att in rollout.attempts if att.parsed_guess is None)

    return GameEvalResult(
        model_name=model_name,
        secret_word=rollout.secret_word,
        solved=rollout.solved,
        num_turns=turns,
        invalid_guesses=invalid,
        guesses=_guesses_from_rollout(rollout),
        with_history=with_history,
    )


def _summarize(results: list[GameEvalResult]) -> dict:
    n = len(results)
    if n == 0:
        return {"games": 0, "wins": 0, "win_rate": 0.0, "avg_turns_on_win": 0.0, "invalid_guess_rate": 0.0}
    wins = [r for r in results if r.solved]
    total_turns = sum(r.num_turns for r in results)
    return {
        "games": n,
        "wins": len(wins),
        "win_rate": round(len(wins) / n * 100, 1),
        "avg_turns_on_win": round(sum(r.num_turns for r in wins) / len(wins), 2) if wins else 0.0,
        "invalid_guess_rate": round(sum(r.invalid_guesses for r in results) / max(total_turns, 1) * 100, 1),
    }


def _summarize_comparisons(comparisons: list[GameComparison]) -> dict:
    base_results = [c.base for c in comparisons]
    lora_results = [c.lora for c in comparisons]
    outcomes = [c.outcome for c in comparisons]
    return {
        "base": _summarize(base_results),
        "lora": _summarize(lora_results),
        "outcomes": {
            "both_won": outcomes.count("both_won"),
            "lora_only": outcomes.count("lora_only"),
            "base_only": outcomes.count("base_only"),
            "neither": outcomes.count("neither"),
        },
        "delta_win_rate": round(
            _summarize(lora_results)["win_rate"] - _summarize(base_results)["win_rate"], 1
        ),
    }


def _comparison_to_dict(comparison: GameComparison) -> dict:
    return {
        "game_index": comparison.game_index,
        "secret_word": comparison.secret_word,
        "with_history": comparison.with_history,
        "outcome": comparison.outcome,
        "base": asdict(comparison.base),
        "lora": asdict(comparison.lora),
    }


def save_eval_report(report: dict, output_dir: Path) -> Path:
    """Write summary.json and per_game.jsonl under output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(report, indent=2))

    per_game_path = output_dir / "per_game.jsonl"
    with open(per_game_path, "w") as f:
        for condition in report.get("conditions", {}).values():
            for game in condition.get("per_game", []):
                f.write(json.dumps(game) + "\n")

    return output_dir


def run_side_by_side_eval(
    repo_root: Path,
    config_path: Path,
    adapter_path: Path,
    output_dir: Path | None = None,
    num_samples: int = 50,
    with_history: bool = True,
    also_without_history: bool = False,
) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = load_train_config(config_path)
    _, _, test_dataset = prepare_data(config, repo_root)

    print(f"Device: {device}")
    print(f"Adapter: {adapter_path}")
    print(f"Test games per condition: {num_samples}")

    base_model, tokenizer = _load_base_model(config, device)
    lora_model, _ = _load_lora_model(config, adapter_path, device)

    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    if output_dir is None:
        output_dir = adapter_path.parent.parent / "eval" / "results" / f"base_and_lora_{num_samples}games_{run_id}"

    report: dict = {
        "run_id": run_id,
        "adapter_path": str(adapter_path),
        "num_samples": num_samples,
        "model": config.model_name,
        "conditions": {},
    }

    history_flags = [True, False] if also_without_history else [with_history]
    for history_flag in history_flags:
        label = "with_history" if history_flag else "without_history"
        subset = test_dataset.shuffle(seed=42).select(range(min(num_samples, len(test_dataset))))
        comparisons: list[GameComparison] = []

        for i, sample in enumerate(tqdm(subset, desc=f"paired eval ({label})")):
            base_result = _play_single_game(
                base_model, tokenizer, "base", sample, config, repo_root, device, history_flag
            )
            lora_result = _play_single_game(
                lora_model, tokenizer, "lora", sample, config, repo_root, device, history_flag
            )
            comparisons.append(
                GameComparison(
                    game_index=i,
                    secret_word=sample["secret"],
                    with_history=history_flag,
                    base=base_result,
                    lora=lora_result,
                )
            )

        stats = _summarize_comparisons(comparisons)
        report["conditions"][label] = {
            **stats,
            "per_game": [_comparison_to_dict(c) for c in comparisons],
        }

    saved_dir = save_eval_report(report, output_dir)
    report["output_dir"] = str(saved_dir)

    print("\n" + "=" * 60)
    print("SIDE-BY-SIDE EVAL SUMMARY")
    print("=" * 60)
    for label, stats in report["conditions"].items():
        print(f"\n--- {label.replace('_', ' ').title()} ---")
        print(f"  Base: win_rate={stats['base']['win_rate']}%  wins={stats['base']['wins']}/{stats['base']['games']}")
        print(f"  LoRA: win_rate={stats['lora']['win_rate']}%  wins={stats['lora']['wins']}/{stats['lora']['games']}")
        print(f"  Delta (LoRA - Base): {stats['delta_win_rate']:+.1f}%")
        o = stats["outcomes"]
        print(f"  Outcomes: both_won={o['both_won']}  lora_only={o['lora_only']}  "
              f"base_only={o['base_only']}  neither={o['neither']}")
    print(f"\nSaved to: {saved_dir}")
    print(f"  - summary.json")
    print(f"  - per_game.jsonl")
    print("=" * 60)

    return report
