# Eval outputs

Side-by-side eval (base vs LoRA) lives in this folder:

```
modal-grpo/eval/
├── modal-eval-gemma3-4B.py   # Modal launcher
├── eval_base_and_lora.py     # eval logic
├── README.md
└── results/                  # downloaded / local copies of eval reports
    ├── training_results.md   # training run logs
    ├── eval_results.md       # base vs LoRA eval stats
    └── <timestamp>_gemma-3-4b-it-<N>steps/
        ├── summary.json
        └── per_game.jsonl
```

New evals write to the Modal volume under each training run:

```
/models/runs/<timestamp>_gemma-3-4b-it/
└── eval/
    └── results/
        └── base_and_lora_<N>games_<eval_timestamp>/
            ├── summary.json      # aggregate stats + full per_game list
            └── per_game.jsonl    # one JSON object per game (easy to grep)
```

## Run eval

```sh
cd modal-grpo

uv run modal run eval/modal-eval-gemma3-4B.py::evaluate_sxs \
  --adapter-path /models/runs/20260704-060851_gemma-3-4b-it/adapters/adapter_final \
  --num-samples 25
```

## `summary.json` fields

| Field | Meaning |
|-------|---------|
| `conditions.with_history.base` | Base model stats |
| `conditions.with_history.lora` | LoRA stats |
| `conditions.with_history.outcomes` | both_won / lora_only / base_only / neither counts |
| `conditions.with_history.delta_win_rate` | LoRA win% − Base win% |
| `conditions.with_history.per_game` | Full per-game breakdown |

## `per_game.jsonl` (one line per game)

Each line includes:

- `secret_word` — answer for that puzzle
- `outcome` — `lora_only`, `base_only`, `both_won`, or `neither`
- `base.guesses` / `lora.guesses` — turn-by-turn guesses played
- `base.solved` / `lora.solved` — win or loss

## Download a report (optional)

```sh
mkdir -p ./eval/results/<timestamp>_gemma-3-4b-it-<N>steps

uv run modal volume get wordle-grpo-checkpoints \
  runs/<timestamp>_gemma-3-4b-it/eval/results/base_and_lora_25games_<eval_timestamp>/summary.json \
  ./eval/results/<timestamp>_gemma-3-4b-it-<N>steps/summary.json

uv run modal volume get wordle-grpo-checkpoints \
  runs/<timestamp>_gemma-3-4b-it/eval/results/base_and_lora_25games_<eval_timestamp>/per_game.jsonl \
  ./eval/results/<timestamp>_gemma-3-4b-it-<N>steps/per_game.jsonl
```

See `results/training_results.md` for training logs and `results/eval_results.md` for eval stats.
