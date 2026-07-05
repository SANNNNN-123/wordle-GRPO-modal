"""Wordle game logic and rewards — PyTorch/Modal port (no MLX dependencies)."""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Callable, Dict, List, Optional

import torch
from datasets import Dataset

GUESS_TAG_RE = re.compile(r"<guess>(.*?)</guess>", re.DOTALL | re.IGNORECASE)
FIVE_LETTER_WORD_RE = re.compile(r"\b([A-Z]{5})\b")

SYSTEM_PROMPT = """
You are an expert Wordle-solving AI. Your primary directive is to deduce the secret 5-letter English word with flawless logic and strategy. Adherence to the rules and format is critical.

### Core Principles
1.  **Deductive Reasoning:** Analyze all available clues from the "Current Knowledge" summary to logically eliminate possibilities.
2.  **Strategic Guessing:** In early turns, your goal is to reveal the most information. In later turns, your goal is to pinpoint the exact word.
3.  **Self-Correction & Rule Adherence:** Before finalizing a guess, ALWAYS double-check that it does not violate any Green, Yellow, or Gray clues. Your guess must be a valid 5-letter English word that has not been used before.

### Rules of Engagement
1.  **Clue Analysis:** The clues are provided in a structured "Current Knowledge" block.
    *   **Correct Position (Green):** Shows letters in their exact, confirmed positions. Your guess MUST match this pattern.
    *   **Wrong Position (Yellow):** Lists letters that are in the word. Your guess MUST include these letters.
    *   **Not in Word (Gray):** Lists letters that are not in the word. Your guess must NOT use any of these letters.
    *   **Words Already Guessed:** A list of words you cannot use again.

2.  **Chain of Thought:** You MUST explain your reasoning inside `<think>` tags. Detail your deductions from the clues, your strategy, and why your chosen word is the optimal choice.

3.  **Final Guess:** You MUST provide your final 5-letter English word guess inside `<guess>` tags.

---
### EXAMPLES
---

**Example 1: Optimal First Guess**

You are playing a game of Wordle. Analyze the clues and provide your next guess.
**Current Knowledge:**
*   **Correct Position (Green):** `_ _ _ _ _`
*   **Wrong Position (Yellow):** None
*   **Not in Word (Gray):** None
*   **Words Already Guessed:** None

<think>
This is the first guess with no prior clues. The best strategy is to use a word with common, distinct letters to maximize information gain. 'SLATE' is an excellent choice as it tests three common consonants and two common vowels.
</think>
<guess>SLATE</guess>

**Example 2: Complex Mid-Game Deduction**

You are playing a game of Wordle. Analyze the clues and provide your next guess.
**Current Knowledge:**
*   **Correct Position (Green):** `A _ _ _ _`
*   **Wrong Position (Yellow):** 'O', 'R', 'T', 'U'
*   **Not in Word (Gray):** B, E, I, S
*   **Words Already Guessed:** ARISE, ABOUT

<think>
From the clues, I have a strong set of constraints.
- The word must match the pattern `A _ _ _ _`.
- It must contain the letters O, R, T, and U in the remaining four slots.
- It must not contain the gray letters B, E, I, or S.
- It cannot be ARISE or ABOUT.
The only possible anagram of the yellow letters that fits the green pattern is 'AUTOR'. This word satisfies all known clues and is the only logical solution.
</think>
<guess>AUTOR</guess>

--- END OF EXAMPLES ---

You are now ready. The new puzzle begins. Take a deep breath and play!
""".strip()


def load_word_list(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {line.strip().upper() for line in path.read_text().splitlines() if line.strip()}


def load_word_entropy(repo_root: Path) -> dict[str, float]:
    entropy_path = repo_root / "data" / "word_entropy.json"
    if not entropy_path.exists():
        print("Warning: word_entropy.json not found; information-gain bonus will be zero.")
        return {}
    return json.loads(entropy_path.read_text())


@dataclass
class GuessFeedback:
    guess: str
    feedback: str
    is_in_dictionary: bool = True


@dataclass
class GenerationAttempt:
    prompt_string: str
    prompt_tokens: list[int]
    full_response: str
    response_tokens: list[int]
    parsed_guess: Optional[str]
    game_score: float
    training_reward: float
    feedback_given: Optional[GuessFeedback] = None


@dataclass
class GameRollout:
    attempts: List[GenerationAttempt] = field(default_factory=list)
    secret_word: str = ""
    solved: bool = False


@dataclass
class TrainConfig:
    model_name: str
    data_path: str
    iterations: int
    learning_rate: float
    use_lr_scheduler: bool
    lr_min: float
    lr_decay_steps: int
    checkpoint_steps: int
    log_steps: int
    lora_rank: int
    lora_alpha: float
    lora_dropout: float
    lora_layers: int
    num_generations: int
    max_completion_length: int
    sampling_temperature: float
    max_trials: int
    grpo_beta: float
    grpo_kl_coeff: float
    grpo_clip_epsilon: float
    reward: dict[str, float]


def load_train_config(config_path: Path, model_override: Optional[str] = None) -> TrainConfig:
    raw = json.loads(config_path.read_text())
    model_name = model_override or raw["model"]["name"]
    # Map MLX community model IDs to Hugging Face equivalents when needed.
    if model_name.startswith("mlx-community/"):
        short = model_name.removeprefix("mlx-community/").removesuffix("-bf16")
        model_name = f"google/{short}"

    training = raw["training"]
    lora = raw["lora"]
    rl = raw["rl"]
    grpo = raw["grpo"]
    return TrainConfig(
        model_name=model_name,
        data_path=training["data_path"],
        iterations=training["iterations"],
        learning_rate=training["learning_rate"],
        use_lr_scheduler=training.get("use_lr_scheduler", False),
        lr_min=training.get("lr_min", training["learning_rate"] / 10),
        lr_decay_steps=training.get("lr_decay_steps", training["iterations"]),
        checkpoint_steps=training["checkpoint_steps"],
        log_steps=training["log_steps"],
        lora_rank=lora["rank"],
        lora_alpha=lora["alpha"],
        lora_dropout=lora["dropout"],
        lora_layers=lora["layers_to_tune"],
        num_generations=rl["num_generations"],
        max_completion_length=rl["max_completion_length"],
        sampling_temperature=rl["sampling_temperature"],
        max_trials=rl["max_trials"],
        grpo_beta=grpo["beta"],
        grpo_kl_coeff=grpo["kl_coeff"],
        grpo_clip_epsilon=grpo["clip_epsilon"],
        reward=raw["reward"],
    )


def get_feedback(guess: str, secret_word: str) -> GuessFeedback:
    guess = guess.upper()
    secret_word = secret_word.upper()
    if len(guess) != 5:
        return GuessFeedback(guess, "INVALID_FORMAT")

    feedback = [""] * 5
    secret_counts = Counter(secret_word)
    for i in range(5):
        if guess[i] == secret_word[i]:
            feedback[i] = "G"
            secret_counts[guess[i]] -= 1
    for i in range(5):
        if feedback[i] == "":
            if guess[i] in secret_counts and secret_counts[guess[i]] > 0:
                feedback[i] = "Y"
                secret_counts[guess[i]] -= 1
            else:
                feedback[i] = "X"
    return GuessFeedback(guess, " ".join(feedback))


def parse_guess(response: str) -> Optional[str]:
    match = GUESS_TAG_RE.search(response)
    if not match:
        return None
    words = FIVE_LETTER_WORD_RE.findall(match.group(1))
    return words[-1] if words else None


def is_valid_guess(guess: str, allowed_words: set[str]) -> bool:
    return bool(guess and len(guess) == 5 and guess.isalpha() and guess.upper() in allowed_words)


def get_clue_summary(guesses: List[str], feedback: List[str]) -> Dict:
    greens = ["_"] * 5
    yellows: set[str] = set()
    greys: set[str] = set()
    yellow_positions = {chr(ord("A") + i): set() for i in range(26)}
    for guess, fb in zip(guesses, feedback):
        guess = guess.upper()
        fb = fb.replace(" ", "")
        for i, (letter, status) in enumerate(zip(guess, fb)):
            if status == "G":
                greens[i] = letter
                if letter in yellows:
                    yellows.remove(letter)
            elif status == "Y":
                yellows.add(letter)
                yellow_positions[letter].add(i)
            elif status == "X" and letter not in "".join(greens) and letter not in yellows:
                greys.add(letter)
    return {"greens": greens, "yellows": yellows, "greys": greys, "yellow_positions": yellow_positions}


def find_valid_completions(clues: Dict, word_list: List[str]) -> List[str]:
    valid_words = []
    for word in word_list:
        word = word.upper()
        is_valid = True
        word_counts = Counter(word)
        for i, letter in enumerate(clues["greens"]):
            if letter != "_" and word[i] != letter:
                is_valid = False
                break
        if not is_valid:
            continue
        for letter in clues["greys"]:
            if letter in word_counts:
                is_valid = False
                break
        if not is_valid:
            continue
        if not all(letter in word_counts for letter in clues["yellows"]):
            continue
        for letter, positions in clues["yellow_positions"].items():
            for pos in positions:
                if word[pos] == letter:
                    is_valid = False
                    break
            if not is_valid:
                break
        if is_valid:
            valid_words.append(word)
    return valid_words


def format_prompt_for_model(past_feedback: List[GuessFeedback], system_prompt: str) -> List[dict]:
    if not past_feedback:
        user_content = "This is the first turn. Please provide your best starting word."
        return [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_content}]

    known_green: dict[int, str] = {}
    known_yellow = Counter()
    known_gray: set[str] = set()
    for fb in past_feedback:
        counts_in_secret_this_turn = Counter()
        for i, f_char in enumerate(fb.feedback.split()):
            if f_char in ("G", "Y"):
                counts_in_secret_this_turn[fb.guess[i]] += 1
        for letter, count in counts_in_secret_this_turn.items():
            known_yellow[letter] = max(known_yellow[letter], count)
        for i, f_char in enumerate(fb.feedback.split()):
            letter = fb.guess[i]
            if f_char == "G":
                known_green[i] = letter
            elif f_char == "X" and counts_in_secret_this_turn[letter] == 0:
                known_gray.add(letter)

    for letter in set(known_green.values()):
        if letter in known_yellow:
            del known_yellow[letter]
        if letter in known_gray:
            known_gray.remove(letter)

    parts = [
        "You are playing a game of Wordle. Analyze the clues and provide your next guess.",
        "**Current Knowledge:**",
    ]
    green_display = ["_"] * 5
    for idx, letter in known_green.items():
        green_display[idx] = letter
    parts.append(f"*   **Correct Position (Green):** `{' '.join(green_display)}`")
    if known_yellow:
        yellow_display = [f"'{k}' (at least {v})" for k, v in sorted(known_yellow.items())]
        parts.append(f"*   **Wrong Position (Yellow):** {', '.join(yellow_display)}")
    else:
        parts.append("*   **Wrong Position (Yellow):** None")
    if known_gray:
        parts.append(f"*   **Not in Word (Gray):** {', '.join(sorted(known_gray))}")
    else:
        parts.append("*   **Not in Word (Gray):** None")
    parts.append(f"*   **Words Already Guessed:** {', '.join(fb.guess for fb in past_feedback)}")
    parts.append(
        "\nYour task is to find a valid 5-letter English word that fits all the clues above."
        "\nProvide your reasoning within <think> tags, and then your final guess within <guess> tags."
    )
    return [{"role": "system", "content": system_prompt}, {"role": "user", "content": "\n".join(parts)}]


def parse_history_from_prompt(user_prompt: str, secret_word: str) -> List[GuessFeedback]:
    match = re.search(r"\*   \*\*Words Already Guessed:\*\* (.+)", user_prompt)
    if not match:
        return []
    guessed_words_str = match.group(1)
    if guessed_words_str.lower() == "none":
        return []
    past_feedback = []
    for guess in [w.strip() for w in guessed_words_str.split(",")]:
        if guess:
            past_feedback.append(get_feedback(guess, secret_word))
    return past_feedback


def calculate_total_reward(
    response: str,
    secret_word: str,
    past_feedback: List[GuessFeedback],
    config: TrainConfig,
    allowed_words: set[str],
    tokenizer,
    word_entropy: dict[str, float],
    answers_words: list[str],
) -> tuple[float, float]:
    reward_config = config.reward
    guess = parse_guess(response)
    time_penalty = -reward_config.get("time_penalty_per_guess", 0.0)
    length_penalty = 0.0
    if tokenizer and response:
        length_penalty = -(len(tokenizer.encode(response)) * reward_config.get("length_penalty_per_token", 0.0))

    if not guess:
        game_score = -reward_config.get("format_fail_penalty", 200.0)
        return game_score, game_score + time_penalty + length_penalty

    if guess == secret_word.upper():
        game_score = reward_config.get("solution_correct_guess", 150.0)
        return game_score, game_score + time_penalty + length_penalty

    if any(fb.guess == guess for fb in past_feedback):
        game_score = -reward_config.get("repetition_penalty", 40.0)
        return game_score, game_score + time_penalty + length_penalty

    known_green: dict[int, str] = {}
    known_yellow = Counter()
    known_gray: set[str] = set()
    for fb in past_feedback:
        counts_in_secret_this_turn = Counter()
        for i in range(5):
            letter = fb.guess[i]
            if fb.feedback.split()[i] in ("G", "Y"):
                counts_in_secret_this_turn[letter] += 1
        for letter, count in counts_in_secret_this_turn.items():
            known_yellow[letter] = max(known_yellow[letter], count)
        for i in range(5):
            letter = fb.guess[i]
            feedback = fb.feedback.split()[i]
            if feedback == "G":
                known_green[i] = letter
            elif feedback == "X" and counts_in_secret_this_turn[letter] == 0:
                known_gray.add(letter)

    for letter in set(known_green.values()):
        if letter in known_yellow:
            del known_yellow[letter]
        if letter in known_gray:
            known_gray.remove(letter)

    green_violations = sum(1 for idx, correct_letter in known_green.items() if guess[idx] != correct_letter)
    guess_counts = Counter(guess)
    yellow_violations = sum(
        1 for yellow_letter, required_count in known_yellow.items() if guess_counts[yellow_letter] < required_count
    )
    gray_violations = sum(1 for letter_in_guess in set(guess) if letter_in_guess in known_gray)

    total_penalty = (
        green_violations * reward_config.get("green_position_penalty", 30.0)
        + yellow_violations * reward_config.get("yellow_letter_penalty", 20.0)
        + gray_violations * reward_config.get("gray_letter_penalty", 20.0)
    )
    if not is_valid_guess(guess, allowed_words):
        total_penalty += reward_config.get("not_in_dictionary_penalty", 35.0)

    stagnation_penalty = 0.0
    for idx, letter in known_green.items():
        if guess[idx] == letter:
            stagnation_penalty += reward_config.get("green_reuse_penalty", 3.0)
    for letter in set(guess):
        if letter in known_yellow:
            stagnation_penalty += reward_config.get("yellow_reuse_penalty", 1.5)
    total_penalty += stagnation_penalty

    turn_number = len(past_feedback) + 1
    if turn_number == 1:
        strategic_bonus = word_entropy.get(guess.upper(), 0.0) * reward_config.get("information_gain_bonus_coeff", 7.5)
    else:
        known_letters = {letter for fb in past_feedback for letter in fb.guess.upper()}
        strategic_bonus = len(set(guess.upper()) - known_letters) * reward_config.get("new_letter_bonus", 2.0)

    current_feedback = get_feedback(guess, secret_word)
    reduction_bonus = 0.0
    if past_feedback:
        clues_before = get_clue_summary(
            [f.guess for f in past_feedback],
            [f.feedback for f in past_feedback],
        )
        possibilities_before = find_valid_completions(clues_before, answers_words)
        combined = past_feedback + [current_feedback]
        clues_after = get_clue_summary(
            [f.guess for f in combined],
            [f.feedback for f in combined],
        )
        possibilities_after = find_valid_completions(clues_after, answers_words)
        if possibilities_before:
            reduction_fraction = (len(possibilities_before) - len(possibilities_after)) / len(possibilities_before)
            reduction_bonus = reduction_fraction * reward_config.get("possibility_reduction_bonus", 15.0)

    potential_score = reward_config.get("valid_guess_base", 15.0) + strategic_bonus + reduction_bonus
    game_score = potential_score - total_penalty
    training_reward = game_score + time_penalty + length_penalty
    return game_score, training_reward


def load_wordle_dataset(dataset_path: Path) -> Dataset:
    trajectories = []
    with open(dataset_path) as f:
        for line in f:
            if not line.strip():
                continue
            data_point = json.loads(line)
            data_content = data_point.get("data", {})
            secret_word = data_content.get("secret")
            messages = data_content.get("messages")
            if secret_word and messages:
                trajectories.append({"secret": secret_word.upper(), "messages": messages})
    if not trajectories:
        raise ValueError(f"No trajectories loaded from {dataset_path}")
    return Dataset.from_list(trajectories)


def _is_playable_trajectory(example, max_trials: int) -> bool:
    secret_word = example["secret"].upper()
    assistant_messages = [m for m in example["messages"] if m.get("role") == "assistant"]
    if len(assistant_messages) >= max_trials:
        return False
    for message in assistant_messages:
        guess = parse_guess(message.get("content", ""))
        if guess and guess == secret_word:
            return False
    return True


def prepare_data(config: TrainConfig, repo_root: Path, seed: int = 42):
    dataset_path = repo_root / config.data_path
    dataset = load_wordle_dataset(dataset_path)
    playable_filter = partial(_is_playable_trajectory, max_trials=config.max_trials)
    playable = dataset.filter(playable_filter)
    shuffled = playable.shuffle(seed=seed)
    train_end = int(0.70 * len(shuffled))
    validation_end = int(0.85 * len(shuffled))
    return (
        shuffled.select(range(0, train_end)),
        shuffled.select(range(train_end, validation_end)),
        shuffled.select(range(validation_end, len(shuffled))),
    )


@torch.no_grad()
def generate_responses(
    model,
    tokenizer,
    prompt_string: str,
    num_generations: int,
    max_new_tokens: int,
    temperature: float,
    device: torch.device,
) -> list[str]:
    # The chat template already injects Gemma's special tokens (<bos>, <start_of_turn>,
    # etc.). Passing add_special_tokens=True here prepends a SECOND <bos>, which Gemma
    # is very sensitive to and can push sampled decoding into degenerate token loops.
    inputs = tokenizer(prompt_string, return_tensors="pt", add_special_tokens=False).to(device)
    gen_kwargs: dict = {
        "max_new_tokens": max_new_tokens,
        "num_return_sequences": num_generations,
        "pad_token_id": tokenizer.pad_token_id,
        # Stop at end-of-turn so rollouts don't ramble past the guess.
        "eos_token_id": tokenizer.eos_token_id,
    }
    # Eval uses temperature=0 (greedy). Newer transformers rejects temp=0 with do_sample=True.
    if temperature <= 0.0:
        gen_kwargs["do_sample"] = False
    else:
        # Bare temperature sampling on Gemma 3 collapses into '<<<>>>' special-token
        # spam. Constrain the distribution the way the MLX sampler / HF recommend:
        # nucleus + top-k + a mild repetition penalty.
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = 0.95
        gen_kwargs["top_k"] = 64
        gen_kwargs["repetition_penalty"] = 1.1

    # Rollouts should generate with dropout OFF (the policy is otherwise in train()
    # mode for the GRPO backward pass). Restore the prior mode afterwards.
    was_training = model.training
    model.eval()
    try:
        outputs = model.generate(**inputs, **gen_kwargs)
    finally:
        if was_training:
            model.train()

    prompt_len = inputs["input_ids"].shape[1]
    responses = []
    for seq in outputs:
        response_ids = seq[prompt_len:]
        text = tokenizer.decode(response_ids, skip_special_tokens=True).strip()
        responses.append(text)
    return responses


def play_wordle_game(
    model,
    tokenizer,
    secret_word: str,
    config: TrainConfig,
    allowed_words: set[str],
    word_entropy: dict[str, float],
    answers_words: list[str],
    device: torch.device,
    initial_history: str = "",
    print_debug: bool = False,
    is_eval: bool = False,
) -> GameRollout:
    rollout = GameRollout(secret_word=secret_word)
    past_feedback = parse_history_from_prompt(initial_history, secret_word) if initial_history else []
    already_guessed_words = {fb.guess for fb in past_feedback}
    num_generations = 1 if is_eval else config.num_generations

    for attempt_num in range(len(past_feedback), config.max_trials):
        messages = format_prompt_for_model(past_feedback, SYSTEM_PROMPT)
        prompt_string = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        # add_special_tokens=False: the chat template already contains <bos>; avoid a
        # double <bos> in the tokens fed to the GRPO loss.
        prompt_tokens = tokenizer.encode(prompt_string, add_special_tokens=False)

        generations = generate_responses(
            model,
            tokenizer,
            prompt_string,
            num_generations,
            config.max_completion_length,
            0.0 if is_eval else config.sampling_temperature,
            device,
        )

        current_turn_attempts: List[GenerationAttempt] = []
        for i, response in enumerate(generations):
            guess = parse_guess(response)
            game_score, training_reward = calculate_total_reward(
                response,
                secret_word,
                past_feedback,
                config,
                allowed_words,
                tokenizer,
                word_entropy,
                answers_words,
            )
            if print_debug:
                num_resp_tokens = len(tokenizer.encode(response))
                has_open = "<guess>" in response.lower()
                has_close = "</guess>" in response.lower()
                print(
                    f"  [Gen {i + 1}/{num_generations}] guess={guess} reward={training_reward:.2f} "
                    f"| tokens={num_resp_tokens} guess_tag(open/close)={has_open}/{has_close}"
                )
                if guess is None:
                    # Surface why parsing failed: show the tail where the guess should be.
                    print(f"    RAW[:800]: {response[:800]!r}")
                    print(f"    RAW[-300:]: {response[-300:]!r}")
            current_turn_attempts.append(
                GenerationAttempt(
                    prompt_string=prompt_string,
                    prompt_tokens=prompt_tokens,
                    full_response=response,
                    response_tokens=tokenizer.encode(response, add_special_tokens=False),
                    parsed_guess=guess,
                    game_score=game_score,
                    training_reward=training_reward,
                )
            )

        valid_candidates = [
            att
            for att in current_turn_attempts
            if att.parsed_guess and att.parsed_guess not in already_guessed_words
        ]
        if not valid_candidates:
            rollout.attempts.extend(current_turn_attempts)
            break

        best_attempt = max(valid_candidates, key=lambda att: att.game_score) if not is_eval else valid_candidates[0]
        best_guess = best_attempt.parsed_guess
        feedback = get_feedback(best_guess, secret_word)
        feedback.is_in_dictionary = is_valid_guess(best_guess, allowed_words)
        already_guessed_words.add(best_guess)
        best_attempt.feedback_given = feedback
        rollout.attempts.extend(current_turn_attempts)
        past_feedback.append(feedback)

        if best_guess == secret_word.upper():
            rollout.solved = True
            break

    return rollout


def build_grpo_pairs(rollout: GameRollout) -> list[tuple[list[int], list[int], list[int]]]:
    grouped = defaultdict(list)
    for attempt in rollout.attempts:
        grouped[attempt.prompt_string].append(attempt)

    pairs = []
    for attempts_for_prompt in grouped.values():
        if len(attempts_for_prompt) < 2:
            continue
        winner = max(attempts_for_prompt, key=lambda att: att.training_reward)
        for loser in attempts_for_prompt:
            if loser is not winner:
                pairs.append((winner.prompt_tokens, winner.response_tokens, loser.response_tokens))
    return pairs
