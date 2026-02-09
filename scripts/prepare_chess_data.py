#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import re
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import chess
import datasets
from datasets import load_dataset

# Allow running this file directly without installing the package.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from chess_lm.data import (
    NextMoveExample,
    generate_prefix_target_pairs,
    save_examples_jsonl,
)
from chess_lm.pgn import normalize_move_sequence
from chess_lm.vocab import MoveVocab

_RESULT_TAG_RE = re.compile(r'\[Result\s+"(1-0|0-1|1/2-1/2|\*)"\]', re.IGNORECASE)


def disable_datasets_torch_integration() -> None:
    """
    Avoid datasets streaming calling torch shared-memory APIs on environments
    where torch_shm_manager execution is restricted.
    """
    datasets.config.TORCH_AVAILABLE = False


def setup_logger(log_level: str, log_file: str | None) -> logging.Logger:
    logger = logging.getLogger("prepare_chess_data")
    logger.handlers.clear()
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def load_dataset_with_retries(
    dataset_name: str,
    split: str,
    streaming: bool,
    retries: int,
    retry_sleep: int,
    logger: logging.Logger,
) -> Any:
    last_error: Exception | None = None
    max_retries = max(1, retries)

    for attempt in range(1, max_retries + 1):
        try:
            return load_dataset(dataset_name, split=split, streaming=streaming)
        except Exception as exc:
            last_error = exc
            if attempt < max_retries:
                logger.warning("load_dataset failed (attempt %d/%d): %s", attempt, max_retries, exc)
                time.sleep(max(0, retry_sleep))

    raise RuntimeError(
        f"Failed to load dataset={dataset_name} split={split} streaming={streaming} "
        f"after {max_retries} attempt(s)"
    ) from last_error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build train/val/test files for chess next-move prediction.",
    )
    parser.add_argument("--dataset", default="Lichess/tournament-chess-games")
    parser.add_argument("--split", default="train")
    parser.add_argument("--output-dir", default="data/chess_lm")
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--max-games", type=int, default=50000)
    parser.add_argument("--max-examples", type=int, default=500000)
    parser.add_argument("--min-prefix-len", type=int, default=1)
    parser.add_argument("--max-prefix-len", type=int, default=120)
    parser.add_argument("--min-token-freq", type=int, default=2)
    parser.add_argument("--max-vocab-size", type=int, default=None)
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--test-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--verbose", action="store_true", help="Shortcut for --log-level DEBUG")
    parser.add_argument("--log-file", default=None, help="Path to write detailed logs.")
    parser.add_argument(
        "--log-seconds",
        type=int,
        default=30,
        help="Also log progress every N seconds (helps when streaming is slow).",
    )
    parser.add_argument(
        "--load-retries",
        type=int,
        default=5,
        help="Retries for load_dataset() in case of transient network errors.",
    )
    parser.add_argument(
        "--load-retry-sleep",
        type=int,
        default=3,
        help="Seconds to sleep between load_dataset() retries.",
    )
    parser.add_argument(
        "--stream-read-retries",
        type=int,
        default=5,
        help="Retries while iterating streaming dataset when transient errors occur.",
    )
    parser.add_argument(
        "--winner-moves-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Keep only target moves played by the side that eventually won the game. "
            "Default: enabled."
        ),
    )
    parser.add_argument(
        "--no-legality-check",
        action="store_true",
        help="Skip SAN legality validation for each game.",
    )
    return parser.parse_args()


def is_legal_san_sequence(moves: Iterable[str]) -> bool:
    board = chess.Board()
    for move in moves:
        try:
            board.push_san(move)
        except ValueError:
            return False
    return True


def _extract_moves_from_value(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return normalize_move_sequence(value)
    if isinstance(value, list):
        if value and isinstance(value[0], dict):
            return []
        return normalize_move_sequence([str(item) for item in value])
    if isinstance(value, tuple):
        return normalize_move_sequence([str(item) for item in value])
    return []


def _parse_winner_value(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"white", "w", "1-0"}:
        return "white"
    if text in {"black", "b", "0-1"}:
        return "black"
    return None


def _extract_winner_from_result_text(text: str) -> str | None:
    normalized = text.strip().lower()
    if normalized == "1-0":
        return "white"
    if normalized == "0-1":
        return "black"
    match = _RESULT_TAG_RE.search(text)
    if match:
        token = match.group(1)
        return _parse_winner_value(token)
    # Also catch plain result tokens that may appear in text blobs.
    if re.search(r"(^|\s)1-0($|\s)", text):
        return "white"
    if re.search(r"(^|\s)0-1($|\s)", text):
        return "black"
    return None


def _iter_nested_values(value: Any) -> Iterable[Any]:
    if isinstance(value, dict):
        for nested_value in value.values():
            yield nested_value
            yield from _iter_nested_values(nested_value)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield item
            yield from _iter_nested_values(item)


def _get_value_case_insensitive(record: dict[str, Any], key_name: str) -> Any:
    key_lc = key_name.lower()
    for key, value in record.items():
        if str(key).lower() == key_lc:
            return value
    return None


def extract_winner_from_record(record: dict[str, Any]) -> str | None:
    priority_keys = ("winner", "result", "outcome", "game_result", "termination")
    for key in priority_keys:
        value = _get_value_case_insensitive(record, key)
        if value is None:
            continue
        winner = _parse_winner_value(value)
        if winner is not None:
            return winner
        if isinstance(value, str):
            winner = _extract_winner_from_result_text(value)
            if winner is not None:
                return winner

    for key in ("pgn", "game", "game_pgn", "notation", "movetext", "moves"):
        value = _get_value_case_insensitive(record, key)
        if isinstance(value, str):
            winner = _extract_winner_from_result_text(value)
            if winner is not None:
                return winner
        if isinstance(value, (list, tuple)) and value:
            last_item = value[-1]
            if isinstance(last_item, str):
                winner = _extract_winner_from_result_text(last_item)
                if winner is not None:
                    return winner

    # Last resort: scan nested values (headers/tags/metadata) recursively.
    for nested in _iter_nested_values(record):
        if isinstance(nested, str):
            winner = _extract_winner_from_result_text(nested)
            if winner is not None:
                return winner
    return None


def filter_examples_to_winner_moves(
    examples: list[NextMoveExample],
    winner: str,
    min_prefix_len: int,
) -> list[NextMoveExample]:
    filtered: list[NextMoveExample] = []
    for offset, example in enumerate(examples):
        target_index = min_prefix_len + offset
        target_side = "white" if target_index % 2 == 0 else "black"
        if target_side == winner:
            filtered.append(example)
    return filtered


def extract_moves_from_record(record: dict[str, Any]) -> list[str]:
    priority_keys = (
        "moves",
        "pgn",
        "game",
        "movetext",
        "game_pgn",
        "notation",
    )
    for key in priority_keys:
        value = _get_value_case_insensitive(record, key)
        if value is not None:
            moves = _extract_moves_from_value(value)
            if len(moves) >= 2:
                return moves

    for value in record.values():
        moves = _extract_moves_from_value(value)
        if len(moves) >= 2:
            return moves
    return []


def split_dataset(
    examples: list[NextMoveExample],
    val_ratio: float,
    test_ratio: float,
) -> tuple[list[NextMoveExample], list[NextMoveExample], list[NextMoveExample]]:
    total = len(examples)
    test_count = int(total * test_ratio)
    val_count = int(total * val_ratio)
    train_count = total - val_count - test_count
    if train_count <= 0:
        raise ValueError("Not enough examples after split. Lower val_ratio/test_ratio.")

    test_examples = examples[:test_count]
    val_examples = examples[test_count : test_count + val_count]
    train_examples = examples[test_count + val_count :]
    return train_examples, val_examples, test_examples


def main() -> None:
    args = parse_args()
    if args.verbose:
        args.log_level = "DEBUG"

    logger = setup_logger(log_level=args.log_level, log_file=args.log_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Loading dataset=%s split=%s streaming=%s",
        args.dataset,
        args.split,
        args.streaming,
    )
    if args.streaming:
        disable_datasets_torch_integration()
    try:
        dataset = load_dataset_with_retries(
            dataset_name=args.dataset,
            split=args.split,
            streaming=args.streaming,
            retries=args.load_retries,
            retry_sleep=args.load_retry_sleep,
            logger=logger,
        )
    except RuntimeError as exc:
        if args.streaming and "torch_shm_manager" in str(exc):
            disable_datasets_torch_integration()
            dataset = load_dataset_with_retries(
                dataset_name=args.dataset,
                split=args.split,
                streaming=True,
                retries=args.load_retries,
                retry_sleep=args.load_retry_sleep,
                logger=logger,
            )
        elif (not args.streaming) and ("An error occurred while generating the dataset" in str(exc)):
            logger.warning("Non-streaming load failed; retrying with streaming=True fallback.")
            args.streaming = True
            disable_datasets_torch_integration()
            dataset = load_dataset_with_retries(
                dataset_name=args.dataset,
                split=args.split,
                streaming=True,
                retries=args.load_retries,
                retry_sleep=args.load_retry_sleep,
                logger=logger,
            )
        else:
            raise

    legality_check = not args.no_legality_check
    randomizer = random.Random(args.seed)

    examples: list[NextMoveExample] = []
    records_seen = 0
    games_used = 0
    winner_filtered_games = 0
    winner_filtered_examples = 0
    last_log_ts = time.time()
    stream_failures = 0

    iterator = iter(dataset)
    while True:
        try:
            record = next(iterator)
        except StopIteration:
            break
        except Exception as exc:
            if args.streaming and stream_failures < args.stream_read_retries:
                stream_failures += 1
                logger.warning(
                    "Streaming read failed (%d/%d): %s",
                    stream_failures,
                    args.stream_read_retries,
                    exc,
                )
                time.sleep(1)
                iterator = iter(
                    load_dataset_with_retries(
                        dataset_name=args.dataset,
                        split=args.split,
                        streaming=True,
                        retries=args.load_retries,
                        retry_sleep=args.load_retry_sleep,
                        logger=logger,
                    )
                )
                continue
            raise

        records_seen += 1
        if not isinstance(record, dict):
            continue

        moves = extract_moves_from_record(record)
        if len(moves) < args.min_prefix_len + 1:
            logger.debug("Skipping record %d: too short moves len=%d", records_seen, len(moves))
            continue
        if legality_check and not is_legal_san_sequence(moves):
            logger.debug("Skipping record %d: illegal SAN sequence", records_seen)
            continue

        game_examples = generate_prefix_target_pairs(
            moves=moves,
            min_prefix_len=args.min_prefix_len,
            max_prefix_len=args.max_prefix_len,
        )
        if args.winner_moves_only:
            winner = extract_winner_from_record(record)
            if winner is None:
                continue
            game_examples = filter_examples_to_winner_moves(
                examples=game_examples,
                winner=winner,
                min_prefix_len=args.min_prefix_len,
            )
            winner_filtered_games += 1
            winner_filtered_examples += len(game_examples)
            if not game_examples:
                logger.debug("Skipping record %d: no winner-side examples", records_seen)
                continue

        examples.extend(game_examples)
        games_used += 1

        if args.log_every > 0 and records_seen % args.log_every == 0:
            logger.info(
                "records_seen=%d games_used=%d examples=%d",
                records_seen,
                games_used,
                len(examples),
            )
        if args.log_seconds > 0 and (time.time() - last_log_ts) >= args.log_seconds:
            last_log_ts = time.time()
            logger.info(
                "[heartbeat] records_seen=%d games_used=%d examples=%d",
                records_seen,
                games_used,
                len(examples),
            )

        if args.max_games is not None and games_used >= args.max_games:
            break
        if args.max_examples is not None and len(examples) >= args.max_examples:
            examples = examples[: args.max_examples]
            break

    if not examples:
        raise RuntimeError("No usable examples produced. Try disabling legality check.")

    randomizer.shuffle(examples)
    train_examples, val_examples, test_examples = split_dataset(
        examples=examples,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )

    train_path = output_dir / "train.jsonl"
    val_path = output_dir / "val.jsonl"
    test_path = output_dir / "test.jsonl"

    save_examples_jsonl(train_path, train_examples)
    save_examples_jsonl(val_path, val_examples)
    save_examples_jsonl(test_path, test_examples)

    vocab_sequences = [example.prefix + [example.target] for example in train_examples]
    vocab = MoveVocab.build(
        sequences=vocab_sequences,
        min_freq=args.min_token_freq,
        max_size=args.max_vocab_size,
    )
    vocab_path = output_dir / "vocab.json"
    vocab.save(vocab_path)

    metadata = {
        "dataset": args.dataset,
        "split": args.split,
        "streaming": args.streaming,
        "records_seen": records_seen,
        "games_used": games_used,
        "examples_total": len(examples),
        "train_examples": len(train_examples),
        "val_examples": len(val_examples),
        "test_examples": len(test_examples),
        "vocab_size": len(vocab),
        "min_prefix_len": args.min_prefix_len,
        "max_prefix_len": args.max_prefix_len,
        "min_token_freq": args.min_token_freq,
        "max_vocab_size": args.max_vocab_size,
        "legality_check": legality_check,
        "winner_moves_only": args.winner_moves_only,
        "winner_filtered_games": winner_filtered_games,
        "winner_filtered_examples": winner_filtered_examples,
        "seed": args.seed,
    }

    metadata_path = output_dir / "metadata.json"
    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    logger.info("Saved files:")
    logger.info("  %s", train_path)
    logger.info("  %s", val_path)
    logger.info("  %s", test_path)
    logger.info("  %s", vocab_path)
    logger.info("  %s", metadata_path)
    logger.info(
        "Done. train=%d val=%d test=%d",
        len(train_examples),
        len(val_examples),
        len(test_examples),
    )


if __name__ == "__main__":
    main()
