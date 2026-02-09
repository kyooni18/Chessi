from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from chess_lm.vocab import BOS_TOKEN, SEP_TOKEN, MoveVocab


@dataclass(frozen=True)
class NextMoveExample:
    prefix: list[str]
    target: str

    def to_dict(self) -> dict[str, object]:
        return {"prefix": self.prefix, "target": self.target}

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "NextMoveExample":
        prefix = [str(token) for token in payload["prefix"]]
        target = str(payload["target"])
        return cls(prefix=prefix, target=target)


def generate_prefix_target_pairs(
    moves: Sequence[str],
    min_prefix_len: int = 1,
    max_prefix_len: int | None = None,
) -> list[NextMoveExample]:
    if min_prefix_len < 1:
        raise ValueError("min_prefix_len must be at least 1")
    if len(moves) < min_prefix_len + 1:
        return []

    pairs: list[NextMoveExample] = []
    for index in range(min_prefix_len, len(moves)):
        start = max(0, index - max_prefix_len) if max_prefix_len else 0
        prefix = list(moves[start:index])
        target = moves[index]
        pairs.append(NextMoveExample(prefix=prefix, target=target))
    return pairs


def save_examples_jsonl(path: str | Path, examples: Iterable[NextMoveExample]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as file:
        for example in examples:
            file.write(json.dumps(example.to_dict(), ensure_ascii=True))
            file.write("\n")


def load_examples_jsonl(path: str | Path, max_examples: int | None = None) -> list[NextMoveExample]:
    source = Path(path)
    examples: list[NextMoveExample] = []
    with source.open("r", encoding="utf-8") as file:
        for line in file:
            payload = json.loads(line)
            examples.append(NextMoveExample.from_dict(payload))
            if max_examples is not None and len(examples) >= max_examples:
                break
    return examples

