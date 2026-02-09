from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

PAD_TOKEN = "<pad>"
BOS_TOKEN = "<bos>"
SEP_TOKEN = "<sep>"
UNK_TOKEN = "<unk>"

SPECIAL_TOKENS = [PAD_TOKEN, BOS_TOKEN, SEP_TOKEN, UNK_TOKEN]


@dataclass(frozen=True)
class MoveVocab:
    token_to_id: dict[str, int]
    id_to_token: list[str]

    @classmethod
    def build(
        cls,
        sequences: Iterable[Sequence[str]],
        min_freq: int = 1,
        max_size: int | None = None,
    ) -> "MoveVocab":
        if max_size is not None and max_size < len(SPECIAL_TOKENS):
            raise ValueError(f"max_size must be >= {len(SPECIAL_TOKENS)}")

        counter: Counter[str] = Counter()
        for seq in sequences:
            counter.update(seq)

        candidates = [
            token
            for token, count in counter.items()
            if count >= min_freq and token not in SPECIAL_TOKENS
        ]
        candidates.sort(key=lambda token: (-counter[token], token))

        if max_size is not None:
            room = max_size - len(SPECIAL_TOKENS)
            candidates = candidates[:room]

        id_to_token = [*SPECIAL_TOKENS, *candidates]
        token_to_id = {token: index for index, token in enumerate(id_to_token)}
        return cls(token_to_id=token_to_id, id_to_token=id_to_token)

    @classmethod
    def load(cls, path: str | Path) -> "MoveVocab":
        with Path(path).open("r", encoding="utf-8") as file:
            payload = json.load(file)
        id_to_token = payload["id_to_token"]
        token_to_id = {token: idx for idx, token in enumerate(id_to_token)}
        return cls(token_to_id=token_to_id, id_to_token=id_to_token)

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {"id_to_token": self.id_to_token}
        with target.open("w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2)

    def encode(self, tokens: Sequence[str]) -> list[int]:
        return [self.token_to_id.get(token, self.unk_id) for token in tokens]

    def decode(self, token_ids: Sequence[int], skip_special: bool = False) -> list[str]:
        decoded: list[str] = []
        for token_id in token_ids:
            if token_id < 0 or token_id >= len(self.id_to_token):
                token = UNK_TOKEN
            else:
                token = self.id_to_token[token_id]
            if skip_special and token in SPECIAL_TOKENS:
                continue
            decoded.append(token)
        return decoded

    @property
    def pad_id(self) -> int:
        return self.token_to_id[PAD_TOKEN]

    @property
    def bos_id(self) -> int:
        return self.token_to_id[BOS_TOKEN]

    @property
    def sep_id(self) -> int:
        return self.token_to_id[SEP_TOKEN]

    @property
    def unk_id(self) -> int:
        return self.token_to_id[UNK_TOKEN]

    def __len__(self) -> int:
        return len(self.id_to_token)

