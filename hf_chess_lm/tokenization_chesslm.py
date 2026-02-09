from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from transformers import PreTrainedTokenizer


_RESULT_TOKENS = {"1-0", "0-1", "1/2-1/2", "*"}
_MOVE_NUMBER_TOKEN_RE = re.compile(r"^\d+\.(\.\.)?$")
_MOVE_NUMBER_PREFIX_RE = re.compile(r"^\d+\.(\.\.)?")
_ANNOTATION_SUFFIX_RE = re.compile(r"[!?]+$")


def _clean_token(token: str) -> str | None:
    token = token.strip()
    if not token:
        return None
    if token in _RESULT_TOKENS or token == "...":
        return None
    if _MOVE_NUMBER_TOKEN_RE.match(token):
        return None

    token = _MOVE_NUMBER_PREFIX_RE.sub("", token)
    token = token.strip(".")
    token = token.replace("0-0-0", "O-O-O").replace("0-0", "O-O")
    token = _ANNOTATION_SUFFIX_RE.sub("", token)
    if not token or token in _RESULT_TOKENS:
        return None
    return token


class ChessLMTokenizer(PreTrainedTokenizer):
    vocab_files_names = {"vocab_file": "vocab.json"}
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file: str,
        bos_token: str = "<bos>",
        eos_token: str = "<sep>",
        sep_token: str = "<sep>",
        unk_token: str = "<unk>",
        pad_token: str = "<pad>",
        **kwargs: Any,
    ) -> None:
        self.vocab_file = str(vocab_file)
        with Path(vocab_file).open("r", encoding="utf-8") as file:
            payload = json.load(file)

        if "id_to_token" in payload:
            self.id_to_token = [str(token) for token in payload["id_to_token"]]
        elif isinstance(payload, dict):
            max_index = max(int(index) for index in payload.values())
            self.id_to_token = ["<unk>"] * (max_index + 1)
            for token, index in payload.items():
                self.id_to_token[int(index)] = str(token)
        else:
            raise ValueError("vocab_file must contain {'id_to_token': [...]} or token->id mapping.")

        self.token_to_id = {token: index for index, token in enumerate(self.id_to_token)}
        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            sep_token=sep_token,
            unk_token=unk_token,
            pad_token=pad_token,
            **kwargs,
        )

    @property
    def vocab_size(self) -> int:
        return len(self.id_to_token)

    def get_vocab(self) -> dict[str, int]:
        return dict(self.token_to_id)

    def _tokenize(self, text: str) -> list[str]:
        tokens: list[str] = []
        for raw in text.split():
            cleaned = _clean_token(raw)
            if cleaned:
                tokens.append(cleaned)
        return tokens

    def _convert_token_to_id(self, token: str) -> int:
        return self.token_to_id.get(token, self.token_to_id.get(self.unk_token, 0))

    def _convert_id_to_token(self, index: int) -> str:
        if index < 0 or index >= len(self.id_to_token):
            return self.unk_token
        return self.id_to_token[index]

    def convert_tokens_to_string(self, tokens: list[str]) -> str:
        return " ".join(tokens)

    def build_inputs_with_special_tokens(
        self,
        token_ids_0: list[int],
        token_ids_1: list[int] | None = None,
    ) -> list[int]:
        if token_ids_1 is None:
            return [self.bos_token_id, *token_ids_0, self.sep_token_id]
        return [self.bos_token_id, *token_ids_0, self.sep_token_id, *token_ids_1, self.sep_token_id]

    def create_token_type_ids_from_sequences(
        self,
        token_ids_0: list[int],
        token_ids_1: list[int] | None = None,
    ) -> list[int]:
        return [0] * len(self.build_inputs_with_special_tokens(token_ids_0, token_ids_1))

    def save_vocabulary(
        self,
        save_directory: str,
        filename_prefix: str | None = None,
    ) -> tuple[str]:
        target_dir = Path(save_directory)
        target_dir.mkdir(parents=True, exist_ok=True)
        filename = "vocab.json" if filename_prefix is None else f"{filename_prefix}-vocab.json"
        path = target_dir / filename
        with path.open("w", encoding="utf-8") as file:
            json.dump({"id_to_token": self.id_to_token}, file, indent=2)
        return (str(path),)
