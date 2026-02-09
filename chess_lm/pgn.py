from __future__ import annotations

import re
from typing import Iterable

RESULT_TOKENS = {"1-0", "0-1", "1/2-1/2", "*"}

_TAG_LINE_RE = re.compile(r"^\s*\[.*\]\s*$")
_BRACE_COMMENT_RE = re.compile(r"\{[^}]*\}")
_SEMICOLON_COMMENT_RE = re.compile(r";[^\n]*")
_NAG_RE = re.compile(r"\$\d+")
_MOVE_NUMBER_TOKEN_RE = re.compile(r"^\d+\.(\.\.)?$")
_MOVE_NUMBER_PREFIX_RE = re.compile(r"^\d+\.(\.\.)?")
_ANNOTATION_SUFFIX_RE = re.compile(r"[!?]+$")


def _strip_variations(text: str) -> str:
    current = text
    while True:
        updated = re.sub(r"\([^()]*\)", " ", current)
        if updated == current:
            return current
        current = updated


def _clean_move_token(token: str) -> str | None:
    token = token.strip()
    if not token:
        return None
    if token in RESULT_TOKENS:
        return None
    if token.startswith("$"):
        return None
    if token == "...":
        return None
    if _MOVE_NUMBER_TOKEN_RE.match(token):
        return None

    token = _MOVE_NUMBER_PREFIX_RE.sub("", token)
    token = token.strip(".")
    token = token.replace("0-0-0", "O-O-O").replace("0-0", "O-O")
    token = _ANNOTATION_SUFFIX_RE.sub("", token)

    if not token or token in RESULT_TOKENS:
        return None
    return token


def parse_pgn_moves(pgn_text: str) -> list[str]:
    lines = [line for line in pgn_text.splitlines() if not _TAG_LINE_RE.match(line)]
    text = "\n".join(lines)
    text = _BRACE_COMMENT_RE.sub(" ", text)
    text = _SEMICOLON_COMMENT_RE.sub(" ", text)
    text = _NAG_RE.sub(" ", text)
    text = _strip_variations(text)

    tokens = text.split()
    moves: list[str] = []
    for raw in tokens:
        cleaned = _clean_move_token(raw)
        if cleaned:
            moves.append(cleaned)
    return moves


def normalize_move_sequence(move_source: str | Iterable[str]) -> list[str]:
    if isinstance(move_source, str):
        return parse_pgn_moves(move_source)

    moves: list[str] = []
    for raw in move_source:
        cleaned = _clean_move_token(str(raw))
        if cleaned:
            moves.append(cleaned)
    return moves

