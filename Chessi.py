from __future__ import annotations

from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from chess_lm.pgn import normalize_move_sequence
from chessgamemanager import ChessGameManager


class Chessi:
    def __init__(
        self,
        repo: str = "kyooni18/chessi-0.1",
        device: str = "auto",
        force_download: bool = False,
    ) -> None:
        self.repo = repo
        self.device_name = device
        self.force_download = force_download
        self._tokenizer: Any | None = None
        self._model: Any | None = None
        self._device: torch.device | None = None
        self.gm = ChessGameManager()

    def resolve_device(self, device: str) -> torch.device:
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                return torch.device("cpu")
            return torch.device("cpu")
        return torch.device(device)

    def assign_game_manager(self, gm: ChessGameManager) -> None:
        self.gm = gm

    def load(
        self,
        repo: str | None = None,
        device: str | None = None,
        force_download: bool | None = None,
    ) -> tuple[Any, Any, torch.device]:
        if repo is not None:
            self.repo = repo
        if device is not None:
            self.device_name = device
        if force_download is not None:
            self.force_download = force_download

        resolved_device = self.resolve_device(self.device_name)
        if self._tokenizer is None or self._model is None:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.repo,
                trust_remote_code=True,
                use_fast=False,
                force_download=self.force_download,
            )
            self._model = AutoModelForCausalLM.from_pretrained(
                self.repo,
                trust_remote_code=True,
                force_download=self.force_download,
            )

        if self._device is None or self._device != resolved_device:
            self._model.to(resolved_device)
            self._model.eval()
            self._device = resolved_device

        return self._tokenizer, self._model, resolved_device

    def gen(
        self,
        moves: str,
        top_k: int = 5,
        temperature: float = 1.0,
        device: str | None = None,
        with_prob: bool = False,
        force_download: bool | None = None,
    ) -> list[str] | list[dict[str, float | str]]:
        if top_k < 1:
            raise ValueError("top_k must be >= 1")

        tokenizer, model, resolved_device = self.load(
            device=device,
            force_download=force_download,
        )

        prompt = " ".join(normalize_move_sequence(moves))
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(resolved_device)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(resolved_device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits[:, -1, :].squeeze(0)

        scaled_temperature = max(1e-4, float(temperature))
        probs = torch.softmax(logits / scaled_temperature, dim=-1)
        ranked_ids = torch.argsort(probs, descending=True).tolist()

        special_ids = set(tokenizer.all_special_ids)
        predictions: list[tuple[str, float]] = []
        seen_moves: set[str] = set()
        for token_id in ranked_ids:
            if token_id in special_ids:
                continue

            decoded = tokenizer.decode([token_id], skip_special_tokens=True).strip()
            parsed = normalize_move_sequence(decoded)
            if len(parsed) != 1:
                continue
            move = parsed[0]
            if move in seen_moves:
                continue

            seen_moves.add(move)
            predictions.append((move, float(probs[token_id].item())))
            if len(predictions) >= top_k:
                break

        if with_prob:
            return [{"move": move, "prob": prob} for move, prob in predictions]
        return [move for move, _ in predictions]

    def predict(self, moves: str, top_k: int = 10) -> str:
        predictions = self.gen(moves=moves, top_k=1000)
        if not predictions:
            raise RuntimeError("Model returned no SAN candidates.")

        for prediction in predictions:
            if self.gm.is_pan_move_legal(prediction):
                return prediction
        return predictions[0]
