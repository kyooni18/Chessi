from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from chess_lm.model import ChessLMConfig, ChessNextMoveModel, gather_last_token_logits
from chess_lm.pgn import normalize_move_sequence
from chess_lm.vocab import BOS_TOKEN, SEP_TOKEN, SPECIAL_TOKENS, MoveVocab


def resolve_device(raw_device: str) -> torch.device:
    if raw_device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(raw_device)


@dataclass(frozen=True)
class MovePrediction:
    move: str
    prob: float


def load_vocab_from_checkpoint_payload(
    payload: dict,
    vocab_file: str | Path | None = None,
) -> MoveVocab:
    if "id_to_token" in payload:
        id_to_token = [str(token) for token in payload["id_to_token"]]
        token_to_id = {token: index for index, token in enumerate(id_to_token)}
        return MoveVocab(token_to_id=token_to_id, id_to_token=id_to_token)

    if vocab_file:
        return MoveVocab.load(vocab_file)

    raise RuntimeError("No vocabulary found in checkpoint, and vocab_file was not provided.")


class ChessMovePredictor:
    def __init__(
        self,
        checkpoint: str | Path,
        vocab_file: str | Path | None = None,
        device: str = "auto",
    ) -> None:
        self.checkpoint = Path(checkpoint)
        self.vocab_file = Path(vocab_file) if vocab_file else None
        self.device = resolve_device(device)
        self.model: ChessNextMoveModel | None = None
        self.config: ChessLMConfig | None = None
        self.vocab: MoveVocab | None = None

    def load(self) -> None:
        payload = torch.load(self.checkpoint, map_location=self.device)
        self.config = ChessLMConfig.from_dict(payload["config"])
        self.vocab = load_vocab_from_checkpoint_payload(payload, vocab_file=self.vocab_file)

        model = ChessNextMoveModel(self.config)
        model.load_state_dict(payload["state_dict"])
        model.to(self.device)
        model.eval()
        self.model = model

    def predict(
        self,
        moves_text: str,
        top_k: int = 5,
        temperature: float = 1.0,
    ) -> list[MovePrediction]:
        if self.model is None or self.config is None or self.vocab is None:
            self.load()

        assert self.model is not None
        assert self.config is not None
        assert self.vocab is not None

        moves = normalize_move_sequence(moves_text)
        if not moves:
            raise RuntimeError("Could not parse input moves.")

        tokens = [BOS_TOKEN, *moves, SEP_TOKEN]
        input_ids = self.vocab.encode(tokens)
        if len(input_ids) > self.config.max_position_embeddings:
            input_ids = input_ids[-self.config.max_position_embeddings :]

        model_input = torch.tensor(input_ids, dtype=torch.long, device=self.device).unsqueeze(0)
        attention_mask = torch.ones_like(model_input, dtype=torch.long)

        with torch.no_grad():
            logits = self.model(input_ids=model_input, attention_mask=attention_mask)
            next_logits = gather_last_token_logits(logits=logits, attention_mask=attention_mask).squeeze(0)

        scaled_temperature = max(1e-4, temperature)
        probs = torch.softmax(next_logits / scaled_temperature, dim=-1)
        ranked_ids = torch.argsort(probs, descending=True)

        predictions: list[MovePrediction] = []
        for token_id in ranked_ids.tolist():
            token = self.vocab.id_to_token[token_id]
            if token in SPECIAL_TOKENS:
                continue
            predictions.append(MovePrediction(move=token, prob=float(probs[token_id].item())))
            if len(predictions) >= top_k:
                break
        return predictions
