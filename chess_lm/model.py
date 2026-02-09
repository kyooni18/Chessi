from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


@dataclass
class ChessLMConfig:
    vocab_size: int
    hidden_size: int = 256
    num_hidden_layers: int = 6
    num_attention_heads: int = 8
    intermediate_size: int = 1024
    dropout: float = 0.1
    max_position_embeddings: int = 256
    pad_token_id: int = 0
    bos_token_id: int = 1
    sep_token_id: int = 2

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ChessLMConfig":
        return cls(**payload)

    def save_json(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as file:
            json.dump(self.to_dict(), file, indent=2)


class ChessNextMoveModel(nn.Module):
    def __init__(self, config: ChessLMConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embeddings = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            padding_idx=config.pad_token_id,
        )
        self.position_embeddings = nn.Embedding(
            num_embeddings=config.max_position_embeddings,
            embedding_dim=config.hidden_size,
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=config.num_attention_heads,
            dim_feedforward=config.intermediate_size,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=config.num_hidden_layers,
        )
        self.final_layer_norm = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.token_embeddings.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.position_embeddings.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.02)

    @staticmethod
    def _make_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
        mask = torch.ones((seq_len, seq_len), dtype=torch.bool, device=device)
        return torch.triu(mask, diagonal=1)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, seq_len]")

        batch_size, seq_len = input_ids.shape
        if seq_len > self.config.max_position_embeddings:
            raise ValueError(
                f"Sequence length {seq_len} exceeds max_position_embeddings "
                f"{self.config.max_position_embeddings}"
            )

        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        hidden_states = self.token_embeddings(input_ids) + self.position_embeddings(position_ids)
        hidden_states = self.dropout(hidden_states)

        causal_mask = self._make_causal_mask(seq_len=seq_len, device=input_ids.device)
        key_padding_mask = (attention_mask == 0) if attention_mask is not None else None

        hidden_states = self.transformer(
            src=hidden_states,
            mask=causal_mask,
            src_key_padding_mask=key_padding_mask,
        )
        hidden_states = self.final_layer_norm(hidden_states)
        logits = self.lm_head(hidden_states)
        return logits


def gather_last_token_logits(logits: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
    if logits.ndim != 3:
        raise ValueError("logits must have shape [batch, seq_len, vocab_size]")
    batch_size = logits.size(0)

    if attention_mask is None:
        index = torch.full((batch_size,), logits.size(1) - 1, device=logits.device, dtype=torch.long)
    else:
        lengths = attention_mask.sum(dim=1).clamp(min=1)
        index = lengths - 1

    batch_index = torch.arange(batch_size, device=logits.device)
    return logits[batch_index, index, :]
