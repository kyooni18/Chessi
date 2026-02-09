from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GenerationMixin, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutput

from hf_chess_lm.configuration_chesslm import ChessLMConfig


class ChessLMForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = ChessLMConfig
    main_input_name = "input_ids"

    def __init__(self, config: ChessLMConfig) -> None:
        super().__init__(config)
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
        self.post_init()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.token_embeddings

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.token_embeddings = value

    def get_output_embeddings(self) -> nn.Linear:
        return self.lm_head

    def set_output_embeddings(self, new_embeddings: nn.Linear) -> None:
        self.lm_head = new_embeddings

    @staticmethod
    def _make_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
        mask = torch.ones((seq_len, seq_len), dtype=torch.bool, device=device)
        return torch.triu(mask, diagonal=1)

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.LongTensor | None = None,
        labels: torch.LongTensor | None = None,
        return_dict: bool | None = None,
        **kwargs,
    ) -> CausalLMOutput | tuple[torch.Tensor, ...]:
        del kwargs
        if input_ids is None:
            raise ValueError("input_ids is required")
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

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        use_return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if not use_return_dict:
            output: tuple[torch.Tensor, ...] = (logits,)
            if loss is not None:
                output = (loss, *output)
            return output

        return CausalLMOutput(loss=loss, logits=logits)

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor | None = None,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        del kwargs
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        return {"input_ids": input_ids, "attention_mask": attention_mask}
