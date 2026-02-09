from __future__ import annotations

from transformers import PretrainedConfig


class ChessLMConfig(PretrainedConfig):
    model_type = "chesslm"

    def __init__(
        self,
        vocab_size: int = 1024,
        hidden_size: int = 256,
        num_hidden_layers: int = 6,
        num_attention_heads: int = 8,
        intermediate_size: int = 1024,
        dropout: float = 0.1,
        max_position_embeddings: int = 256,
        pad_token_id: int = 0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        sep_token_id: int = 2,
        tie_word_embeddings: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            sep_token_id=sep_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.dropout = dropout
        self.max_position_embeddings = max_position_embeddings
