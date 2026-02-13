#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class CausalLMLogitsWrapper(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        # Export int32 ONNX inputs for better browser runtime compatibility.
        outputs = self.model(
            input_ids=input_ids.to(torch.long),
            attention_mask=attention_mask.to(torch.long),
            return_dict=True,
        )
        return outputs.logits


def disable_nested_tensor_fastpath(model: nn.Module) -> None:
    # The PT2 ONNX export path can fail on TransformerEncoder nested-tensor ops.
    transformer = getattr(model, "transformer", None)
    if transformer is None:
        return
    if hasattr(transformer, "enable_nested_tensor"):
        transformer.enable_nested_tensor = False
    if hasattr(transformer, "use_nested_tensor"):
        transformer.use_nested_tensor = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export HF chess model to ONNX for on-device web inference.")
    parser.add_argument("--repo-id", default="kyooni18/chessi-0.1")
    parser.add_argument("--output-dir", default="web/model")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--dummy-seq-len", type=int, default=32)
    parser.add_argument("--force-download", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        args.repo_id,
        trust_remote_code=True,
        use_fast=False,
        force_download=args.force_download,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.repo_id,
        trust_remote_code=True,
        force_download=args.force_download,
    )
    model.eval()
    disable_nested_tensor_fastpath(model)

    wrapper = CausalLMLogitsWrapper(model)

    dummy_seq_len = max(4, int(args.dummy_seq_len))
    input_ids = torch.ones((1, dummy_seq_len), dtype=torch.int32)
    attention_mask = torch.ones((1, dummy_seq_len), dtype=torch.int32)

    onnx_path = output_dir / "chessi.onnx"
    torch.onnx.export(
        wrapper,
        (input_ids, attention_mask),
        str(onnx_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "logits": {0: "batch", 1: "seq"},
        },
        opset_version=int(args.opset),
        do_constant_folding=True,
        dynamo=False,
    )

    tokenizer.save_pretrained(output_dir)

    metadata = {
        "repo_id": args.repo_id,
        "onnx_file": "chessi.onnx",
        "vocab_file": "vocab.json",
        "input_dtype": "int32",
        "fixed_seq_len": int(dummy_seq_len),
        "bos_token_id": tokenizer.bos_token_id,
        "sep_token_id": tokenizer.sep_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "unk_token_id": tokenizer.unk_token_id,
    }
    with (output_dir / "web_model_config.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    print(f"Exported ONNX model to {onnx_path}")
    print(f"Saved tokenizer assets to {output_dir}")


if __name__ == "__main__":
    main()
