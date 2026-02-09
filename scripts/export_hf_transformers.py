#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hf_chess_lm.configuration_chesslm import ChessLMConfig
from hf_chess_lm.modeling_chesslm import ChessLMForCausalLM
from hf_chess_lm.tokenization_chesslm import ChessLMTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export trained checkpoint to Hugging Face Transformers format.")
    parser.add_argument("--checkpoint", default="outputs/chess_lm/checkpoint_best.pt")
    parser.add_argument("--vocab-file", default=None, help="Optional fallback vocab file path.")
    parser.add_argument("--output-dir", default="hf_export/chessi-0.1")
    parser.add_argument("--repo-id", default=None, help="Hugging Face repo id, e.g. Kyoung1229/chessi-0.1")
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--commit-message", default="Update chess model")
    parser.add_argument("--token", default=None, help="Optional HF token. Uses local login if omitted.")
    return parser.parse_args()


def write_vocab(output_dir: Path, checkpoint_payload: dict, fallback_vocab_file: str | None) -> Path:
    vocab_path = output_dir / "vocab.json"
    if "id_to_token" in checkpoint_payload:
        with vocab_path.open("w", encoding="utf-8") as file:
            json.dump({"id_to_token": checkpoint_payload["id_to_token"]}, file, indent=2)
        return vocab_path

    if not fallback_vocab_file:
        raise RuntimeError("Checkpoint does not include vocabulary; provide --vocab-file.")

    source = Path(fallback_vocab_file)
    shutil.copy2(source, vocab_path)
    return vocab_path


def copy_remote_code(output_dir: Path) -> None:
    source_dir = PROJECT_ROOT / "hf_chess_lm"
    for filename in [
        "__init__.py",
        "configuration_chesslm.py",
        "modeling_chesslm.py",
        "tokenization_chesslm.py",
    ]:
        shutil.copy2(source_dir / filename, output_dir / filename)


def patch_config_for_auto_map(output_dir: Path) -> None:
    config_path = output_dir / "config.json"
    with config_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    payload["auto_map"] = {
        "AutoConfig": "configuration_chesslm.ChessLMConfig",
        "AutoModelForCausalLM": "modeling_chesslm.ChessLMForCausalLM",
        "AutoTokenizer": ["tokenization_chesslm.ChessLMTokenizer", None],
    }
    payload["tokenizer_class"] = "ChessLMTokenizer"
    with config_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


def patch_tokenizer_config(output_dir: Path) -> None:
    tokenizer_config_path = output_dir / "tokenizer_config.json"
    with tokenizer_config_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    payload["auto_map"] = {
        "AutoTokenizer": ["tokenization_chesslm.ChessLMTokenizer", None],
    }
    payload["tokenizer_class"] = "ChessLMTokenizer"
    payload["use_fast"] = False
    with tokenizer_config_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


def save_readme(output_dir: Path, repo_id: str | None) -> None:
    target_name = repo_id or output_dir.name
    readme = f"""# {target_name}

Chess next-move language model trained on SAN move sequences.

## Usage

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

repo = "{target_name}"
tokenizer = AutoTokenizer.from_pretrained(repo, trust_remote_code=True, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(repo, trust_remote_code=True)

prompt = "e4 e5 Nf3 Nc6 Bb5"
inputs = tokenizer(prompt, return_tensors="pt")
out = model.generate(**inputs, max_new_tokens=1)
new_ids = out[:, inputs["input_ids"].shape[1]:]
print(tokenizer.decode(new_ids[0], skip_special_tokens=True))
```
"""
    with (output_dir / "README.md").open("w", encoding="utf-8") as file:
        file.write(readme)


def push_to_hub(
    output_dir: Path,
    repo_id: str,
    private: bool,
    commit_message: str,
    token: str | None,
) -> None:
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, private=private, exist_ok=True)
    api.upload_folder(
        repo_id=repo_id,
        folder_path=str(output_dir),
        commit_message=commit_message,
    )


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = torch.load(checkpoint_path, map_location="cpu")
    model_cfg = payload["config"]
    hf_config = ChessLMConfig(
        vocab_size=model_cfg["vocab_size"],
        hidden_size=model_cfg["hidden_size"],
        num_hidden_layers=model_cfg["num_hidden_layers"],
        num_attention_heads=model_cfg["num_attention_heads"],
        intermediate_size=model_cfg["intermediate_size"],
        dropout=model_cfg["dropout"],
        max_position_embeddings=model_cfg["max_position_embeddings"],
        pad_token_id=model_cfg["pad_token_id"],
        bos_token_id=model_cfg["bos_token_id"],
        eos_token_id=model_cfg["sep_token_id"],
        sep_token_id=model_cfg["sep_token_id"],
        tie_word_embeddings=False,
    )

    model = ChessLMForCausalLM(hf_config)
    missing_keys, unexpected_keys = model.load_state_dict(payload["state_dict"], strict=False)
    if missing_keys:
        raise RuntimeError(f"Missing keys when loading checkpoint: {missing_keys}")
    if unexpected_keys:
        raise RuntimeError(f"Unexpected keys when loading checkpoint: {unexpected_keys}")

    vocab_path = write_vocab(output_dir=output_dir, checkpoint_payload=payload, fallback_vocab_file=args.vocab_file)
    tokenizer = ChessLMTokenizer(vocab_file=str(vocab_path), use_fast=False)

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    copy_remote_code(output_dir)
    patch_config_for_auto_map(output_dir)
    patch_tokenizer_config(output_dir)
    save_readme(output_dir=output_dir, repo_id=args.repo_id)

    print(f"Exported model to {output_dir}", flush=True)

    if args.push_to_hub:
        if not args.repo_id:
            raise RuntimeError("--repo-id is required with --push-to-hub")
        push_to_hub(
            output_dir=output_dir,
            repo_id=args.repo_id,
            private=args.private,
            commit_message=args.commit_message,
            token=args.token,
        )
        print(f"Pushed to https://huggingface.co/{args.repo_id}", flush=True)


if __name__ == "__main__":
    main()
