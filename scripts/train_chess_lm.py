#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from chess_lm.data import NextMoveExample, load_examples_jsonl
from chess_lm.model import ChessLMConfig, ChessNextMoveModel, gather_last_token_logits
from chess_lm.vocab import BOS_TOKEN, SEP_TOKEN, MoveVocab


class PrefixTargetDataset(Dataset):
    def __init__(
        self,
        examples: list[NextMoveExample],
        vocab: MoveVocab,
        max_seq_len: int,
    ) -> None:
        self.samples: list[tuple[Tensor, int]] = []
        for example in examples:
            model_tokens = [BOS_TOKEN, *example.prefix, SEP_TOKEN]
            token_ids = vocab.encode(model_tokens)
            if len(token_ids) > max_seq_len:
                token_ids = token_ids[-max_seq_len:]
            input_ids = torch.tensor(token_ids, dtype=torch.long)
            target_id = vocab.token_to_id.get(example.target, vocab.unk_id)
            self.samples.append((input_ids, target_id))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        return self.samples[index]


def collate_batch(
    batch: list[tuple[Tensor, int]],
    pad_id: int,
) -> dict[str, Tensor]:
    batch_size = len(batch)
    max_len = max(item[0].numel() for item in batch)
    input_ids = torch.full((batch_size, max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
    labels = torch.empty((batch_size,), dtype=torch.long)

    for row, (ids, target) in enumerate(batch):
        seq_len = ids.numel()
        input_ids[row, :seq_len] = ids
        attention_mask[row, :seq_len] = 1
        labels[row] = target

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a chess next-move LM from JSONL prefix/target pairs.")
    parser.add_argument("--data-dir", default="data/chess_lm")
    parser.add_argument("--train-file", default=None)
    parser.add_argument("--val-file", default=None)
    parser.add_argument("--vocab-file", default=None)
    parser.add_argument("--output-dir", default="outputs/chess_lm")
    parser.add_argument("--max-train-examples", type=int, default=None)
    parser.add_argument("--max-val-examples", type=int, default=None)
    parser.add_argument("--max-seq-len", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--intermediate-size", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-epochs", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3.3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--log-every", type=int, default=100)
    return parser.parse_args()

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(raw_device: str) -> torch.device:
    if raw_device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(raw_device)


@torch.no_grad()
def evaluate(
    model: ChessNextMoveModel,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_examples = 0
    total_correct = 0

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        next_move_logits = gather_last_token_logits(logits=logits, attention_mask=attention_mask)
        loss = F.cross_entropy(next_move_logits, labels)

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_examples += batch_size
        predictions = next_move_logits.argmax(dim=-1)
        total_correct += (predictions == labels).sum().item()

    if total_examples == 0:
        return 0.0, 0.0
    return total_loss / total_examples, total_correct / total_examples


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)

    data_dir = Path(args.data_dir)
    train_path = Path(args.train_file) if args.train_file else data_dir / "train.jsonl"
    val_path = Path(args.val_file) if args.val_file else data_dir / "val.jsonl"
    vocab_path = Path(args.vocab_file) if args.vocab_file else data_dir / "vocab.json"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    vocab = MoveVocab.load(vocab_path)
    train_examples = load_examples_jsonl(train_path, max_examples=args.max_train_examples)
    val_examples = load_examples_jsonl(val_path, max_examples=args.max_val_examples)

    if not train_examples:
        raise RuntimeError("No training examples loaded.")
    if not val_examples:
        raise RuntimeError("No validation examples loaded.")

    train_dataset = PrefixTargetDataset(train_examples, vocab=vocab, max_seq_len=args.max_seq_len)
    val_dataset = PrefixTargetDataset(val_examples, vocab=vocab, max_seq_len=args.max_seq_len)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=lambda batch: collate_batch(batch, pad_id=vocab.pad_id),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=lambda batch: collate_batch(batch, pad_id=vocab.pad_id),
    )

    model_config = ChessLMConfig(
        vocab_size=len(vocab),
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_layers,
        num_attention_heads=args.num_heads,
        intermediate_size=args.intermediate_size,
        dropout=args.dropout,
        max_position_embeddings=args.max_seq_len,
        pad_token_id=vocab.pad_id,
        bos_token_id=vocab.bos_id,
        sep_token_id=vocab.sep_id,
    )
    model = ChessNextMoveModel(model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    best_val_loss = float("inf")
    history: list[dict[str, float | int]] = []

    for epoch in range(1, args.num_epochs + 1):
        model.train()
        running_loss = 0.0
        seen_examples = 0

        for step, batch in enumerate(train_loader, start=1):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            next_move_logits = gather_last_token_logits(logits=logits, attention_mask=attention_mask)
            loss = F.cross_entropy(next_move_logits, labels)
            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            optimizer.step()

            batch_size = labels.size(0)
            running_loss += loss.item() * batch_size
            seen_examples += batch_size

            if args.log_every > 0 and step % args.log_every == 0:
                avg_train_loss = running_loss / max(1, seen_examples)
                print(
                    f"[epoch {epoch}] step={step} train_loss={avg_train_loss:.4f} "
                    f"seen={seen_examples}",
                    flush=True,
                )

        train_loss = running_loss / max(1, seen_examples)
        val_loss, val_acc = evaluate(model=model, loader=val_loader, device=device)
        print(
            f"[epoch {epoch}] train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"val_acc={val_acc:.4f}",
            flush=True,
        )

        epoch_payload = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_acc": val_acc,
        }
        history.append(epoch_payload)

        checkpoint_payload = {
            "config": model_config.to_dict(),
            "state_dict": model.state_dict(),
            "id_to_token": vocab.id_to_token,
            "metrics": epoch_payload,
        }
        torch.save(checkpoint_payload, output_dir / "checkpoint_last.pt")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(checkpoint_payload, output_dir / "checkpoint_best.pt")

    shutil.copy2(vocab_path, output_dir / "vocab.json")
    model_config.save_json(output_dir / "model_config.json")
    with (output_dir / "train_history.json").open("w", encoding="utf-8") as file:
        json.dump(history, file, indent=2)
    with (output_dir / "train_args.json").open("w", encoding="utf-8") as file:
        json.dump(vars(args), file, indent=2)
    print(f"Saved best checkpoint to {output_dir / 'checkpoint_best.pt'}", flush=True)


if __name__ == "__main__":
    main()
