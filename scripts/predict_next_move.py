#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from chess_lm.model import ChessLMConfig, ChessNextMoveModel, gather_last_token_logits
from chess_lm.pgn import normalize_move_sequence
from chess_lm.vocab import SPECIAL_TOKENS, BOS_TOKEN, SEP_TOKEN, MoveVocab


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict top-k next SAN moves from a move prefix.")
    parser.add_argument("--checkpoint", default="outputs/chess_lm/checkpoint_best.pt")
    parser.add_argument("--vocab-file", default=None, help="Optional vocab.json path.")
    parser.add_argument("--moves", required=True, help='Move prefix. Example: "e4 e5 Nf3 Nc6 Bb5"')
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--with-prob", action="store_true", help="Include probability scores.")
    parser.add_argument("--json", action="store_true", help="Emit JSON output.")
    return parser.parse_args()


def resolve_device(raw_device: str) -> torch.device:
    if raw_device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(raw_device)


def load_vocab(args: argparse.Namespace, payload: dict) -> MoveVocab:
    if "id_to_token" in payload:
        id_to_token = [str(token) for token in payload["id_to_token"]]
        token_to_id = {token: index for index, token in enumerate(id_to_token)}
        return MoveVocab(token_to_id=token_to_id, id_to_token=id_to_token)

    if args.vocab_file:
        return MoveVocab.load(args.vocab_file)

    raise RuntimeError("No vocabulary found in checkpoint, and --vocab-file was not provided.")


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    checkpoint_path = Path(args.checkpoint)
    payload = torch.load(checkpoint_path, map_location=device)

    config = ChessLMConfig.from_dict(payload["config"])
    vocab = load_vocab(args=args, payload=payload)

    model = ChessNextMoveModel(config)
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    model.eval()

    moves = normalize_move_sequence(args.moves)
    if not moves:
        raise RuntimeError("Could not parse input moves.")

    tokens = [BOS_TOKEN, *moves, SEP_TOKEN]
    input_ids = vocab.encode(tokens)
    if len(input_ids) > config.max_position_embeddings:
        input_ids = input_ids[-config.max_position_embeddings :]

    model_input = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)
    attention_mask = torch.ones_like(model_input, dtype=torch.long)

    with torch.no_grad():
        logits = model(input_ids=model_input, attention_mask=attention_mask)
        next_logits = gather_last_token_logits(logits=logits, attention_mask=attention_mask).squeeze(0)

    temperature = max(1e-4, args.temperature)
    probs = torch.softmax(next_logits / temperature, dim=-1)
    ranked_ids = torch.argsort(probs, descending=True)

    predictions: list[tuple[str, float]] = []
    for token_id in ranked_ids.tolist():
        token = vocab.id_to_token[token_id]
        if token in SPECIAL_TOKENS:
            continue
        predictions.append((token, float(probs[token_id].item())))
        if len(predictions) >= args.top_k:
            break

    if args.with_prob:
        payload = [{"move": move, "prob": prob} for move, prob in predictions]
        if args.json:
            print(json.dumps(payload, ensure_ascii=False))
        else:
            print(payload)
        return

    move_only = [move for move, _ in predictions]
    if args.json:
        print(json.dumps(move_only, ensure_ascii=False))
    else:
        print(f"[{', '.join(move_only)}]")


if __name__ == "__main__":
    main()
