#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from chess_lm.inference import ChessMovePredictor


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


def main() -> None:
    args = parse_args()
    predictor = ChessMovePredictor(
        checkpoint=Path(args.checkpoint),
        vocab_file=args.vocab_file,
        device=args.device,
    )
    predictions = predictor.predict(
        moves_text=args.moves,
        top_k=args.top_k,
        temperature=args.temperature,
    )

    if args.with_prob:
        predicted_payload = [{"move": item.move, "prob": item.prob} for item in predictions]
        if args.json:
            print(json.dumps(predicted_payload, ensure_ascii=False))
        else:
            print(predicted_payload)
        return

    move_only = [item.move for item in predictions]
    if args.json:
        print(json.dumps(move_only, ensure_ascii=False))
    else:
        print(f"[{', '.join(move_only)}]")


if __name__ == "__main__":
    main()
