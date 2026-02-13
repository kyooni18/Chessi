# Chessi

End-to-end pipeline for a SAN-based chess next-move language model.

## 1) Data generation

```bash
python scripts/prepare_chess_data.py \
  --dataset Lichess/tournament-chess-games \
  --split train \
  --output-dir data/chess_lm \
  --max-games 50000 \
  --max-examples 500000 \
  --winner-moves-only \
  --streaming
```

Key controls:
- `--max-games`: how many games to consume
- `--max-examples`: max prefix-target pairs
- `--winner-moves-only` / `--no-winner-moves-only`: keep only eventual winner-side target moves
- `--log-file prep.log --verbose`: detailed logs

## 2) Train

```bash
python scripts/train_chess_lm.py \
  --data-dir data/chess_lm \
  --output-dir outputs/chess_lm \
  --num-epochs 6 \
  --batch-size 128 \
  --learning-rate 3e-4
```

Artifacts:
- `outputs/chess_lm/checkpoint_best.pt`
- `outputs/chess_lm/checkpoint_last.pt`
- `outputs/chess_lm/vocab.json`

## 3) Local top-k inference

```bash
python scripts/predict_next_move.py \
  --checkpoint outputs/chess_lm/checkpoint_best.pt \
  --moves "e4 e5 Nf3 Nc6 Bb5" \
  --top-k 5
```

Output format is move-only list, e.g.:

```text
[Nf6, a6, d6, Bc5, Be7]
```

## 4) PyQt GUI app

```bash
python chess_lm_gui.py
```

GUI features:
- AI vs human chess game
- click-to-move pieces on visual board
- `ChessGameManager` based game state tracking
- turn-by-turn synchronization (`Chessi.gm = GUI CGM`)
- SAN load/undo/suggest/play-AI controls for testing

## 5) Web app (human vs AI)

Export model assets for browser inference:

```bash
python scripts/export_webgpu_onnx.py \
  --repo-id kyooni18/chessi-0.1 \
  --output-dir web/model
```

Serve static files (no backend inference server):

```bash
cd web
python3 -m http.server 5173
```

Then open `http://127.0.0.1:5173/static/`.

Web behavior:
- all move generation runs on-device in the browser (`onnxruntime-web`)
- model loading indicator shown on board
- click-to-move board for human
- automatic AI response after each legal human move
- checkmate/stalemate/draw overlay shown directly from client-side game state

Optional model override at runtime:
- define `window.CHESSI_MODEL_URL` and/or `window.CHESSI_VOCAB_URL` before loading `/static/app.js`

## 6) Export to Transformers format

```bash
python scripts/export_hf_transformers.py \
  --checkpoint outputs/chess_lm/checkpoint_best.pt \
  --output-dir hf_export/chessi-0.1
```

## 7) Upload to Hugging Face

```bash
python scripts/export_hf_transformers.py \
  --checkpoint outputs/chess_lm/checkpoint_best.pt \
  --output-dir hf_export/chessi-0.1 \
  --repo-id Kyoung1229/chessi-0.1 \
  --push-to-hub
```

## 8) Use from Transformers

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

repo = "kyooni18/chessi-0.1"
tokenizer = AutoTokenizer.from_pretrained(repo, trust_remote_code=True, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(repo, trust_remote_code=True)

prompt = "e4 e5 Nf3 Nc6 Bb5"
inputs = tokenizer(prompt, return_tensors="pt")
out = model.generate(**inputs, max_new_tokens=1)
new_ids = out[:, inputs["input_ids"].shape[1]:]
print(tokenizer.decode(new_ids[0], skip_special_tokens=True))  # Nf6
```
