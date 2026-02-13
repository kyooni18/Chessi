#!/usr/bin/env python3
from __future__ import annotations

import json
import mimetypes
import os
import sys
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import chess

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Chessi import Chessi
from chess_lm.pgn import normalize_move_sequence
from chessgamemanager import ChessGameManager


def color_name(color: chess.Color) -> str:
    return "white" if color == chess.WHITE else "black"


class WebChessService:
    def __init__(
        self,
        repo_id: str,
        fallback_repo_id: str,
        device: str,
        top_k: int,
        temperature: float,
    ) -> None:
        self.lock = threading.RLock()
        self.repo_id = repo_id
        self.fallback_repo_id = fallback_repo_id
        self.active_repo_id = repo_id
        self.device = device
        self.top_k = max(5, int(top_k))
        self.temperature = max(0.1, float(temperature))

        self.chessi = Chessi(repo=repo_id, device=device)
        self.gm = ChessGameManager()
        self.move_history: list[str] = []
        self.human_color: chess.Color = chess.WHITE

        self.model_loading = False
        self.model_loaded = False
        self.model_error: str | None = None
        self.model_warning: str | None = None

    def load_model_async(self, force_download: bool = False) -> None:
        with self.lock:
            if self.model_loading:
                return
            self.model_loading = True
            self.model_error = None
            self.model_warning = None
            requested_repo = self.repo_id
            fallback_repo = self.fallback_repo_id
            device = self.device

        def _loader() -> None:
            try:
                self.chessi.load(
                    repo=requested_repo,
                    device=device,
                    force_download=force_download,
                )
            except Exception as exc:  # pragma: no cover - runtime/environment issue
                if fallback_repo and fallback_repo != requested_repo:
                    try:
                        self.chessi.load(
                            repo=fallback_repo,
                            device=device,
                            force_download=force_download,
                        )
                    except Exception as fallback_exc:
                        with self.lock:
                            self.model_loaded = False
                            self.model_error = (
                                f"primary={requested_repo}: {exc} | fallback={fallback_repo}: {fallback_exc}"
                            )
                    else:
                        with self.lock:
                            self.model_loaded = True
                            self.model_error = None
                            self.model_warning = (
                                f"{requested_repo} load failed, fallback loaded: {fallback_repo}"
                            )
                            self.active_repo_id = fallback_repo
                else:
                    with self.lock:
                        self.model_loaded = False
                        self.model_error = str(exc)
            else:
                with self.lock:
                    self.model_loaded = True
                    self.model_error = None
                    self.model_warning = None
                    self.active_repo_id = requested_repo
            finally:
                with self.lock:
                    self.model_loading = False

        thread = threading.Thread(target=_loader, daemon=True)
        thread.start()

    def reload_model(self, repo_id: str | None, force_download: bool) -> None:
        with self.lock:
            if repo_id:
                self.repo_id = repo_id.strip()
            self.chessi = Chessi(repo=self.repo_id, device=self.device)
            self.active_repo_id = self.repo_id
            self.model_loaded = False
            self.model_error = None
            self.model_warning = None
        self.load_model_async(force_download=force_download)

    def _board_rows(self) -> list[list[str]]:
        rows: list[list[str]] = []
        for rank in range(7, -1, -1):
            row: list[str] = []
            for file_idx in range(8):
                square = chess.square(file_idx, rank)
                piece = self.gm.board.piece_at(square)
                row.append(piece.symbol() if piece else "")
            rows.append(row)
        return rows

    def _legal_targets(self) -> dict[str, list[str]]:
        mapping: dict[str, list[str]] = {}
        for move in self.gm.board.legal_moves:
            src = chess.square_name(move.from_square)
            dst = chess.square_name(move.to_square)
            if src not in mapping:
                mapping[src] = []
            if dst not in mapping[src]:
                mapping[src].append(dst)

        for src in mapping:
            mapping[src].sort()
        return mapping

    def state_payload(self) -> dict[str, Any]:
        with self.lock:
            board = self.gm.board
            last_move_uci = board.peek().uci() if board.move_stack else None
            status = self.gm.game_status()
            return {
                "repo_id": self.repo_id,
                "active_repo_id": self.active_repo_id,
                "fallback_repo_id": self.fallback_repo_id,
                "device": self.device,
                "model": {
                    "loading": self.model_loading,
                    "loaded": self.model_loaded,
                    "error": self.model_error,
                    "warning": self.model_warning,
                },
                "game": {
                    "fen": board.fen(),
                    "turn": color_name(board.turn),
                    "human_color": color_name(self.human_color),
                    "status": {
                        "state": status.state,
                        "winner": status.winner,
                        "result": status.result,
                    },
                    "move_history": list(self.move_history),
                    "last_move_uci": last_move_uci,
                    "board_rows": self._board_rows(),
                    "legal_targets": self._legal_targets(),
                },
            }

    def new_game(self, human_color: str, ai_opening: bool) -> dict[str, Any]:
        with self.lock:
            self.gm.reset()
            self.move_history = []
            self.human_color = chess.BLACK if human_color == "black" else chess.WHITE

            ai_move: str | None = None
            if ai_opening and self.gm.board.turn != self.human_color:
                ai_move = self._play_ai_locked()

        payload = self.state_payload()
        payload["events"] = {
            "human_move": None,
            "ai_move": ai_move,
        }
        return payload

    def _select_move(
        self,
        candidates: list[chess.Move],
        promotion: str,
    ) -> chess.Move:
        if len(candidates) == 1:
            return candidates[0]

        promotion_map = {
            "q": chess.QUEEN,
            "r": chess.ROOK,
            "b": chess.BISHOP,
            "n": chess.KNIGHT,
        }
        preferred = promotion_map.get(promotion.lower(), chess.QUEEN)

        for move in candidates:
            if move.promotion == preferred:
                return move
        for move in candidates:
            if move.promotion == chess.QUEEN:
                return move
        return candidates[0]

    def _first_legal_san(self) -> str:
        for move in self.gm.board.legal_moves:
            return self.gm.board.san(move)
        raise RuntimeError("No legal moves available.")

    def _normalize_single_san(self, candidate: str) -> str | None:
        parsed = normalize_move_sequence(candidate)
        if len(parsed) != 1:
            return None
        return parsed[0]

    def _choose_ai_move(self) -> str:
        history = " ".join(self.move_history)

        self.chessi.assign_game_manager(self.gm)
        try:
            guessed = self.chessi.predict(
                moves=history,
                top_k=max(self.top_k, 20),
            )
            normalized = self._normalize_single_san(guessed) or guessed.strip()
            if self.gm.is_pan_move_legal(normalized):
                return normalized
        except Exception:
            pass

        try:
            fallback_moves = self.chessi.gen(
                moves=history,
                top_k=max(self.top_k, 40),
                temperature=self.temperature,
            )
        except Exception:
            fallback_moves = []

        for raw_san in fallback_moves:
            normalized = self._normalize_single_san(raw_san) or raw_san.strip()
            if self.gm.is_pan_move_legal(normalized):
                return normalized

        return self._first_legal_san()

    def _play_ai_locked(self) -> str:
        if self.gm.game_status().state != "ongoing":
            raise RuntimeError("Game already finished.")
        if self.gm.board.turn == self.human_color:
            raise RuntimeError("Not AI turn.")
        if not self.model_loaded:
            if self.model_loading:
                raise RuntimeError("Model is still loading. 잠시 후 다시 시도하세요.")
            if self.model_error:
                raise RuntimeError(f"Model load failed: {self.model_error}")
            raise RuntimeError("Model is not loaded yet.")

        ai_san = self._choose_ai_move()
        self.gm.apply_pan_move(ai_san)
        self.move_history.append(ai_san)
        return ai_san

    def play_ai_move(self) -> dict[str, Any]:
        with self.lock:
            ai_move = self._play_ai_locked()

        payload = self.state_payload()
        payload["events"] = {
            "human_move": None,
            "ai_move": ai_move,
        }
        return payload

    def play_human_move(
        self,
        from_square_name: str,
        to_square_name: str,
        promotion: str,
        auto_ai: bool,
    ) -> dict[str, Any]:
        from_text = from_square_name.strip().lower()
        to_text = to_square_name.strip().lower()

        with self.lock:
            if self.gm.game_status().state != "ongoing":
                raise ValueError("Game already finished.")
            if self.gm.board.turn != self.human_color:
                raise ValueError("지금은 AI 차례입니다.")

            try:
                from_square = chess.parse_square(from_text)
                to_square = chess.parse_square(to_text)
            except ValueError as exc:
                raise ValueError("잘못된 좌표입니다. 예: e2 -> e4") from exc

            candidates = [
                move
                for move in self.gm.board.legal_moves
                if move.from_square == from_square and move.to_square == to_square
            ]
            if not candidates:
                raise ValueError("불가능한 수입니다.")

            chosen_move = self._select_move(candidates, promotion)
            human_san = self.gm.board.san(chosen_move)
            self.gm.apply_pan_move(human_san)
            self.move_history.append(human_san)

            ai_move: str | None = None
            if auto_ai and self.gm.game_status().state == "ongoing" and self.gm.board.turn != self.human_color:
                ai_move = self._play_ai_locked()

        payload = self.state_payload()
        payload["events"] = {
            "human_move": human_san,
            "ai_move": ai_move,
        }
        return payload


DEFAULT_REPO = os.getenv("CHESSI_REPO", "kyooni/chessi-0.1")
DEFAULT_FALLBACK_REPO = os.getenv("CHESSI_REPO_FALLBACK", "kyooni18/chessi-0.1")
DEFAULT_DEVICE = os.getenv("CHESSI_DEVICE", "auto")
DEFAULT_TOP_K = int(os.getenv("CHESSI_TOP_K", "50"))
DEFAULT_TEMPERATURE = float(os.getenv("CHESSI_TEMPERATURE", "1.0"))

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

service = WebChessService(
    repo_id=DEFAULT_REPO,
    fallback_repo_id=DEFAULT_FALLBACK_REPO,
    device=DEFAULT_DEVICE,
    top_k=DEFAULT_TOP_K,
    temperature=DEFAULT_TEMPERATURE,
)
service.load_model_async(force_download=False)


class ChessiWebHandler(BaseHTTPRequestHandler):
    server_version = "ChessiWeb/0.1"

    def log_message(self, fmt: str, *args: Any) -> None:
        # Keep terminal noise low during local interaction.
        return

    def _send_json(self, payload: Any, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: Path, content_type: str) -> None:
        if not path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND, "File not found")
            return

        data = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _parse_json(self) -> dict[str, Any]:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        if length <= 0:
            return {}

        raw = self.rfile.read(length)
        if not raw:
            return {}

        try:
            body = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError("Invalid JSON payload") from exc

        if not isinstance(body, dict):
            raise ValueError("JSON body must be an object")
        return body

    def _handle_get_api(self, path: str) -> bool:
        if path == "/api/state":
            self._send_json(service.state_payload())
            return True

        if path == "/api/health":
            payload = service.state_payload()
            self._send_json(
                {
                    "repo_id": payload["repo_id"],
                    "active_repo_id": payload["active_repo_id"],
                    "fallback_repo_id": payload["fallback_repo_id"],
                    "device": payload["device"],
                    "model": payload["model"],
                }
            )
            return True

        return False

    def _handle_post_api(self, path: str, body: dict[str, Any]) -> bool:
        if path == "/api/reload-model":
            repo_id = body.get("repo_id")
            force_download = bool(body.get("force_download", False))
            if repo_id is not None and not isinstance(repo_id, str):
                self._send_json({"error": "repo_id must be a string"}, status=400)
                return True
            cleaned_repo = repo_id.strip() if isinstance(repo_id, str) else None
            if cleaned_repo == "":
                self._send_json({"error": "repo_id cannot be empty"}, status=400)
                return True

            service.reload_model(repo_id=cleaned_repo, force_download=force_download)
            self._send_json(service.state_payload())
            return True

        if path == "/api/new-game":
            human_color = str(body.get("human_color", "white")).lower()
            ai_opening = bool(body.get("ai_opening", True))
            if human_color not in {"white", "black"}:
                self._send_json({"error": "human_color must be 'white' or 'black'"}, status=400)
                return True

            try:
                payload = service.new_game(human_color=human_color, ai_opening=ai_opening)
            except RuntimeError as exc:
                self._send_json({"error": str(exc)}, status=409)
                return True

            self._send_json(payload)
            return True

        if path == "/api/player-move":
            from_square = body.get("from")
            to_square = body.get("to")
            promotion = str(body.get("promotion", "q"))
            auto_ai = bool(body.get("auto_ai", True))

            if not isinstance(from_square, str) or not isinstance(to_square, str):
                self._send_json({"error": "from/to fields are required"}, status=400)
                return True

            try:
                payload = service.play_human_move(
                    from_square_name=from_square,
                    to_square_name=to_square,
                    promotion=promotion,
                    auto_ai=auto_ai,
                )
            except ValueError as exc:
                self._send_json({"error": str(exc)}, status=400)
                return True
            except RuntimeError as exc:
                self._send_json({"error": str(exc)}, status=409)
                return True

            self._send_json(payload)
            return True

        if path == "/api/ai-move":
            try:
                payload = service.play_ai_move()
            except RuntimeError as exc:
                self._send_json({"error": str(exc)}, status=409)
                return True

            self._send_json(payload)
            return True

        return False

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path

        if self._handle_get_api(path):
            return

        if path == "/":
            self._send_file(STATIC_DIR / "index.html", "text/html; charset=utf-8")
            return

        if path.startswith("/static/"):
            rel_path = path[len("/static/") :]
            target = (STATIC_DIR / rel_path).resolve()
            if STATIC_DIR.resolve() not in target.parents:
                self.send_error(HTTPStatus.FORBIDDEN, "Forbidden")
                return

            content_type = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
            if content_type.startswith("text/") or content_type in {"application/javascript", "application/json"}:
                content_type = f"{content_type}; charset=utf-8"
            self._send_file(target, content_type)
            return

        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path

        try:
            body = self._parse_json()
        except ValueError as exc:
            self._send_json({"error": str(exc)}, status=400)
            return

        if self._handle_post_api(path, body):
            return

        self.send_error(HTTPStatus.NOT_FOUND, "Not found")


def main() -> None:
    host = os.getenv("CHESSI_WEB_HOST", "127.0.0.1")
    port = int(os.getenv("CHESSI_WEB_PORT", "5173"))

    server = ThreadingHTTPServer((host, port), ChessiWebHandler)
    print(f"Chessi web server running on http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
