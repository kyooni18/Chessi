from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import chess


@dataclass(frozen=True)
class GameStatus:
    state: str
    winner: Optional[str]
    result: str


class ChessGameManager:
    """Manage a chess game using PAN/SAN move notation."""

    def __init__(self, fen: Optional[str] = None) -> None:
        self.board = chess.Board(fen) if fen else chess.Board()

    def reset(self, fen: Optional[str] = None) -> None:
        self.board = chess.Board(fen) if fen else chess.Board()

    def board_to_string(self) -> str:
        return str(self.board)

    def is_pan_move_legal(self, pan_move: str, fen: Optional[str] = None) -> bool:
        board = chess.Board(fen) if fen else self.board.copy(stack=False)
        try:
            board.parse_san(pan_move)
        except ValueError:
            return False
        return True

    def apply_pan_move(self, pan_move: str) -> chess.Move:
        try:
            move = self.board.parse_san(pan_move)
        except ValueError as exc:
            raise ValueError(f"Illegal PAN/SAN move for current position: {pan_move}") from exc
        self.board.push(move)
        return move

    def game_status(self) -> GameStatus:
        outcome = self.board.outcome(claim_draw=True)
        if outcome is None:
            return GameStatus(state="ongoing", winner=None, result="*")

        if outcome.termination == chess.Termination.CHECKMATE:
            winner = "white" if outcome.winner else "black"
            return GameStatus(state="checkmate", winner=winner, result=outcome.result())

        if outcome.termination == chess.Termination.STALEMATE:
            return GameStatus(state="stalemate", winner=None, result=outcome.result())

        if outcome.termination == chess.Termination.INSUFFICIENT_MATERIAL:
            return GameStatus(state="insufficient_material", winner=None, result=outcome.result())

        if outcome.termination == chess.Termination.FIFTY_MOVES:
            return GameStatus(state="fifty_moves", winner=None, result=outcome.result())

        if outcome.termination == chess.Termination.THREEFOLD_REPETITION:
            return GameStatus(state="threefold_repetition", winner=None, result=outcome.result())

        if outcome.termination == chess.Termination.SEVENTYFIVE_MOVES:
            return GameStatus(state="seventyfive_moves", winner=None, result=outcome.result())

        if outcome.termination == chess.Termination.FIVEFOLD_REPETITION:
            return GameStatus(state="fivefold_repetition", winner=None, result=outcome.result())

        if outcome.termination == chess.Termination.VARIANT_WIN:
            winner = "white" if outcome.winner else "black"
            return GameStatus(state="variant_win", winner=winner, result=outcome.result())

        if outcome.termination == chess.Termination.VARIANT_LOSS:
            winner = "white" if outcome.winner else "black"
            return GameStatus(state="variant_loss", winner=winner, result=outcome.result())

        if outcome.termination == chess.Termination.VARIANT_DRAW:
            return GameStatus(state="variant_draw", winner=None, result=outcome.result())

        return GameStatus(state="draw", winner=None, result=outcome.result())

    def print_board_after_each_move(self, moves: Iterable[str]) -> None:
        print(self.board_to_string())
        for index, pan_move in enumerate(moves, start=1):
            self.apply_pan_move(pan_move)
            print(f"\nMove {index}: {pan_move}")
            print(self.board_to_string())