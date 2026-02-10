#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from PyQt6.QtCore import Qt
    from PyQt6.QtGui import QColor, QFont
    from PyQt6.QtWidgets import (
        QApplication,
        QCheckBox,
        QComboBox,
        QDoubleSpinBox,
        QFormLayout,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QSpinBox,
        QTableWidget,
        QTableWidgetItem,
        QVBoxLayout,
        QWidget,
    )
except ImportError as exc:
    raise SystemExit("PyQt6 is not installed. Install with: pip install PyQt6") from exc

import chess

from Chessi import Chessi
from chess_lm.pgn import normalize_move_sequence
from chessgamemanager import ChessGameManager

PIECE_UNICODE = {
    "P": "♙",
    "N": "♘",
    "B": "♗",
    "R": "♖",
    "Q": "♕",
    "K": "♔",
    "p": "♟",
    "n": "♞",
    "b": "♝",
    "r": "♜",
    "q": "♛",
    "k": "♚",
}


class BoardWidget(QTableWidget):
    LIGHT = QColor("#f0d9b5")
    DARK = QColor("#b58863")
    SELECTED = QColor("#f7ec5b")
    TARGET = QColor("#8fd175")
    LAST_MOVE = QColor("#89b4fa")

    def __init__(self) -> None:
        super().__init__(8, 8)
        self.verticalHeader().setVisible(False)
        self.horizontalHeader().setVisible(False)
        self.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        font = QFont("Arial Unicode MS", 24)

        for row in range(8):
            self.setRowHeight(row, 60)
            for col in range(8):
                self.setColumnWidth(col, 60)
                item = QTableWidgetItem("")
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                item.setFont(font)
                self.setItem(row, col, item)

    @staticmethod
    def square_from_cell(row: int, col: int) -> chess.Square:
        return chess.square(col, 7 - row)

    @staticmethod
    def cell_from_square(square: chess.Square) -> tuple[int, int]:
        return 7 - chess.square_rank(square), chess.square_file(square)

    def render_position(
        self,
        board: chess.Board,
        selected_square: chess.Square | None = None,
        legal_targets: set[chess.Square] | None = None,
        last_move: chess.Move | None = None,
    ) -> None:
        legal_targets = legal_targets or set()

        for row in range(8):
            for col in range(8):
                square = self.square_from_cell(row, col)
                piece = board.piece_at(square)
                symbol = PIECE_UNICODE[piece.symbol()] if piece else ""
                item = self.item(row, col)
                assert item is not None
                item.setText(symbol)
                base_color = self.LIGHT if (row + col) % 2 == 0 else self.DARK
                item.setBackground(base_color)

        if last_move is not None:
            for square in [last_move.from_square, last_move.to_square]:
                row, col = self.cell_from_square(square)
                item = self.item(row, col)
                assert item is not None
                item.setBackground(self.LAST_MOVE)

        if selected_square is not None:
            row, col = self.cell_from_square(selected_square)
            item = self.item(row, col)
            assert item is not None
            item.setBackground(self.SELECTED)

        for square in legal_targets:
            row, col = self.cell_from_square(square)
            item = self.item(row, col)
            assert item is not None
            item.setBackground(self.TARGET)


class ChessLMWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Chessi - Play vs AI")
        self.resize(1240, 760)

        self.chessi = Chessi()
        self.gm = ChessGameManager()
        self.move_history: list[str] = []
        self.last_move: chess.Move | None = None
        self.selected_square: chess.Square | None = None
        self.current_targets: set[chess.Square] = set()
        self.last_predictions: list[tuple[str, float]] = []

        self.repo_input = QLineEdit("kyooni18/chessi-0.1")
        self.device_input = QComboBox()
        self.device_input.addItems(["auto", "cpu", "mps", "cuda"])
        self.force_download_input = QCheckBox("Force download")
        self.top_k_input = QSpinBox()
        self.top_k_input.setRange(1, 100)
        self.top_k_input.setValue(50)
        self.temperature_input = QDoubleSpinBox()
        self.temperature_input.setRange(0.1, 5.0)
        self.temperature_input.setSingleStep(0.1)
        self.temperature_input.setValue(1.0)
        self.human_color_input = QComboBox()
        self.human_color_input.addItems(["white", "black"])
        self.auto_ai_input = QCheckBox("Auto AI response")
        self.auto_ai_input.setChecked(True)

        self.moves_input = QLineEdit()
        self.moves_input.setPlaceholderText("e4 e5 Nf3 Nc6 Bb5")
        self.output_line = QLineEdit()
        self.output_line.setReadOnly(True)
        self.fen_label = QLineEdit()
        self.fen_label.setReadOnly(True)
        self.turn_label = QLineEdit()
        self.turn_label.setReadOnly(True)
        self.status_label = QLabel("Ready")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

        self.board_widget = BoardWidget()
        self.board_widget.cellClicked.connect(self._on_board_click)

        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["Move", "Probability"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.cellDoubleClicked.connect(self._play_prediction_row)

        self._build_layout()
        self._start_new_game(play_ai_opening=False)

    def _build_layout(self) -> None:
        main = QWidget()
        layout = QHBoxLayout(main)

        left_panel = QVBoxLayout()
        right_panel = QVBoxLayout()

        form = QFormLayout()
        form.addRow("HF Repo", self.repo_input)
        form.addRow("Device", self.device_input)
        form.addRow("", self.force_download_input)
        form.addRow("Human Color", self.human_color_input)
        form.addRow("", self.auto_ai_input)
        form.addRow("Temperature", self.temperature_input)
        form.addRow("Moves (SAN)", self.moves_input)
        left_panel.addLayout(form)

        row1 = QHBoxLayout()
        load_button = QPushButton("Load Model")
        load_button.clicked.connect(self._load_model)
        row1.addWidget(load_button)

        new_game_button = QPushButton("New Game")
        new_game_button.clicked.connect(lambda: self._start_new_game(play_ai_opening=True))
        row1.addWidget(new_game_button)

        undo_button = QPushButton("Undo")
        undo_button.clicked.connect(self._undo_move)
        row1.addWidget(undo_button)
        left_panel.addLayout(row1)

        row2 = QHBoxLayout()
        load_moves_button = QPushButton("Load Moves")
        load_moves_button.clicked.connect(self._load_moves_from_text)
        row2.addWidget(load_moves_button)

        suggest_button = QPushButton("Suggest")
        suggest_button.clicked.connect(self._suggest_moves)
        row2.addWidget(suggest_button)

        play_ai_button = QPushButton("Play AI Move")
        play_ai_button.clicked.connect(self._play_ai_move)
        row2.addWidget(play_ai_button)
        left_panel.addLayout(row2)

        left_panel.addWidget(QLabel("Predictions (double-click row to play move)"))
        left_panel.addWidget(self.output_line)
        left_panel.addWidget(self.table)
        left_panel.addWidget(self.status_label)

        right_panel.addWidget(QLabel("Board (click piece -> click target)"))
        right_panel.addWidget(self.board_widget)
        right_panel.addWidget(QLabel("Turn"))
        right_panel.addWidget(self.turn_label)
        right_panel.addWidget(QLabel("FEN"))
        right_panel.addWidget(self.fen_label)

        layout.addLayout(left_panel, 3)
        layout.addLayout(right_panel, 2)
        self.setCentralWidget(main)

    def _set_status(self, text: str) -> None:
        self.status_label.setText(text)

    def _sync_chessi_manager(self) -> None:
        # Required by user: assign GUI CGM to Chessi on each turn.
        self.chessi.assign_game_manager(self.gm)

    def _history_to_text(self) -> str:
        return " ".join(self.move_history)

    def _is_human_turn(self) -> bool:
        human_is_white = self.human_color_input.currentText() == "white"
        human_color = chess.WHITE if human_is_white else chess.BLACK
        return self.gm.board.turn == human_color

    def _is_ai_turn(self) -> bool:
        return not self._is_human_turn()

    def _refresh_ui(self, status_text: str | None = None) -> None:
        self.board_widget.render_position(
            board=self.gm.board,
            selected_square=self.selected_square,
            legal_targets=self.current_targets,
            last_move=self.last_move,
        )
        self.moves_input.setText(self._history_to_text())
        self.fen_label.setText(self.gm.board.fen())
        self.turn_label.setText("white" if self.gm.board.turn == chess.WHITE else "black")

        status = self.gm.game_status()
        if status.state == "ongoing":
            if status_text:
                self._set_status(status_text)
            return

        if status.winner is None:
            self._set_status(f"Game over: {status.state} ({status.result})")
        else:
            self._set_status(f"Game over: {status.state}, winner={status.winner} ({status.result})")

    def _clear_selection(self) -> None:
        self.selected_square = None
        self.current_targets = set()

    def _legal_targets_for(self, square: chess.Square) -> set[chess.Square]:
        targets: set[chess.Square] = set()
        for move in self.gm.board.legal_moves:
            if move.from_square == square:
                targets.add(move.to_square)
        return targets

    def _pick_legal_move(self, from_square: chess.Square, to_square: chess.Square) -> chess.Move | None:
        candidates = [
            move
            for move in self.gm.board.legal_moves
            if move.from_square == from_square and move.to_square == to_square
        ]
        if not candidates:
            return None
        for move in candidates:
            if move.promotion == chess.QUEEN:
                return move
        return candidates[0]

    def _apply_san_move(self, san_move: str, actor: str) -> bool:
        self._sync_chessi_manager()
        try:
            move = self.gm.apply_pan_move(san_move)
        except ValueError as exc:
            self._set_status(f"{actor} move rejected: {san_move} ({exc})")
            return False

        self.move_history.append(san_move)
        self.last_move = move
        self._clear_selection()
        self._refresh_ui(status_text=f"{actor}: {san_move}")
        return True

    def _load_model(self) -> None:
        try:
            self.chessi.load(
                repo=self.repo_input.text().strip(),
                device=self.device_input.currentText(),
                force_download=self.force_download_input.isChecked(),
            )
            self._set_status("Model loaded")
        except Exception as exc:
            self._error(str(exc))

    def _start_new_game(self, play_ai_opening: bool) -> None:
        self.gm.reset()
        self.move_history = []
        self.last_move = None
        self.last_predictions = []
        self.table.setRowCount(0)
        self.output_line.clear()
        self._clear_selection()
        self._sync_chessi_manager()
        self._refresh_ui(status_text="New game started")

        if play_ai_opening and self._is_ai_turn():
            self._play_ai_move()

    def _undo_move(self) -> None:
        if not self.gm.board.move_stack:
            self._set_status("No move to undo")
            return
        self.gm.board.pop()
        if self.move_history:
            self.move_history.pop()
        self.last_move = self.gm.board.peek() if self.gm.board.move_stack else None
        self._clear_selection()
        self._sync_chessi_manager()
        self._refresh_ui(status_text="Undid last move")

    def _load_moves_from_text(self) -> None:
        raw = self.moves_input.text().strip()
        moves = normalize_move_sequence(raw)
        self.gm.reset()
        self.move_history = []
        self.last_move = None
        self._clear_selection()

        try:
            for move in moves:
                applied = self.gm.apply_pan_move(move)
                self.move_history.append(move)
                self.last_move = applied
        except ValueError as exc:
            self._error(str(exc))
            return

        self._sync_chessi_manager()
        self._refresh_ui(status_text="Loaded SAN move list")

    def _suggest_moves(self) -> None:
        self._sync_chessi_manager()
        try:
            predicted_payload = self.chessi.gen(
                moves=self._history_to_text(),
                top_k=int(self.top_k_input.value()),
                temperature=float(self.temperature_input.value()),
                device=self.device_input.currentText(),
                with_prob=True,
                force_download=self.force_download_input.isChecked(),
            )
        except Exception as exc:
            self._error(str(exc))
            return

        self.last_predictions = [
            (str(item["move"]), float(item["prob"]))
            for item in predicted_payload
        ]
        self.output_line.setText(f"[{', '.join(move for move, _ in self.last_predictions)}]")
        self.table.setRowCount(len(self.last_predictions))
        for row, (move, prob) in enumerate(self.last_predictions):
            self.table.setItem(row, 0, QTableWidgetItem(move))
            self.table.setItem(row, 1, QTableWidgetItem(f"{prob:.6f}"))
        self._set_status(f"Suggested {len(self.last_predictions)} move(s)")

    def _play_ai_move(self) -> None:
        if self.gm.game_status().state != "ongoing":
            self._set_status("Game is already finished")
            return
        if not self._is_ai_turn():
            self._set_status("Not AI turn")
            return

        self._sync_chessi_manager()
        try:
            predicted = self.chessi.predict(
                moves=self._history_to_text(),
                top_k=max(10, int(self.top_k_input.value())),
            )
        except Exception as exc:
            self._error(str(exc))
            return

        if not self._apply_san_move(predicted, actor="AI"):
            return

    def _play_prediction_row(self, row: int, _column: int) -> None:
        if row < 0 or row >= len(self.last_predictions):
            return
        move, _ = self.last_predictions[row]
        self._apply_san_move(move, actor="Manual")

        if self.auto_ai_input.isChecked() and self._is_ai_turn():
            self._play_ai_move()

    def _on_board_click(self, row: int, col: int) -> None:
        if self.gm.game_status().state != "ongoing":
            self._set_status("Game is finished")
            return
        if not self._is_human_turn():
            self._set_status("AI turn")
            return

        square = self.board_widget.square_from_cell(row, col)
        board = self.gm.board
        clicked_piece = board.piece_at(square)

        if self.selected_square is None:
            if clicked_piece is None or clicked_piece.color != board.turn:
                self._set_status("Select your piece")
                return
            self.selected_square = square
            self.current_targets = self._legal_targets_for(square)
            self._refresh_ui(status_text=f"Selected {chess.square_name(square)}")
            return

        if square == self.selected_square:
            self._clear_selection()
            self._refresh_ui(status_text="Selection cleared")
            return

        move = self._pick_legal_move(self.selected_square, square)
        if move is None:
            if clicked_piece is not None and clicked_piece.color == board.turn:
                self.selected_square = square
                self.current_targets = self._legal_targets_for(square)
                self._refresh_ui(status_text=f"Selected {chess.square_name(square)}")
            else:
                self._set_status("Illegal target square")
            return

        san_move = board.san(move)
        if not self._apply_san_move(san_move, actor="You"):
            return

        if self.auto_ai_input.isChecked() and self._is_ai_turn():
            self._play_ai_move()

    def _error(self, message: str) -> None:
        QMessageBox.critical(self, "Error", message)
        self._set_status("Error")


def main() -> None:
    app = QApplication(sys.argv)
    window = ChessLMWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
