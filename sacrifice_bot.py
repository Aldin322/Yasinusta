"""Sacrifice-friendly chess engine for Lichess bots.

The script exposes a ``SacrificeBot`` class together with a CLI entry point
that can be used to pick a move for the current position.  The core idea is to
run a conventional alpha-beta search with a simple material/position evaluation
function while biasing the move selection towards lines where we give up more
material ("sacrifices") as long as the resulting evaluation is still good.
"""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import chess

PIECE_VALUES: Dict[chess.PieceType, int] = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,  # King value does not matter for evaluation – checkmate is handled separately.
}

# Basic piece-square tables (in centipawns) that give small positional hints.
# Values are defined from White's perspective.  For Black we simply mirror them.
PIECE_SQUARE_TABLES: Dict[chess.PieceType, List[int]] = {
    chess.PAWN: [
         0,  0,  0,  0,  0,  0,  0,  0,
         5, 10, 10,-20,-20, 10, 10,  5,
         5, -5,-10,  0,  0,-10, -5,  5,
         0,  0,  0, 20, 20,  0,  0,  0,
         5,  5, 10, 25, 25, 10,  5,  5,
        10, 10, 20, 30, 30, 20, 10, 10,
        50, 50, 50, 50, 50, 50, 50, 50,
         0,  0,  0,  0,  0,  0,  0,  0,
    ],
    chess.KNIGHT: [
        -50,-40,-30,-30,-30,-30,-40,-50,
        -40,-20,  0,  0,  0,  0,-20,-40,
        -30,  0, 10, 15, 15, 10,  0,-30,
        -30,  5, 15, 20, 20, 15,  5,-30,
        -30,  0, 15, 20, 20, 15,  0,-30,
        -30,  5, 10, 15, 15, 10,  5,-30,
        -40,-20,  0,  5,  5,  0,-20,-40,
        -50,-40,-30,-30,-30,-30,-40,-50,
    ],
    chess.BISHOP: [
        -20,-10,-10,-10,-10,-10,-10,-20,
        -10,  5,  0,  0,  0,  0,  5,-10,
        -10, 10, 10, 10, 10, 10, 10,-10,
        -10,  0, 10, 10, 10, 10,  0,-10,
        -10,  5,  5, 10, 10,  5,  5,-10,
        -10,  0,  5, 10, 10,  5,  0,-10,
        -10,  0,  0,  0,  0,  0,  0,-10,
        -20,-10,-10,-10,-10,-10,-10,-20,
    ],
    chess.ROOK: [
         0,  0,  0,  5,  5,  0,  0,  0,
        -5,  0,  0,  0,  0,  0,  0, -5,
        -5,  0,  0,  0,  0,  0,  0, -5,
        -5,  0,  0,  0,  0,  0,  0, -5,
        -5,  0,  0,  0,  0,  0,  0, -5,
        -5,  0,  0,  0,  0,  0,  0, -5,
         5, 10, 10, 10, 10, 10, 10,  5,
         0,  0,  0,  0,  0,  0,  0,  0,
    ],
    chess.QUEEN: [
        -20,-10,-10, -5, -5,-10,-10,-20,
        -10,  0,  0,  0,  0,  0,  0,-10,
        -10,  0,  5,  5,  5,  5,  0,-10,
         -5,  0,  5,  5,  5,  5,  0, -5,
          0,  0,  5,  5,  5,  5,  0, -5,
        -10,  5,  5,  5,  5,  5,  0,-10,
        -10,  0,  5,  0,  0,  0,  0,-10,
        -20,-10,-10, -5, -5,-10,-10,-20,
    ],
    chess.KING: [
        -30,-40,-40,-50,-50,-40,-40,-30,
        -30,-40,-40,-50,-50,-40,-40,-30,
        -30,-40,-40,-50,-50,-40,-40,-30,
        -30,-40,-40,-50,-50,-40,-40,-30,
        -20,-30,-30,-40,-40,-30,-30,-20,
        -10,-20,-20,-20,-20,-20,-20,-10,
         20, 20,  0,  0,  0,  0, 20, 20,
         20, 30, 10,  0,  0, 10, 30, 20,
    ],
}


@dataclass
class SearchResult:
    move: Optional[chess.Move]
    score: float
    sacrifices: float


class SacrificeBot:
    """A chess searcher that prefers sacrificing material when advantageous."""

    def __init__(self, depth: int = 3, sacrifice_margin: int = 40) -> None:
        """Create a new bot.

        Args:
            depth: The maximum depth to search in plies.
            sacrifice_margin: Two scores within ``sacrifice_margin`` centipawns
                of each other are treated as roughly equal.  In that case the bot
                will prefer the variation that gives up more of its own material.
        """

        if depth < 1:
            raise ValueError("depth must be >= 1")

        self.depth = depth
        self.sacrifice_margin = sacrifice_margin
        self.root_color = chess.WHITE
        self.initial_material = 0

    # ------------------------------------------------------------------
    # Evaluation helpers
    def _material_score(self, board: chess.Board, color: chess.Color) -> int:
        total = 0
        for piece_type, value in PIECE_VALUES.items():
            total += len(board.pieces(piece_type, color)) * value
        return total

    def _piece_square_score(self, board: chess.Board, color: chess.Color) -> int:
        total = 0
        for piece_type, table in PIECE_SQUARE_TABLES.items():
            for square in board.pieces(piece_type, color):
                index = square if color == chess.WHITE else chess.square_mirror(square)
                total += table[index]
        return total

    def evaluate_white(self, board: chess.Board) -> float:
        """Return a centipawn evaluation from White's perspective."""

        material = self._material_score(board, chess.WHITE) - self._material_score(board, chess.BLACK)
        positional = self._piece_square_score(board, chess.WHITE) - self._piece_square_score(board, chess.BLACK)
        return material + positional

    def evaluate(self, board: chess.Board) -> float:
        score = self.evaluate_white(board)
        return score if self.root_color == chess.WHITE else -score

    # ------------------------------------------------------------------
    def choose(self, board: chess.Board) -> SearchResult:
        """Search for the best move from the current position."""

        self.root_color = board.turn
        self.initial_material = self._material_score(board, self.root_color)

        best_move = None
        best_score = -math.inf
        best_sacrifices = -math.inf

        alpha = -math.inf
        beta = math.inf

        for move in self._order_moves(board):
            board.push(move)
            score, sacrifices = self._search(board, self.depth - 1, alpha, beta)
            board.pop()

            if self._is_better(score, sacrifices, best_score, best_sacrifices, maximizing=True):
                best_move = move
                best_score = score
                best_sacrifices = sacrifices

            alpha = max(alpha, best_score)

        return SearchResult(best_move, best_score, best_sacrifices)

    def _order_moves(self, board: chess.Board) -> List[chess.Move]:
        def move_score(move: chess.Move) -> Tuple[int, int]:
            # Captures and checking moves are usually more critical – score them higher.
            return (int(board.is_capture(move)), int(board.gives_check(move)))

        return sorted(board.legal_moves, key=move_score, reverse=True)

    def _search(self, board: chess.Board, depth: int, alpha: float, beta: float) -> Tuple[float, float]:
        if depth == 0 or board.is_game_over():
            if board.is_checkmate():
                score = -math.inf if board.turn == self.root_color else math.inf
            else:
                score = self.evaluate(board)
            sacrifices = max(0, self.initial_material - self._material_score(board, self.root_color))
            return score, sacrifices

        maximizing = board.turn == self.root_color
        best_score = -math.inf if maximizing else math.inf
        best_sacrifices = -math.inf if maximizing else math.inf

        for move in board.legal_moves:
            board.push(move)
            child_score, child_sacrifices = self._search(board, depth - 1, alpha, beta)
            board.pop()

            if self._is_better(child_score, child_sacrifices, best_score, best_sacrifices, maximizing):
                best_score = child_score
                best_sacrifices = child_sacrifices

            if maximizing:
                alpha = max(alpha, best_score)
            else:
                beta = min(beta, best_score)

            if beta <= alpha:
                break

        return best_score, best_sacrifices

    def _is_better(
        self,
        candidate_score: float,
        candidate_sacrifices: float,
        current_score: float,
        current_sacrifices: float,
        maximizing: bool,
    ) -> bool:
        margin = self.sacrifice_margin
        if maximizing:
            if candidate_score > current_score + margin:
                return True
            if abs(candidate_score - current_score) <= margin and candidate_sacrifices > current_sacrifices:
                return True
        else:
            if candidate_score < current_score - margin:
                return True
            if abs(candidate_score - current_score) <= margin and candidate_sacrifices < current_sacrifices:
                return True
        return False


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sacrifice-preferring chess searcher")
    parser.add_argument("fen", nargs="?", default=chess.STARTING_FEN, help="FEN string for the position")
    parser.add_argument("--depth", type=int, default=3, help="Search depth in plies")
    parser.add_argument(
        "--moves",
        nargs="*",
        help="Optional UCI moves already played from the starting position."
             " When provided, the bot will apply them before searching.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    board = chess.Board(args.fen)

    if args.moves:
        for move in args.moves:
            board.push_uci(move)

    bot = SacrificeBot(depth=args.depth)
    result = bot.choose(board)

    if result.move is None:
        print("No legal moves available.")
        return

    print(f"Best move: {result.move.uci()}")
    print(f"Score: {result.score:.1f} centipawns")
    print(f"Sacrifice score: {result.sacrifices} centipawns of our material given up")


if __name__ == "__main__":
    main()
