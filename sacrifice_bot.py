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
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import DefaultDict, Dict, List, Optional, Tuple

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


class SearchTimeout(Exception):
    """Raised when the search must stop because the time budget expired."""


@dataclass
class SearchResult:
    move: Optional[chess.Move]
    score: float
    sacrifices: float


@dataclass
class TTEntry:
    depth: int
    score: float
    sacrifices: float
    node_type: str  # "exact", "lower", "upper"
    move: Optional[chess.Move]


class SacrificeBot:
    """A chess searcher that prefers sacrificing material when advantageous."""

    def __init__(self, depth: int = 3, sacrifice_margin: int = 40, quiescence_depth: int = 6) -> None:
        """Create a new bot.

        Args:
            depth: The maximum depth to search in plies.
            sacrifice_margin: Two scores within ``sacrifice_margin`` centipawns
                of each other are treated as roughly equal.  In that case the bot
                will prefer the variation that gives up more of its own material.
            quiescence_depth: Maximum plies to extend capture-only search once
                the fixed depth is exhausted.  This keeps tactical lines stable
                without letting the search explode indefinitely.
        """

        if depth < 1:
            raise ValueError("depth must be >= 1")
        if quiescence_depth < 0:
            raise ValueError("quiescence_depth must be >= 0")

        self.depth = depth
        self.sacrifice_margin = sacrifice_margin
        self.quiescence_depth = quiescence_depth
        self.root_color = chess.WHITE
        self.initial_material = 0
        self._deadline: Optional[float] = None
        self._tt: Dict[int, TTEntry] = {}
        self._killer_moves: DefaultDict[int, List[chess.Move]] = defaultdict(list)
        self._history_scores: DefaultDict[Tuple[int, int], int] = defaultdict(int)
        self._principal_variation: List[chess.Move] = []

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

    def _sacrifice_score(self, board: chess.Board) -> float:
        return max(0, self.initial_material - self._material_score(board, self.root_color))

    # ------------------------------------------------------------------
    def _check_time(self) -> None:
        if self._deadline is None:
            return
        if time.perf_counter() >= self._deadline:
            raise SearchTimeout

    def _time_budget(self, time_remaining_ms: int, increment_ms: int) -> int:
        """Return how many milliseconds to spend on the current move."""

        if time_remaining_ms <= 0:
            return 0

        # Keep a comfortable safety margin (at least 200 ms or 10% of the clock).
        safety_margin = max(200, int(time_remaining_ms * 0.1))
        usable = max(0, time_remaining_ms - safety_margin)
        # Spend up to 5% of the remaining time plus half of the increment.
        allocation = int(time_remaining_ms * 0.05) + increment_ms // 2
        allocation = max(50, allocation)
        return min(usable, allocation)

    # ------------------------------------------------------------------
    def choose(
        self,
        board: chess.Board,
        time_remaining_ms: Optional[int] = None,
        increment_ms: int = 0,
    ) -> SearchResult:
        """Search for the best move from the current position.

        Args:
            board: Current position.
            time_remaining_ms: Remaining clock time in milliseconds for the
                side to move.  When provided the bot performs iterative
                deepening and aborts the search if the allotted budget runs out.
            increment_ms: Clock increment in milliseconds, used to slightly
                extend the search budget when plenty of increment is available.
        """

        self.root_color = board.turn
        self.initial_material = self._material_score(board, self.root_color)

        self._deadline = None
        self._tt.clear()
        self._killer_moves.clear()
        self._history_scores.clear()
        self._principal_variation.clear()
        use_time_management = False
        if time_remaining_ms is not None:
            budget = self._time_budget(time_remaining_ms, increment_ms)
            if budget > 0:
                self._deadline = time.perf_counter() + budget / 1000
                use_time_management = True
            else:
                move = next(iter(board.legal_moves), None)
                sacrifices = self._sacrifice_score(board)
                return SearchResult(move, self.evaluate(board), sacrifices)

        best_result = SearchResult(None, -math.inf, -math.inf)
        start_depth = 1 if use_time_management else self.depth
        max_depth = self.depth

        depth = start_depth
        while depth <= max_depth:
            try:
                result = self._search_root(board, depth)
            except SearchTimeout:
                break
            best_result = result
            self._update_principal_variation(board)
            depth += 1 if use_time_management else max_depth  # exit loop when fixed depth

        if best_result.move is None:
            # Fall back to the first legal move if we never finished a search.
            move = next(iter(board.legal_moves), None)
            sacrifices = self._sacrifice_score(board)
            return SearchResult(move, self.evaluate(board), sacrifices)

        return best_result

    def _order_moves(
        self,
        board: chess.Board,
        ply: int,
        tt_move: Optional[chess.Move] = None,
        pv_move: Optional[chess.Move] = None,
    ) -> List[chess.Move]:
        killer_moves = self._killer_moves.get(ply, [])

        def move_score(move: chess.Move) -> Tuple[int, float]:
            score = 0
            if move == pv_move:
                score += 10000
            if move == tt_move:
                score += 9000
            if move in killer_moves:
                score += 4000
            if board.is_capture(move):
                victim = board.piece_type_at(move.to_square)
                if victim is None and board.is_en_passant(move):
                    victim = chess.PAWN
                attacker = board.piece_type_at(move.from_square)
                victim_value = PIECE_VALUES.get(victim, 0)
                attacker_value = PIECE_VALUES.get(attacker, 1)
                score += 3000 + victim_value - attacker_value // 10
            if board.gives_check(move):
                score += 200
            history_key = (move.from_square, move.to_square)
            score += self._history_scores.get(history_key, 0)
            return (score, move.to_square)

        return sorted(board.legal_moves, key=move_score, reverse=True)

    def _search_root(self, board: chess.Board, depth: int) -> SearchResult:
        best_move = None
        best_score = -math.inf
        best_sacrifices = -math.inf

        alpha = -math.inf
        beta = math.inf

        tt_entry = self._tt.get(board.zobrist_hash())
        pv_move = self._principal_variation[0] if self._principal_variation else None
        for move in self._order_moves(
            board,
            ply=0,
            tt_move=tt_entry.move if tt_entry else None,
            pv_move=pv_move,
        ):
            self._check_time()
            board.push(move)
            score, sacrifices = self._search(board, depth - 1, alpha, beta, ply=1, quiescence_level=0)
            board.pop()

            if self._is_better(score, sacrifices, best_score, best_sacrifices, maximizing=True):
                best_move = move
                best_score = score
                best_sacrifices = sacrifices

            alpha = max(alpha, best_score)

        return SearchResult(best_move, best_score, best_sacrifices)

    def _search(
        self,
        board: chess.Board,
        depth: int,
        alpha: float,
        beta: float,
        ply: int,
        quiescence_level: int,
    ) -> Tuple[float, float]:
        self._check_time()

        if board.is_game_over():
            if board.is_checkmate():
                score = -math.inf if board.turn == self.root_color else math.inf
            else:
                score = self.evaluate(board)
            sacrifices = self._sacrifice_score(board)
            return score, sacrifices

        if depth == 0:
            return self._quiescence(board, alpha, beta, quiescence_level)

        key = board.zobrist_hash()
        entry = self._tt.get(key)
        tt_move = entry.move if entry else None
        alpha_orig = alpha
        beta_orig = beta
        if entry and entry.depth >= depth:
            if entry.node_type == "exact":
                return entry.score, entry.sacrifices
            if entry.node_type == "lower" and entry.score >= beta:
                return entry.score, entry.sacrifices
            if entry.node_type == "upper" and entry.score <= alpha:
                return entry.score, entry.sacrifices

        maximizing = board.turn == self.root_color
        best_score = -math.inf if maximizing else math.inf
        best_sacrifices = -math.inf if maximizing else math.inf

        pv_move = self._principal_variation[ply] if ply < len(self._principal_variation) else None
        best_move = tt_move if tt_move in board.legal_moves else None
        for move in self._order_moves(board, ply, tt_move=tt_move, pv_move=pv_move):
            board.push(move)
            child_score, child_sacrifices = self._search(
                board,
                depth - 1,
                alpha,
                beta,
                ply + 1,
                quiescence_level,
            )
            board.pop()

            if self._is_better(child_score, child_sacrifices, best_score, best_sacrifices, maximizing):
                best_score = child_score
                best_sacrifices = child_sacrifices
                best_move = move

            if maximizing:
                alpha = max(alpha, best_score)
            else:
                beta = min(beta, best_score)

            if beta <= alpha:
                if not board.is_capture(move):
                    killers = self._killer_moves[ply]
                    if move in killers:
                        killers.remove(move)
                    killers.insert(0, move)
                    del killers[2:]
                    history_key = (move.from_square, move.to_square)
                    self._history_scores[history_key] += depth * depth
                break

        node_type = "exact"
        if best_score <= alpha_orig:
            node_type = "upper"
        elif best_score >= beta_orig:
            node_type = "lower"

        self._tt[key] = TTEntry(
            depth=depth,
            score=best_score,
            sacrifices=best_sacrifices,
            node_type=node_type,
            move=best_move,
        )

        return best_score, best_sacrifices

    def _quiescence(
        self,
        board: chess.Board,
        alpha: float,
        beta: float,
        quiescence_level: int,
    ) -> Tuple[float, float]:
        self._check_time()

        if board.is_game_over():
            if board.is_checkmate():
                score = -math.inf if board.turn == self.root_color else math.inf
            else:
                score = self.evaluate(board)
            return score, self._sacrifice_score(board)

        stand_pat = self.evaluate(board)
        stand_sacrifices = self._sacrifice_score(board)

        maximizing = board.turn == self.root_color
        if maximizing:
            if stand_pat >= beta:
                return stand_pat, stand_sacrifices
            if stand_pat > alpha:
                alpha = stand_pat
        else:
            if stand_pat <= alpha:
                return stand_pat, stand_sacrifices
            if stand_pat < beta:
                beta = stand_pat

        if quiescence_level >= self.quiescence_depth:
            return stand_pat, stand_sacrifices

        capture_like_moves = [
            move
            for move in board.legal_moves
            if board.is_capture(move) or board.gives_check(move)
        ]
        if not capture_like_moves:
            return stand_pat, stand_sacrifices

        best_score = stand_pat
        best_sacrifices = stand_sacrifices
        for move in capture_like_moves:
            board.push(move)
            child_score, child_sacrifices = self._quiescence(board, alpha, beta, quiescence_level + 1)
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

    def _update_principal_variation(self, board: chess.Board) -> None:
        self._principal_variation.clear()
        probe = board.copy()
        for _ in range(self.depth * 2):
            entry = self._tt.get(probe.zobrist_hash())
            if not entry or entry.move is None:
                break
            move = entry.move
            if move not in probe.legal_moves:
                break
            self._principal_variation.append(move)
            probe.push(move)

    def principal_variation(self) -> List[chess.Move]:
        return list(self._principal_variation)


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
    pv = [move.uci() for move in bot.principal_variation()]
    if pv:
        print("Principal variation:", " ".join(pv))


if __name__ == "__main__":
    main()
