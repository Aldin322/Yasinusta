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
}

KING_MIDGAME_TABLE: List[int] = [
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -20,-30,-30,-40,-40,-30,-30,-20,
    -10,-20,-20,-20,-20,-20,-20,-10,
     20, 20,  0,  0,  0,  0, 20, 20,
     20, 30, 10,  0,  0, 10, 30, 20,
]

KING_ENDGAME_TABLE: List[int] = [
    -50,-40,-30,-20,-20,-30,-40,-50,
    -30,-20,-10,  0,  0,-10,-20,-30,
    -30,-10, 20, 30, 30, 20,-10,-30,
    -30,-10, 30, 40, 40, 30,-10,-30,
    -30,-10, 30, 40, 40, 30,-10,-30,
    -30,-10, 20, 30, 30, 20,-10,-30,
    -30,-30,  0,  0,  0,  0,-30,-30,
    -50,-30,-30,-30,-30,-30,-30,-50,
]

BISHOP_PAIR_BONUS = 40
ROOK_OPEN_FILE_BONUS = 20
ROOK_SEMI_OPEN_FILE_BONUS = 8
PASSED_PAWN_BONUS = [0, 10, 20, 35, 60, 90, 140, 0]
MOBILITY_WEIGHT = 2
PAWN_SHIELD_BONUS = 8
KING_ATTACK_PENALTY = 15


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

    def __init__(self, depth: int = 4, sacrifice_margin: int = 40, quiescence_depth: int = 6) -> None:
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
        self._last_completed_depth = depth

    # ------------------------------------------------------------------
    # Evaluation helpers
    def _material_score(self, board: chess.Board, color: chess.Color) -> int:
        total = 0
        for piece_type, value in PIECE_VALUES.items():
            total += len(board.pieces(piece_type, color)) * value
        return total

    def _piece_square_score(
        self,
        board: chess.Board,
        color: chess.Color,
        endgame_phase: float,
    ) -> float:
        total = 0.0
        for piece_type, table in PIECE_SQUARE_TABLES.items():
            for square in board.pieces(piece_type, color):
                index = square if color == chess.WHITE else chess.square_mirror(square)
                total += table[index]
        total += self._king_table_score(board, color, endgame_phase)
        return total

    def _king_table_score(self, board: chess.Board, color: chess.Color, endgame_phase: float) -> float:
        king_square = board.king(color)
        if king_square is None:
            return 0.0
        index = king_square if color == chess.WHITE else chess.square_mirror(king_square)
        midgame = KING_MIDGAME_TABLE[index]
        endgame = KING_ENDGAME_TABLE[index]
        return midgame * (1.0 - endgame_phase) + endgame * endgame_phase

    def _game_phase(self, board: chess.Board) -> float:
        weights = {
            chess.KNIGHT: 1,
            chess.BISHOP: 1,
            chess.ROOK: 2,
            chess.QUEEN: 4,
        }
        total_phase = 24
        phase = 0
        for piece_type, weight in weights.items():
            phase += weight * (
                len(board.pieces(piece_type, chess.WHITE))
                + len(board.pieces(piece_type, chess.BLACK))
            )
        if total_phase == 0:
            return 1.0
        phase = max(0, min(total_phase, phase))
        return 1.0 - (phase / total_phase)

    def _bishop_pair_bonus(self, board: chess.Board, color: chess.Color) -> float:
        return BISHOP_PAIR_BONUS if len(board.pieces(chess.BISHOP, color)) >= 2 else 0.0

    def _rook_file_bonus(self, board: chess.Board, color: chess.Color) -> float:
        bonus = 0.0
        own_pawns = board.pieces(chess.PAWN, color)
        enemy_pawns = board.pieces(chess.PAWN, not color)
        for square in board.pieces(chess.ROOK, color):
            file_mask = chess.BB_FILES[chess.square_file(square)]
            if not (own_pawns | enemy_pawns) & file_mask:
                bonus += ROOK_OPEN_FILE_BONUS
            elif not (own_pawns & file_mask):
                bonus += ROOK_SEMI_OPEN_FILE_BONUS
        return bonus

    def _passed_pawn_bonus(self, board: chess.Board, color: chess.Color, endgame_phase: float) -> float:
        bonus = 0.0
        enemy_pawns = board.pieces(chess.PAWN, not color)
        direction = 1 if color == chess.WHITE else -1
        for square in board.pieces(chess.PAWN, color):
            file_index = chess.square_file(square)
            rank_index = chess.square_rank(square)
            passed = True
            check_rank = rank_index + direction
            while 0 <= check_rank < 8 and passed:
                for file_offset in (-1, 0, 1):
                    file_candidate = file_index + file_offset
                    if not 0 <= file_candidate < 8:
                        continue
                    target = chess.square(file_candidate, check_rank)
                    if target in enemy_pawns:
                        passed = False
                        break
                check_rank += direction
            if not passed:
                continue
            advancement = rank_index if color == chess.WHITE else 7 - rank_index
            advancement = max(0, min(7, advancement))
            bonus += PASSED_PAWN_BONUS[advancement] * (0.5 + endgame_phase)
        return bonus

    def _mobility_term(self, board: chess.Board, color: chess.Color) -> float:
        temp = board.copy(stack=False)
        temp.turn = color
        move_count = sum(1 for _ in temp.legal_moves)
        return move_count * MOBILITY_WEIGHT

    def _king_safety_score(self, board: chess.Board, color: chess.Color, endgame_phase: float) -> float:
        king_square = board.king(color)
        if king_square is None:
            return 0.0
        rank = chess.square_rank(king_square)
        file_index = chess.square_file(king_square)
        direction = 1 if color == chess.WHITE else -1
        shield_rank = rank + direction
        shield_bonus = 0.0
        if 0 <= shield_rank < 8:
            for file_offset in (-1, 0, 1):
                file_candidate = file_index + file_offset
                if not 0 <= file_candidate < 8:
                    continue
                target = chess.square(file_candidate, shield_rank)
                piece = board.piece_at(target)
                if piece and piece.piece_type == chess.PAWN and piece.color == color:
                    shield_bonus += PAWN_SHIELD_BONUS
        attackers = len(board.attackers(not color, king_square))
        attack_penalty = attackers * KING_ATTACK_PENALTY
        weight = 1.0 - endgame_phase
        return (shield_bonus - attack_penalty) * weight

    def evaluate_white(self, board: chess.Board) -> float:
        """Return a centipawn evaluation from White's perspective."""

        endgame_phase = self._game_phase(board)
        score = 0.0
        for color in (chess.WHITE, chess.BLACK):
            sign = 1 if color == chess.WHITE else -1
            material = self._material_score(board, color)
            positional = self._piece_square_score(board, color, endgame_phase)
            extras = (
                self._bishop_pair_bonus(board, color)
                + self._rook_file_bonus(board, color)
                + self._passed_pawn_bonus(board, color, endgame_phase)
                + self._mobility_term(board, color)
                + self._king_safety_score(board, color, endgame_phase)
            )
            score += sign * (material + positional + extras)
        return score

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

    def _normalize_hash_value(self, value: object) -> Optional[int]:
        """Convert ``value`` to an integer hash if possible."""

        if value is None:
            return None
        if isinstance(value, (int, bool)):
            return int(value)
        if isinstance(value, (bytes, bytearray)):
            if not value:
                return None
            return int.from_bytes(value, byteorder="big", signed=False)
        if isinstance(value, str):
            # ``Board.transposition_key`` returned a hex string in a few
            # historical python-chess revisions.  Accept that form too.
            value = value.strip()
            if not value:
                return None
            try:
                return int(value, 0)
            except ValueError:
                return None
        if isinstance(value, tuple):
            # Some versions expose ``(hash, dirty_flag)`` tuples; others may
            # return nested tuples.  Walk them recursively until a usable int
            # appears.
            for element in value:
                normalized = self._normalize_hash_value(element)
                if normalized is not None:
                    return normalized
            return None
        return None

    def _board_hash(self, board: chess.Board) -> int:
        """Return a transposition key compatible across python-chess versions."""

        preferred_attrs = ("transposition_key", "zobrist_hash", "_transposition_key")
        for attr_name in preferred_attrs:
            attr = getattr(board, attr_name, None)
            if attr is None:
                continue

            value = attr() if callable(attr) else attr
            normalized = self._normalize_hash_value(value)
            if normalized is not None:
                return normalized

        # As a last resort build a deterministic hash from the board state.
        fallback = hash(
            (
                board.board_fen(),
                board.turn,
                board.castling_rights,
                board.ep_square,
                board.halfmove_clock,
                board.fullmove_number,
            )
        )
        return fallback

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
        self._last_completed_depth = 0
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
        if use_time_management:
            start_depth = 1
            max_depth = min(self.depth + 6, 32)
        else:
            start_depth = self.depth
            max_depth = self.depth

        depth = start_depth
        while depth <= max_depth:
            try:
                result = self._search_root(board, depth)
            except SearchTimeout:
                break
            best_result = result
            self._last_completed_depth = depth
            self._update_principal_variation(board)
            if not use_time_management:
                break
            depth += 1

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

        tt_entry = self._tt.get(self._board_hash(board))
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

        key = self._board_hash(board)
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
        limit = max(1, self._last_completed_depth) * 2
        for _ in range(limit):
            entry = self._tt.get(self._board_hash(probe))
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
    parser.add_argument("--depth", type=int, default=4, help="Search depth in plies")
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
