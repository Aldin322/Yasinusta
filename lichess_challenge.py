"""Utility for challenging a specific Lichess user with SacrificeBot.

The script sends a direct challenge from your bot account to a target player
and then drives the game through the official Lichess Bot API.  It uses the
SacrificeBot searcher from ``sacrifice_bot.py`` to pick moves and biases them
towards sacrificial lines just like the standalone CLI.

Usage example:

    python lichess_challenge.py --opponent aldin07 --clock 5 --increment 3

You must supply your bot's API token via ``--token`` or the ``LICHESS_TOKEN``
environment variable.  The token needs the ``bot:play`` scope.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict, Iterable, Iterator, Optional

import chess
import requests

from sacrifice_bot import SacrificeBot


LICHESS_API = "https://lichess.org"


def _auth_headers(token: str) -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
    }


def _stream_sse(response: requests.Response) -> Iterator[Dict]:
    """Yield JSON payloads from a Server-Sent Events response."""

    buffer = ""
    for raw_line in response.iter_lines():
        if raw_line is None:
            continue

        if isinstance(raw_line, bytes):
            line = raw_line.decode("utf-8", errors="replace").strip()
        else:
            line = raw_line.strip()
        if not line:
            if buffer:
                try:
                    yield json.loads(buffer)
                except json.JSONDecodeError as exc:  # pragma: no cover - defensive
                    raise RuntimeError(f"Malformed SSE payload from Lichess: {buffer}") from exc
                buffer = ""
            continue

        if line.startswith(":"):
            # Comment/keep-alive
            continue

        if line.startswith("data:"):
            payload = line[len("data:") :].strip()
            if payload:
                buffer += payload

    if buffer:
        try:
            yield json.loads(buffer)
        except json.JSONDecodeError as exc:  # pragma: no cover - defensive
            raise RuntimeError(f"Malformed trailing SSE payload from Lichess: {buffer}") from exc


def _challenge_player(
    token: str,
    opponent: str,
    clock: int,
    increment: int,
    rated: bool,
) -> Dict:
    payload = {
        "clock.limit": clock * 60,
        "clock.increment": increment,
        "rated": str(rated).lower(),
    }
    response = requests.post(
        f"{LICHESS_API}/api/challenge/{opponent}",
        headers=_auth_headers(token),
        data=payload,
        timeout=30,
    )
    response.raise_for_status()

    try:
        payload = response.json()
    except ValueError as exc:  # pragma: no cover - defensive
        raise RuntimeError(
            f"Unexpected response from Lichess while challenging {opponent}: {response.text}"
        ) from exc

    # ``/api/challenge/{opponent}`` sometimes returns the challenge under
    # ``{"challenge": {...}}`` (documented API response) and other times it
    # returns the challenge object at the top level (what we observed when the
    # user accepted the challenge but the CLI thought it failed).  Accept both
    # forms so we do not flag a false rejection.
    challenge = payload.get("challenge")
    if not challenge and payload.get("id") and payload.get("status"):
        challenge = payload

    if not challenge:
        error_message = payload.get("error") or payload.get("message")
        raise RuntimeError(
            "Lichess rejected the challenge"
            + (f": {error_message}" if error_message else f". Full response: {payload}")
        )
    return challenge


def _challenge_with_retries(
    token: str,
    opponent: str,
    clock: int,
    increment: int,
    rated: bool,
    retries: int,
    retry_wait: int,
) -> Dict:
    attempts = 0
    while True:
        try:
            return _challenge_player(token, opponent, clock, increment, rated)
        except (RuntimeError, requests.RequestException) as exc:
            attempts += 1
            if attempts > max(0, retries):
                raise
            wait_time = max(1, retry_wait)
            print(f"Challenge attempt {attempts} failed: {exc}. Retrying in {wait_time}s...")
            time.sleep(wait_time)


def _wait_for_game(
    token: str,
    challenge_id: str,
    opponent: str,
    timeout_seconds: int = 120,
) -> Dict:
    """Block until the specific challenge turns into a game and return it."""

    opponent_normalized = opponent.lower()
    start_time = time.time()

    def _timeout_expired() -> bool:
        return timeout_seconds > 0 and time.time() - start_time > timeout_seconds

    with requests.get(
        f"{LICHESS_API}/api/stream/event",
        headers=_auth_headers(token),
        stream=True,
        timeout=60,
    ) as response:
        response.raise_for_status()
        for event in _stream_sse(response):
            event_type = event.get("type")
            if event_type == "gameStart":
                game = event.get("game", {})
                opponent_id = game.get("opponent", {}).get("id", "").lower()
                if game.get("id") == challenge_id or opponent_id == opponent_normalized:
                    return game

            if event_type == "challenge":
                challenge = event.get("challenge", {})
                if challenge.get("id") != challenge_id:
                    continue

                status = challenge.get("status")
                if status in {"accepted", "started"}:
                    color = challenge.get("finalColor") or challenge.get("color") or "random"
                    color = color.lower()
                    if color not in {"white", "black"}:
                        # Color still random – default to white until the board stream tells us otherwise.
                        color = "white"
                    opponent_payload = challenge.get("destUser") or {}
                    if opponent_payload.get("id"):
                        opponent_id = opponent_payload["id"]
                    else:
                        opponent_id = opponent
                    return {
                        "id": challenge_id,
                        "color": color,
                        "opponent": {"id": opponent_id},
                    }

                if status in {"declined", "canceled"}:
                    raise RuntimeError(f"Challenge {status} by {opponent} before the game started")

            if _timeout_expired():
                break
    raise RuntimeError("Failed to detect the started game from the event stream before timing out")


def _apply_moves(board: chess.Board, moves: str) -> None:
    if not moves:
        return
    for move in moves.split():
        try:
            board.push_uci(move)
        except ValueError as exc:
            raise RuntimeError(f"Received illegal move sequence from Lichess (bad move: {move})") from exc


def _stream_board(token: str, game_id: str) -> Iterable[Dict]:
    with requests.get(
        f"{LICHESS_API}/api/bot/game/stream/{game_id}",
        headers=_auth_headers(token),
        stream=True,
    ) as response:
        response.raise_for_status()
        yield from _stream_sse(response)


def _submit_move(token: str, game_id: str, move: chess.Move) -> None:
    response = requests.post(
        f"{LICHESS_API}/api/bot/game/{game_id}/move/{move.uci()}",
        headers=_auth_headers(token),
        timeout=30,
    )
    response.raise_for_status()


def _should_move(board: chess.Board, my_color: chess.Color) -> bool:
    return board.turn == my_color and not board.is_game_over()


def _drive_game(token: str, game_id: str, my_color: chess.Color, bot: SacrificeBot) -> Dict:
    board = chess.Board()
    final_state: Dict = {}

    for event in _stream_board(token, game_id):
        event_type = event.get("type")
        if event_type not in {"gameFull", "gameState"}:
            continue

        state = event["state"] if event_type == "gameFull" else event
        moves = state.get("moves", "")
        board.reset()
        _apply_moves(board, moves)

        if _should_move(board, my_color):
            my_time_key = "wtime" if my_color == chess.WHITE else "btime"
            my_inc_key = "winc" if my_color == chess.WHITE else "binc"
            my_time_raw = state.get(my_time_key)
            my_increment_raw = state.get(my_inc_key, 0)
            my_time = int(my_time_raw) if my_time_raw is not None else None
            my_increment = int(my_increment_raw) if my_increment_raw is not None else 0

            search_result = bot.choose(board, my_time, my_increment)
            if not search_result.move:
                raise RuntimeError("No legal move found for the current position")
            _submit_move(token, game_id, search_result.move)
            time.sleep(0.2)

        status = state.get("status")
        if status and status != "started":
            final_state = state
            break

    return final_state


def _summarize_game_outcome(state: Optional[Dict], my_color: chess.Color) -> str:
    if not state:
        return "Game concluded, but no final status was received from Lichess."

    status = state.get("status", "unknown")
    winner = state.get("winner")
    if winner is None:
        outcome = "draw"
    elif (winner == "white" and my_color == chess.WHITE) or (winner == "black" and my_color == chess.BLACK):
        outcome = "win"
    else:
        outcome = "loss"

    moves_played = len(state.get("moves", "").split())
    return f"Result: {outcome} (status: {status}, moves played: {moves_played})."

def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Challenge a Lichess user with SacrificeBot")
    parser.add_argument("token_positional", nargs="?", help="Bot API token (optional positional)")
    parser.add_argument("--token", help="Bot API token (falls back to LICHESS_TOKEN env var)")
    parser.add_argument("--opponent", required=True, help="Lichess username to challenge")
    parser.add_argument("--clock", type=int, default=5, help="Base time in minutes")
    parser.add_argument("--increment", type=int, default=3, help="Increment in seconds")
    parser.add_argument("--rated", action="store_true", help="Play a rated game instead of casual")
    parser.add_argument("--depth", type=int, default=3, help="Search depth for SacrificeBot")
    parser.add_argument(
        "--sacrifice-margin",
        type=int,
        default=40,
        help="Sacrifice margin (centipawns) forwarded to SacrificeBot",
    )
    parser.add_argument("--games", type=int, default=1, help="Number of consecutive games to play")
    parser.add_argument(
        "--challenge-retries",
        type=int,
        default=3,
        help="How many times to retry sending a challenge before giving up",
    )
    parser.add_argument(
        "--retry-wait",
        type=int,
        default=10,
        help="Seconds to wait between challenge retries",
    )
    parser.add_argument(
        "--pause-between-games",
        type=int,
        default=5,
        help="Seconds to wait before starting the next game",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = parse_args(argv)
    token = (args.token or args.token_positional or os.getenv("LICHESS_TOKEN", "")).strip()
    if not token:
        raise SystemExit("Please supply --token or set the LICHESS_TOKEN environment variable")

    bot = SacrificeBot(depth=args.depth, sacrifice_margin=args.sacrifice_margin)
    total_games = max(1, args.games)

    for game_index in range(total_games):
        print(f"=== Match {game_index + 1}/{total_games} ===")
        print(f"Challenging {args.opponent} for a {args.clock}+{args.increment} game...")
        try:
            challenge = _challenge_with_retries(
                token,
                args.opponent,
                args.clock,
                args.increment,
                args.rated,
                retries=args.challenge_retries,
                retry_wait=args.retry_wait,
            )
        except RuntimeError as exc:
            raise SystemExit(str(exc)) from exc
        except requests.RequestException as exc:  # pragma: no cover - network errors
            raise SystemExit(f"Failed to send challenge: {exc}") from exc
        challenge_id = challenge["id"]
        print(f"Challenge created (id: {challenge_id}). Waiting for the game to start...")
        game = _wait_for_game(token, challenge_id, args.opponent)
        my_color = chess.WHITE if game.get("color") == "white" else chess.BLACK
        print(f"Game {game['id']} started. Playing as {'White' if my_color == chess.WHITE else 'Black'}.")
        final_state = _drive_game(token, game["id"], my_color, bot)
        print(_summarize_game_outcome(final_state, my_color))
        print("Game finished.")

        if game_index < total_games - 1:
            pause = max(0, args.pause_between_games)
            if pause:
                print(f"Waiting {pause}s before issuing the next challenge...")
                time.sleep(pause)


if __name__ == "__main__":
    main()
