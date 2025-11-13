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
import threading
import time
from typing import Dict, Iterable, Iterator, Optional, Set, Tuple

import chess
import requests

from sacrifice_bot import SacrificeBot


LICHESS_API = "https://lichess.org"
FALLBACK_MOVE_TIME_MS = 1500


def _auth_headers(token: str) -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
    }


def _stream_json_events(response: requests.Response) -> Iterator[Dict]:
    """Yield JSON payloads from either SSE or NDJSON streams."""

    buffer = ""
    sse_mode: Optional[bool] = None

    for raw_line in response.iter_lines():
        if raw_line is None:
            continue

        if isinstance(raw_line, bytes):
            line = raw_line.decode("utf-8", errors="replace").strip()
        else:
            line = raw_line.strip()

        if not line:
            if sse_mode:
                if buffer:
                    try:
                        yield json.loads(buffer)
                    except json.JSONDecodeError as exc:  # pragma: no cover - defensive
                        raise RuntimeError(
                            f"Malformed SSE payload from Lichess: {buffer}"
                        ) from exc
                    buffer = ""
            continue

        if line.startswith(":"):
            # Comment/keep-alive (SSE only)
            continue

        if line.startswith("data:"):
            sse_mode = True
            payload = line[len("data:") :].strip()
            if payload:
                buffer += payload
            continue

        if sse_mode is None:
            # No delimiter hints yet; detect NDJSON vs SSE by inspecting the line.
            if line.startswith("{") or line.startswith("["):
                sse_mode = False
            else:
                sse_mode = True

        if sse_mode:
            buffer += line
            continue

        try:
            yield json.loads(line)
        except json.JSONDecodeError as exc:  # pragma: no cover - defensive
            raise RuntimeError(f"Malformed NDJSON payload from Lichess: {line}") from exc

    if sse_mode and buffer:
        try:
            yield json.loads(buffer)
        except json.JSONDecodeError as exc:  # pragma: no cover - defensive
            raise RuntimeError(f"Malformed trailing SSE payload from Lichess: {buffer}") from exc


def _fetch_account_id(token: str) -> str:
    """Return the bot account's user ID for reliably inferring our color."""

    response = requests.get(
        f"{LICHESS_API}/api/account",
        headers=_auth_headers(token),
        timeout=15,
    )
    response.raise_for_status()
    try:
        payload = response.json()
    except ValueError as exc:  # pragma: no cover - defensive
        raise RuntimeError(
            "Unexpected response from Lichess while fetching the bot profile"
        ) from exc

    account_id = (payload.get("id") or "").strip()
    if not account_id:
        raise RuntimeError("Unable to determine the bot account ID from /api/account")
    return account_id


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

    challenge = _extract_challenge(payload)

    if not challenge:
        error_message = payload.get("error") or payload.get("message")
        raise RuntimeError(
            "Lichess rejected the challenge"
            + (f": {error_message}" if error_message else f". Full response: {payload}")
        )
    return challenge


def _accept_challenge(token: str, challenge_id: str) -> None:
    response = requests.post(
        f"{LICHESS_API}/api/challenge/{challenge_id}/accept",
        headers=_auth_headers(token),
        timeout=15,
    )
    response.raise_for_status()


def _extract_challenge(payload: Dict) -> Optional[Dict]:
    """Return the challenge dict regardless of how the API structured it."""

    if not isinstance(payload, dict):
        return None

    embedded = payload.get("challenge")
    if isinstance(embedded, dict):
        return embedded

    if payload.get("id") and payload.get("status"):
        return payload

    return None


class ChallengeNotFoundError(RuntimeError):
    """Raised when Lichess reports a challenge no longer exists (HTTP 404)."""


def _fetch_challenge(token: str, challenge_id: str) -> Dict:
    response = requests.get(
        f"{LICHESS_API}/api/challenge/{challenge_id}",
        headers=_auth_headers(token),
        timeout=15,
    )
    if response.status_code == 404:
        raise ChallengeNotFoundError(f"Challenge {challenge_id} is no longer available")
    response.raise_for_status()
    try:
        payload = response.json()
    except ValueError as exc:  # pragma: no cover - defensive
        raise RuntimeError(
            f"Unexpected response from Lichess while checking challenge {challenge_id}: {response.text}"
        ) from exc

    challenge = _extract_challenge(payload)
    if not challenge:
        raise RuntimeError(f"Unable to parse challenge payload for {challenge_id}: {payload}")
    return challenge


def _locate_active_game(token: str, opponent: str) -> Optional[Dict]:
    """Inspect the account's active games to recover the game created from a challenge."""

    response = requests.get(
        f"{LICHESS_API}/api/account/playing",
        headers=_auth_headers(token),
        params={"nb": 10},
        timeout=15,
    )
    response.raise_for_status()
    try:
        payload = response.json()
    except ValueError as exc:  # pragma: no cover - defensive
        raise RuntimeError(
            f"Unexpected response from Lichess while listing active games: {response.text}"
        ) from exc

    opponent_lower = (opponent or "").lower()
    for entry in payload.get("nowPlaying", []):
        opp_payload = entry.get("opponent") or {}
        opponent_id = (opp_payload.get("id") or "").lower()
        if opponent_lower and opponent_id != opponent_lower:
            continue

        color_text = (entry.get("color") or "").lower()
        if color_text == "white":
            color: Optional[str] = "white"
        elif color_text == "black":
            color = "black"
        else:
            color = None

        return {
            "id": entry.get("gameId"),
            "color": color,
            "opponent": {"id": opp_payload.get("id")} if opp_payload.get("id") else {},
            "me": {"id": entry.get("playerId")} if entry.get("playerId") else {},
        }

    return None


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
            wait_time = _retry_delay_for_exception(exc, retry_wait, attempts)
            rate_limit_note = ""
            if _is_rate_limit_error(exc):
                rate_limit_note = " (Lichess rate-limited the request)"
            print(
                f"Challenge attempt {attempts} failed{rate_limit_note}: {exc}. Retrying in {wait_time}s..."
            )
            time.sleep(wait_time)


def _is_rate_limit_error(exc: BaseException) -> bool:
    return isinstance(exc, requests.HTTPError) and getattr(exc, "response", None) is not None and exc.response.status_code == 429


def _retry_delay_for_exception(exc: BaseException, base_wait: int, attempts: int) -> int:
    wait_time = max(1, base_wait)
    if not isinstance(exc, requests.HTTPError):
        return wait_time

    response = getattr(exc, "response", None)
    if response is None or response.status_code != 429:
        return wait_time

    retry_after_header = response.headers.get("Retry-After") if response.headers else None
    parsed_wait: Optional[int] = None
    if retry_after_header:
        try:
            parsed_wait = int(float(retry_after_header))
        except ValueError:
            parsed_wait = None

    if parsed_wait is not None:
        wait_time = max(wait_time, parsed_wait)
    else:
        wait_time = max(wait_time, base_wait * (attempts + 1))

    return wait_time


def _auto_accept_incoming_challenges(token: str, stop_event: threading.Event) -> None:
    """Continuously accept every inbound challenge directed at the bot account."""

    accepted_ids: Set[str] = set()
    while not stop_event.is_set():
        try:
            with requests.get(
                f"{LICHESS_API}/api/stream/event",
                headers=_auth_headers(token),
                stream=True,
                timeout=(10, 15),
            ) as response:
                response.raise_for_status()
                print("Listening for incoming challenges to auto-accept...")
                for event in _stream_json_events(response):
                    if stop_event.is_set():
                        return
                    if event.get("type") != "challenge":
                        continue
                    challenge = event.get("challenge") or {}
                    direction = (challenge.get("direction") or "").lower()
                    if direction not in {"in", "incoming"}:
                        continue
                    challenge_id = challenge.get("id")
                    if not challenge_id or challenge_id in accepted_ids:
                        continue
                    challenger = (
                        (challenge.get("challenger") or {}).get("name")
                        or (challenge.get("challenger") or {}).get("id")
                        or "unknown"
                    )
                    try:
                        _accept_challenge(token, challenge_id)
                        accepted_ids.add(challenge_id)
                        print(
                            f"Accepted incoming challenge {challenge_id} from {challenger}."
                        )
                    except requests.RequestException as exc:
                        print(
                            f"Failed to accept challenge {challenge_id} from {challenger}: {exc}"
                        )
        except requests.Timeout:
            if stop_event.is_set():
                return
            print(
                "Incoming challenge stream timed out waiting for data. Reconnecting..."
            )
        except requests.RequestException as exc:
            if stop_event.is_set():
                return
            print(
                f"Incoming challenge stream error: {exc}. Reconnecting in 3s..."
            )
            time.sleep(3)


def _merge_game_metadata(game: Dict, challenge: Optional[Dict], opponent_fallback: str) -> Dict:
    """Fill any missing opponent/color data in ``game`` using the challenge payload."""

    merged = dict(game)
    color_hint = None
    if challenge:
        color_hint = (challenge.get("finalColor") or challenge.get("color") or "").lower()
    if not merged.get("color") and color_hint in {"white", "black"}:
        merged["color"] = color_hint

    opponent_payload = (challenge or {}).get("destUser") or {}
    challenger_payload = (challenge or {}).get("challenger") or {}

    opponent_id = opponent_payload.get("id") or opponent_fallback
    if opponent_id and not (merged.get("opponent") or {}).get("id"):
        merged["opponent"] = {"id": opponent_id}

    my_id = challenger_payload.get("id")
    if my_id and not (merged.get("me") or {}).get("id"):
        merged["me"] = {"id": my_id}

    return merged


def _wait_for_game(
    token: str,
    challenge_id: str,
    opponent: str,
    timeout_seconds: int = 120,
    poll_interval: int = 2,
) -> Dict:
    """Poll the challenge endpoint until it turns into a game and return metadata."""

    start_time = time.time()
    poll_interval = max(1, poll_interval)
    opponent_fallback = opponent
    last_challenge: Optional[Dict] = None

    while True:
        try:
            challenge = _fetch_challenge(token, challenge_id)
            last_challenge = challenge
        except ChallengeNotFoundError:
            active_game = _locate_active_game(token, opponent_fallback)
            if active_game and active_game.get("id"):
                return _merge_game_metadata(active_game, last_challenge, opponent_fallback)
            challenge = None

        if challenge:
            status = (challenge.get("status") or "").lower()

            if status in {"accepted", "started"}:
                active_game = _locate_active_game(token, opponent_fallback)
                if active_game and active_game.get("id"):
                    return _merge_game_metadata(active_game, challenge, opponent_fallback)

            if status in {"declined", "canceled"}:
                raise RuntimeError(f"Challenge {status} by {opponent} before the game started")

        if timeout_seconds > 0 and time.time() - start_time > timeout_seconds:
            raise RuntimeError("Timed out waiting for Lichess to start the game")

        time.sleep(poll_interval)


def _apply_moves(board: chess.Board, moves: str) -> None:
    if not moves:
        return
    for move in moves.split():
        try:
            board.push_uci(move)
        except ValueError as exc:
            raise RuntimeError(f"Received illegal move sequence from Lichess (bad move: {move})") from exc


def _stream_board(
    token: str,
    game_id: str,
    reconnect_delay: int = 2,
) -> Iterable[Dict]:
    print(f"Connecting to the board stream for game {game_id}...")
    reconnect_delay = max(1, reconnect_delay)

    while True:
        try:
            with requests.get(
                f"{LICHESS_API}/api/bot/game/stream/{game_id}",
                headers=_auth_headers(token),
                stream=True,
                timeout=(10, 15),
            ) as response:
                response.raise_for_status()
                for event in _stream_json_events(response):
                    yield event
                # Lichess closed the stream (usually because the game ended).
                return
        except GeneratorExit:
            # The caller stopped consuming events because the game finished.
            return
        except requests.Timeout:
            print(
                f"Board stream for game {game_id} timed out waiting for data. Reconnecting in {reconnect_delay}s..."
            )
        except requests.RequestException as exc:
            print(
                f"Board stream error for game {game_id}: {exc}. Reconnecting in {reconnect_delay}s..."
            )

        time.sleep(reconnect_delay)


def _submit_move(token: str, game_id: str, move: chess.Move) -> None:
    response = requests.post(
        f"{LICHESS_API}/api/bot/game/{game_id}/move/{move.uci()}",
        headers=_auth_headers(token),
        timeout=30,
    )
    response.raise_for_status()


def _should_move(board: chess.Board, my_color: chess.Color) -> bool:
    return board.turn == my_color and not board.is_game_over()


def _infer_color_from_event(event: Dict, my_user_id: Optional[str]) -> Optional[chess.Color]:
    if not my_user_id:
        return None
    my_user_id = my_user_id.lower()
    white_id = ((event.get("white") or {}).get("id") or "").lower()
    black_id = ((event.get("black") or {}).get("id") or "").lower()
    if my_user_id == white_id:
        return chess.WHITE
    if my_user_id == black_id:
        return chess.BLACK
    return None


def _drive_game(
    token: str,
    game_id: str,
    my_color: Optional[chess.Color],
    bot: SacrificeBot,
    my_user_id: Optional[str] = None,
) -> Tuple[Dict, Optional[chess.Color]]:
    board = chess.Board()
    final_state: Dict = {}
    announced_start = False

    for event in _stream_board(token, game_id):
        event_type = event.get("type")
        if event_type not in {"gameFull", "gameState"}:
            continue

        if event_type == "gameFull":
            my_color = my_color or _infer_color_from_event(event, my_user_id)
            if my_color is not None and not announced_start:
                print(
                    f"Game {game_id} started. Playing as {'White' if my_color == chess.WHITE else 'Black'}."
                )
                announced_start = True
            state = event.get("state", {})
        else:
            state = event

        if my_color is None:
            # Wait until we know our color before playing any moves.
            continue

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

            time_budget = my_time
            increment = my_increment
            if time_budget is None:
                time_budget = FALLBACK_MOVE_TIME_MS
                increment = 0

            search_result = bot.choose(board, time_budget, increment)
            if not search_result.move:
                raise RuntimeError("No legal move found for the current position")
            _submit_move(token, game_id, search_result.move)
            time.sleep(0.2)

        status = state.get("status")
        if status and status != "started":
            final_state = state
            break

    return final_state, my_color


def _summarize_game_outcome(state: Optional[Dict], my_color: Optional[chess.Color]) -> str:
    if not state:
        return "Game concluded, but no final status was received from Lichess."

    status = state.get("status", "unknown")
    winner = state.get("winner")
    if my_color is None:
        outcome = "unknown"
    elif winner is None:
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
    parser.add_argument("--depth", type=int, default=16, help="Search depth for SacrificeBot")
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

    try:
        bot_account_id = _fetch_account_id(token)
    except (requests.RequestException, RuntimeError) as exc:
        print(
            "Warning: unable to query /api/account to learn the bot user ID. "
            "Color detection will rely on board events only."
        )
        print(f"Details: {exc}")
        bot_account_id = None

    bot = SacrificeBot(depth=args.depth, sacrifice_margin=args.sacrifice_margin)
    total_games = max(1, args.games)

    stop_event = threading.Event()
    auto_accept_thread = threading.Thread(
        target=_auto_accept_incoming_challenges,
        args=(token, stop_event),
        daemon=True,
    )
    auto_accept_thread.start()

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
        color_hint = (game.get("color") or "").lower()
        if color_hint == "white":
            my_color_hint: Optional[chess.Color] = chess.WHITE
        elif color_hint == "black":
            my_color_hint = chess.BLACK
        else:
            my_color_hint = None

        if my_color_hint is None:
            print(
                f"Challenge {challenge_id} accepted. Waiting for the first board update to learn our color..."
            )
        else:
            print(
                f"Challenge {challenge_id} accepted. Expecting to play as {'White' if my_color_hint == chess.WHITE else 'Black'} once the board stream opens..."
            )

        final_state, resolved_color = _drive_game(
            token,
            game["id"],
            my_color_hint,
            bot,
            my_user_id=bot_account_id,
        )
        print(_summarize_game_outcome(final_state, resolved_color))
        print("Game finished.")

        if game_index < total_games - 1:
            pause = max(0, args.pause_between_games)
            if pause:
                print(f"Waiting {pause}s before issuing the next challenge...")
                time.sleep(pause)

    stop_event.set()
    auto_accept_thread.join(timeout=1)


if __name__ == "__main__":
    main()
