Sacrifice Bot
=============

This repository contains a lightweight chess engine that can be used as the
brain for a Lichess bot account.  The main entry point is ``sacrifice_bot.py``
which runs a conventional alpha-beta search and evaluates positions based on
material and piece-square tables.  When multiple lines score roughly the same,
the bot picks the one where it gives up more of its own material – effectively
choosing the most sacrificial winning continuation.

Getting started
---------------

1. Create and activate a Python 3.11+ virtual environment.
2. Install dependencies: ``pip install -r requirements.txt``.
3. Run a sample search from the initial position (depth 3 by default):

   ``python sacrifice_bot.py``

   To analyze a custom FEN and depth, pass the arguments explicitly, for example:

   ``python sacrifice_bot.py "r1bqkbnr/pppp1ppp/2n5/4p3/3PP3/5N2/PPP2PPP/RNBQKB1R w KQkq - 2 4" --depth 4``

4. Wire the ``SacrificeBot`` class into your Lichess bot runner (for example by
   plugging it into ``lichess-bot``'s ``engine.py`` callback) and provide your
   API token via environment variables or a configuration file.  The searcher
   now performs basic time management automatically – just pass the remaining
   clock (and increment) when calling ``choose`` if you have it.

Playing test matches on Lichess
-------------------------------

If you simply want to challenge a specific user (for example ``aldin07``) from
your bot account without wiring up the entire ``lichess-bot`` project, run the
``lichess_challenge.py`` helper:

``python lichess_challenge.py --opponent aldin07 --clock 5 --increment 3``

The script expects your bot token via either the optional positional argument
(place it right after the script name), the ``--token`` flag, or the
``LICHESS_TOKEN`` environment variable and drives the full game via the Bot API
once the opponent accepts.  Use ``--depth`` and ``--sacrifice-margin`` if you
want to tweak the engine parameters, and pass ``--rated`` to play rated games.
If Lichess rejects the challenge, the CLI now prints the exact error reported by
the API instead of crashing with a ``KeyError``, and it understands both of the
slightly different JSON shapes that the challenge endpoint can return.  After a
challenge is accepted, the helper now waits until the actual game appears in the
``/api/account/playing`` feed before opening the board stream, ensuring it
attaches to the real game ID even when Lichess keeps the challenge record alive
for a moment.  If Lichess expires the challenge record the moment it turns into
a game (returning HTTP 404), the helper scans the playing feed to recover the
newly created game ID automatically, so you no longer get stuck in "Waiting for
the game to start" limbo when the opponent accepts instantly.
For repeated sparring, supply
``--games`` to automatically re-challenge the same opponent, ``--challenge-retries``
to keep retrying when the player is busy, ``--retry-wait`` to control how long to
wait between those retries, and ``--pause-between-games`` to control how long the
script waits before the next challenge.  Each finished game now prints a short
summary (win/loss/draw, end status, and move count) so you can quickly confirm
that the bot performed as expected.

Engine behavior
---------------

* Search depth is set through the ``--depth`` CLI option or the ``depth``
  parameter when instantiating ``SacrificeBot``.  When a clock is provided it
  will iteratively deepen until it either reaches that depth or the allocated
  time budget expires.
* Move ordering prioritizes principal-variation moves, transposition-table hits,
  killers, history moves, captures, and checks to improve pruning.
* Quiescence search keeps following forcing moves (captures/checks) at the leaf
  nodes so the evaluation only happens after the position settles.
* Scores are reported in centipawns from the side to move's perspective.
* The ``sacrifice_margin`` argument controls how close two moves must score for
  the engine to prefer the line that gives up more of its own material.
* Iterative deepening uses a shared transposition table so each new depth reuses
  the best line found so far; the CLI prints that principal variation for easy
  analysis after every standalone search.
