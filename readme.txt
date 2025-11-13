Sacrifice Bot
=============

This repository contains a lightweight chess engine that can be used as the
brain for a Lichess bot account.  The main entry point is ``sacrifice_bot.py``
which runs a conventional alpha-beta search and evaluates positions using
material, piece-square tables, mobility, rook-file awareness, king safety, pawn
structure, and passed-pawn heuristics.  When multiple lines score roughly the same, the bot
picks the one where it gives up more of its own material – effectively choosing
the most sacrificial winning continuation.

Getting started
---------------

1. Create and activate a Python 3.11+ virtual environment.
2. Install dependencies: ``pip install -r requirements.txt``.
3. Run a sample search from the initial position (depth 16 by default):

   ``python sacrifice_bot.py``

   To analyze a custom FEN and depth, pass the arguments explicitly, for example:

   ``python sacrifice_bot.py "r1bqkbnr/pppp1ppp/2n5/4p3/3PP3/5N2/PPP2PPP/RNBQKB1R w KQkq - 2 4" --depth 5``

4. Wire the ``SacrificeBot`` class into your Lichess bot runner (for example by
   plugging it into ``lichess-bot``'s ``engine.py`` callback) and provide your
  API token via environment variables or a configuration file.  The searcher
  now performs adaptive time management automatically – just pass the remaining
  clock (and increment) when calling ``choose`` if you have it.  The engine now
  spends only a fraction of the early-game clock (where many pieces remain) and
  relaxes the budget as the position simplifies, so it responds quickly out of
  the opening instead of burning most of its time on move one.

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
Once the real game ID is known the helper now speaks the NDJSON format that the ``/api/bot/game/stream/{gameId}`` endpoint actually returns, so the board stream opens immediately and the engine produces its reply right away instead of waiting for Server-Sent Events frames that never arrive.  The board stream connection now uses a short read timeout plus automatic reconnects, so if Lichess delays the first update or drops the socket mid-game the helper simply re-attaches and keeps pushing moves instead of hanging forever at "Waiting for the game to start".  When Lichess omits the ``wtime/btime`` fields for a few moves (which happens right after the game starts in some challenge flows), pass ``--fallback-move-time`` to cap the think time anyway so the engine keeps replying instantly instead of waiting for a full 16-ply fixed-depth search to finish.
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
  parameter when instantiating ``SacrificeBot``.  When a clock is provided the
  engine now starts at depth 1 and keeps deepening past the requested depth (up
  to 32 plies) until the allocated time budget expires, squeezing out extra
  strength in long games while still obeying the clock.
* Move ordering prioritizes principal-variation moves, transposition-table hits,
  killers, history moves, captures, and checks to improve pruning.
* Quiescence search keeps following forcing moves (captures/checks) at the leaf
  nodes so the evaluation only happens after the position settles, and shallow
  nodes run futility and razoring checks to skip hopeless continuations before
  they waste time.
* Tactical extensions fire for checking moves or pawns racing toward promotion,
  so sharp tactical fights get an extra ply of calculation without slowing down
  quiet positions.
* A tiny deterministic opening book covers the most common early structures so
  the bot plays principled developing moves instantly before the heavy search
  kicks in.
* Scores are reported in centipawns from the side to move's perspective.
* Evaluation mixes classical piece-square values with bishop-pair rewards,
  bitboard mobility counts, rook open-file bonuses, king-safety/pawn-shield heuristics, passed
  pawns that scale into the endgame, outpost detection for minor pieces,
  bitboard-accelerated center control tracking, space advantages in the enemy
  half, activity bonuses for rooks planted on the seventh rank, development
  penalties for idle knights and bishops, doubled/isolated/backward pawn
  penalties, and newly added king-ring attack bonuses so the bot now actively
  rewards piling pieces onto the opponent's monarch instead of only counting
  material and structure cues.
* The ``sacrifice_margin`` argument controls how close two moves must score for
  the engine to prefer the line that gives up more of its own material.
* Iterative deepening now rides on top of aspiration windows, null-move
  pruning, and late-move reductions.  Combined with the shared transposition
  table, PV tracking, and killer/history ordering, each new depth reuses the
  best line found so far and prunes hopeless branches extremely quickly.  The
  CLI prints that principal variation for easy analysis after every standalone
  search.
