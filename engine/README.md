# LayaChess engine

Plays chess with the fine-tuned Laya checkpoint from Hugging Face (`datafreak/laya-chess`), with or without search.
Write-up: [LayaChess on my blog](https://devroopsaha744.github.io/portfolio/blog/laya-chess/). Play online: [the Hugging Face Space](https://huggingface.co/spaces/datafreak/laya-chess).

| File | What |
|---|---|
| `laya_chess/encoding.py` | Position + move → Laya `score` question (identical to the training notebook) |
| `laya_chess/model.py` | Loads the checkpoint from HF (or a local folder), scores every legal move in batches, caches positions |
| `laya_chess/search.py` | MCTS / PUCT (AlphaZero/Leela style): Laya's move scores → priors + values, exact mate/draw handling, tree reuse |
| `laya_chess/uci.py` | UCI engine for lichess-bot, cutechess/fastchess, chess GUIs |
| `laya_chess/cli.py` | Analyse one position from the terminal |
| `laya_chess/play.py` + `web/` | Play LayaChess in your browser (local board, Laya's win chance per move) |
| `laya_chess/match.py` | Match vs strength-limited Stockfish with an Elo estimate + PGN |
| `tests/` | Search tests with a fast stand-in model (`pytest tests`) |

## Setup
```bash
cd engine
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install -r requirements.txt
```

## Use
```bash
# analyse a position: network scores, then a 30-position search
python -m laya_chess.cli --fen "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4" --nodes 30

# UCI engine (default checkpoint datafreak/laya-chess, latest revision)
python -m laya_chess.uci
python -m laya_chess.uci --nodes 0 --ignore-clock      # network only, no search
python -m laya_chess.uci --checkpoint convaiinnovations/laya   # untrained Laya baseline
```

UCI options: `Checkpoint`, `Revision`, `Device` (auto/cuda/mps/cpu), `Nodes` (fixed evaluations per move, 0 = use the clock),
`MaxNodes`, `CPuct`, `PriorTemp`, `BatchLeaves` (leaves per network call; 1 on a Mac, 4–16 on a CUDA GPU), `MoveOverheadMs`, `UseClock`.

## Play in the browser
```bash
python -m laya_chess.play            # opens http://localhost:8000
```
Pick your colour and how long Laya thinks (instant = network only; 10 to 30 s = search on a Mac).

- **Opening book** (on by default): Laya plays book moves while the game is on a known line (25 built-in mainstream
  openings, or `--book my_book.bin` for any Polyglot book). It leaves the book by itself when the position isn't in the
  book anymore; **Laya takes over** makes it stop using the book right away. Book moves are tagged in the move list.
- **Stockfish's view** (needs `stockfish` installed): after each Laya move, the move full-strength Stockfish would have
  played in that position and whether Laya agreed; **Ask Stockfish** shows Stockfish's move for the current position.

## Match vs Stockfish
```bash
brew install stockfish
python -m laya_chess.match --games 24 --sf-elo 1320 --laya-nodes 0       # network only (~1 h on a Mac)
python -m laya_chess.match --games 24 --sf-elo 1320 --laya-seconds 10    # with search (~3 h on a Mac)
```
Each opening (12 mainstream lines) is played twice with colours swapped. Prints W/D/L, score, Elo difference with a
95% interval and a performance rating on Stockfish's `UCI_Elo` scale; games go to `matches/*.pgn` with Laya's win
chance on every move.

## How the search works
For a position *s*, one batched Laya call scores every legal move *a*: `Q(s,a)` = win chance for the side to move.
- **Priors** `P(a) = softmax(Q(s,a) / PriorTemp)` decide which moves to explore first.
- **Value** of a new leaf = `max_a Q(leaf, a)`; values are backed up with alternating perspective.
- Each edge starts with the network's `Q(s,a)` as one virtual visit, then averages in search results.
- Mate, stalemate, repetition and 50-move draws come from the rules, never from the network.
- Selection: PUCT `Q + c·P·√N / (1 + n)`; the move played is the most visited one. The subtree is reused between moves.

## Speed (measured)
One position = one Laya pass per legal move (~35), so search is expensive:

| Hardware | Move scores / s | Positions / s |
|---|---|---|
| Apple M4 (MPS, fp16) | ~14 | ~0.4 |

Use slow time controls or small `Nodes` on a Mac; a CUDA GPU is several times faster.
