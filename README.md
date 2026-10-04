# LayaChess

A general-purpose AI **decision model** taught to play chess.
[Laya](https://github.com/NandhaKishorM/laya) (Convai Innovations, ModernBERT-large, 421M, Apache 2.0) was never
trained on chess. LayaChess fine-tunes it on DeepMind's [ChessBench](https://github.com/google-deepmind/searchless_chess)
and wraps it in a Leela-style search engine.

![Learning curve](docs/learning_curve.png)

## Results (v2, 2 Kaggle sessions on 2× T4, 2.05M training examples)
| | Base Laya | LayaChess v2 |
|---|---|---|
| Picks Stockfish's best move (300 held-out positions) | 6% | **27%** |
| Win-chance error (percentage points) | 28.6 | **8.2** |
| Opening move as White | b4 | d4 |
| Mate-in-one (Scholar's mate, Qxf7#) | missed (prefers d3) | found, 94% win chance |

Strength in games: still below Stockfish at its weakest official setting (`UCI_Elo` 1320) on a laptop.
A Lichess bot rating is next.

## How it works
- **Question per move** (DeepMind's action-value recipe, asked in Laya's own format): for each legal move Laya gets a
  `score` question such as *"white plays Nf3 (knight g1-f3). Win chance for white?"* with 10 win-chance levels; the
  board is given as piece lists. The answer's expected value is the move's win chance.
- **Engine:** MCTS / PUCT (AlphaZero/Leela style). Laya's move scores become priors and values; mate, stalemate and
  draws come from the rules. UCI front end, browser board, match runner. See [`engine/README.md`](engine/README.md).
- **v3 (in progress):** read the board once and score every move in one pass with Laya's encoder + a move-query head
  (~35× faster search), warm-started from v2.

## Quickstart (play it)
```bash
cd engine
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install -r requirements.txt
hf auth login                      # access to the model repo
python -m laya_chess.play          # browser board at http://localhost:8000
```

## Files
| Path | What |
|---|---|
| `notebooks/laya_chess_finetune.ipynb` | v2 fine-tuning (Kaggle 2× T4, fp16, checkpoints to HF, resume) |
| `notebooks/laya_chess_v3_kaggle.ipynb` | v3 training (one pass per position) |
| `notebooks/laya_vs_stockfish_kaggle.ipynb` | matches vs Stockfish on Kaggle |
| `notebooks/chessbench_download.ipynb` | ChessBench files → Kaggle dataset |
| `notebooks/build_*.py` | generate the notebooks above |
| `engine/` | model loading, MCTS, UCI, browser board, matches, tests |
| `docs/` | learning curve, plan, post draft |

## Credits
[Laya](https://github.com/NandhaKishorM/laya) by Convai Innovations (Apache 2.0) ·
[ChessBench / searchless chess](https://github.com/google-deepmind/searchless_chess) by Google DeepMind ·
[Stockfish](https://stockfishchess.org) · [python-chess](https://python-chess.readthedocs.io)
