# LayaChess

A general-purpose AI **decision model** taught to play chess.
[Laya](https://github.com/NandhaKishorM/laya) (Convai Innovations, ModernBERT-large, 421M, Apache 2.0) was never
trained on chess. LayaChess fine-tunes it on DeepMind's [ChessBench](https://github.com/google-deepmind/searchless_chess)
and wraps it in a Leela-style search engine.

- **Play it online:** [huggingface.co/spaces/datafreak/laya-chess](https://huggingface.co/spaces/datafreak/laya-chess)
- **Write-up:** [LayaChess: teaching a System 1 decision model to play chess](https://devroopsaha744.github.io/portfolio/blog/laya-chess/)
- **Demo video:** [youtube.com/watch?v=bPpAlWArs7E](https://www.youtube.com/watch?v=bPpAlWArs7E)
- **Model:** [datafreak/laya-chess](https://huggingface.co/datafreak/laya-chess)

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

## Run it on your own machine
You need Python 3.10+ and about 2 GB of disk for the model. A GPU helps but isn't required; on a Mac it uses Apple's
GPU (MPS) automatically.

1. **Get the code**
   ```bash
   git clone https://github.com/devroopsaha744/LayaChess.git
   cd LayaChess/engine
   ```
2. **Create an environment and install** (with [uv](https://docs.astral.sh/uv/), or plain pip below)
   ```bash
   uv venv --python 3.12 .venv && source .venv/bin/activate
   uv pip install -r requirements.txt
   ```
   Without uv: `python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`
3. **Optional: install Stockfish**, to see Stockfish's preferred move next to Laya's
   ```bash
   brew install stockfish          # macOS
   sudo apt install stockfish      # Debian / Ubuntu
   ```
4. **Play**
   ```bash
   python -m laya_chess.play       # opens the board at http://localhost:8000
   ```
   The first start downloads the model from Hugging Face (about 850 MB), so give it a minute.

Pick your colour and how long Laya thinks: *Instantly* is the network alone, 3 to 30 seconds adds the tree search.
More commands (UCI engine for chess GUIs, matches against Stockfish) are in [`engine/README.md`](engine/README.md).

## Files
| Path | What |
|---|---|
| `notebooks/laya_chess_finetune.ipynb` | v2 fine-tuning (Kaggle 2× T4, fp16, checkpoints to HF, resume) |
| `notebooks/laya_chess_v3_kaggle.ipynb` | v3 training (one pass per position) |
| `notebooks/laya_vs_stockfish_kaggle.ipynb` | matches vs Stockfish on Kaggle |
| `notebooks/chessbench_download.ipynb` | ChessBench files → Kaggle dataset |
| `notebooks/build_*.py` | generate the notebooks above |
| `engine/` | model loading, MCTS, UCI, browser board, matches, tests |
| `space/` | the Hugging Face Space (Gradio + ZeroGPU); `./deploy.sh` uploads it |
| `docs/` | learning curve, plan, post draft |

## Credits
[Laya](https://github.com/NandhaKishorM/laya) by Convai Innovations (Apache 2.0) ·
[ChessBench / searchless chess](https://github.com/google-deepmind/searchless_chess) by Google DeepMind ·
[Stockfish](https://stockfishchess.org) · [python-chess](https://python-chess.readthedocs.io)
