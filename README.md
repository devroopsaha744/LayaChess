# LayaChess

Fine-tuning [Laya](https://github.com/NandhaKishorM/laya) (Convai Innovations' open decision model, ModernBERT-large, 421M) to play chess without search, then comparing it with Stockfish, Leela (Lc0) and untrained Laya.

## Idea
Chess as a decision problem: for every legal move, ask Laya a `score` question — *"how good is this move?"* — over 16 win-probability levels. Play the move with the highest expected win probability. Same framing as DeepMind's action-value models in [searchless chess](https://github.com/google-deepmind/searchless_chess).

## Data
DeepMind **ChessBench** `action_value` data: Stockfish win probability for every legal move.
Kaggle dataset (private): `devroopsaha/chessbench-av` — test set + 2 train shards (~3.2 GB).

## Files
| Path | What |
|---|---|
| `notebooks/chessbench_download.ipynb` | Downloads ChessBench shards into Kaggle output → make a dataset from it |
| `notebooks/laya_chess_finetune.ipynb` | Fine-tuning notebook (Kaggle, 2× T4, fp16, checkpoints + resume) |
| `notebooks/kernel-metadata.json` | Kaggle settings for `kaggle kernels push -p notebooks` |
| `engine/` | Chess engine: loads the model from Hugging Face, MCTS search, UCI (see `engine/README.md`) |
| `docs/plan.md` | Research notes and full plan |

## Run on Kaggle
1. Import `notebooks/laya_chess_finetune.ipynb`.
2. Add input: `chessbench-av`. Accelerator **GPU T4 x2**, Internet **ON**.
   Optional: **Add-ons → Secrets** → `HF_TOKEN` (write token) to upload checkpoints to a private Hugging Face repo `<user>/laya-chess`.
3. Run with `QUICK_TEST = True` (~30 min): check sanity cells + loss going down.
4. Set `QUICK_TEST = False` → **Save & Run All (Commit)** (~11 h). Checkpoint lands in `/kaggle/working/ckpt`.

## Plan
- Evaluate: top-move accuracy vs Stockfish, Lichess puzzles.
- Matches (fastchess, opening book): Stockfish `UCI_Elo` 1320 / 1600 / 2000, Lc0 `nodes=1`, untrained Laya.
- Demo: Lichess BOT account + Hugging Face Space showing per-move scores.
