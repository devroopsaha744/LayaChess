# Chess with a Decision Model (Jev / Laya): Research and Plan

_Researched 2026-10-01_

## 1. What the two models are

| | **Jev** (TypeSafe AI) | **Laya** (Convai Innovations) |
|---|---|---|
| Type | Non-autoregressive "decision engine": no generated text, just probabilities over a schema you define | Same idea, open source: bidirectional encoder, one forward pass |
| Architecture | Not disclosed | ModernBERT-large (421M) for English; mmBERT-base (322M) multilingual; a typed-decisions fine-tune |
| Access | Closed API, early access, `POST /v1/systemone` | Apache 2.0 weights on HF (`convaiinnovations/laya`), runs locally (also an MLX port for Mac) |
| Context | 65,536 tokens | 512 (EN) / 1,024 (multi) tokens |
| Options per choice | Up to 255 | ~20 before labels get truncated; server cap 100; `predict_shortlist()` for 50+ |
| Primitives | `choice`, `score` (ordinal), `noul` (calibrated yes/no) | Same three |
| Latency | ~150–276 ms per call (over the network) | ~33 ms on a T4, ~7 ms per question batched; ~13 ms on an M3 Max (MLX) |
| Zero-shot (TypeSafe typed-decisions bench) | 0.727 | 0.362 (random 0.318, majority 0.461) |
| Fine-tuned | No fine-tuning documented | 0.766 after fine-tuning (Kaggle 2×T4 notebook in repo) |
| Price | $0.042 / M input tokens, output free | Free (your own hardware) |

**Takeaway:** neither model was trained on chess. Both are general "score these options given this state" engines. Jev is strong zero-shot but you can't train it. Laya is weak zero-shot, but it's built to be fine-tuned, which is exactly what chess needs.

## 2. Can it be fine-tuned to play chess?

- **Laya: yes.** It has open weights, an Apache 2.0 license, and a fine-tuning notebook that runs on free Kaggle T4s. At 421M parameters it's the same size class as DeepMind's "searchless chess" transformers (9M / 136M / 270M), which reached **2895 Lichess blitz Elo against humans** with no search, trained on 15B Stockfish-labelled data points (ChessBench). So an encoder this size can play strong chess. How strong it gets depends on **data and compute**, not on Laya's text pretraining.
- **Jev: no.** It's a closed API with no fine-tuning. You can only use it as a zero-shot baseline, and we should expect it to play weakly, maybe close to random among legal moves.

## 3. How a decision model plays chess

Chess is already a decision problem: **state = position**, **options = legal moves**. Since we only ever offer legal moves, illegal moves can't happen. That's an advantage over LLM-based chess.

Three ways to frame it, from most "native" to strongest:

**A. One `choice` over legal moves (policy).**
State = FEN, options = the legal moves in UCI form (average ~35, max 218). Jev handles this natively (≤255 options). Laya runs out of option budget after ~20, so it would need `predict_shortlist()` or a cheap pre-filter. Simple to build, but this is behavioural cloning, which DeepMind found to be the weakest target.

**B. One `score` per move (action-value). Recommended first.**
For each legal move, ask "how good is this move?" as an ordinal score over K win-probability bins (e.g. 32–128 bins from Stockfish evals), then play the argmax. This matches DeepMind's best-performing setup (action-value prediction). Each question has a tiny option set, which fits Laya's budget, and the questions batch well: ~35 × 7 ms ≈ 250 ms per move on a T4, faster on an M-series Mac.

**C. Drop the Laya API and use the ModernBERT backbone directly with custom heads.**
Add a policy head over the fixed 1,968-move UCI vocabulary plus a value head. This is the strongest and cheapest option at inference time (one forward pass per position), and it also gives you a value function for search later. You keep Laya's weights and tooling but lose its schema flexibility.

**Input encoding (applies to all three).** Raw FEN tokenizes badly with a BPE tokenizer. Expand it to a fixed 64-square string, with side to move, castling and en passant as separate fields, like DeepMind's fixed 77-char encoding. That gives the model a stable spatial layout.

## 4. Training data

- **Lichess evaluation database**: hundreds of millions of positions with Stockfish evals. Free, and the easiest source.
- **ChessBench** (DeepMind `searchless_chess` repo): action-values for every legal move, the exact labels option B needs.
- **Lichess puzzle DB**: for evaluation, and optionally for fine-tuning tactics.
- Self-labelling: run Stockfish at a fixed depth over positions from Lichess games.

Scale ladder: 1M → 10M → 100M positions. DeepMind's results show Elo keeps rising with data and model size, so plan on stopping where compute runs out rather than at a fixed target.

## 5. Comparing to Stockfish, Leela and others

Be honest about expectations: **a no-search model will not beat full Stockfish 17+ or Lc0 (both ~3600+ CCRL).** Even DeepMind's 270M model is far below them against engines. The interesting comparisons are:

| Opponent | Why it's useful |
|---|---|
| Stockfish with `UCI_LimitStrength` + `UCI_Elo` (1320–3190) | A calibrated Elo ladder: find where we land |
| Stockfish at a fixed depth/nodes (d1, d5, d10) | Search vs no search |
| **Lc0 at `nodes=1`** (policy only, no search) | The fairest comparison: neural net vs neural net, no search on either side |
| Lc0 with full search | Upper bound / reality check |
| Maia (1100–1900) | Human-like reference levels |
| DeepMind searchless 9M/136M/270M | Same "transformer, no search" category |
| Jev zero-shot, Laya zero-shot | Floor / baselines |

**Metrics**
- Elo from matches run with `fastchess` or `cutechess-cli` (UCI), rated with Ordo/BayesElo. Use ≥200 games per pairing with an opening book.
- Lichess puzzle accuracy by rating bucket (DeepMind's headline metric).
- Action accuracy against Stockfish's top move, and Kendall τ of move rankings.
- Latency per move, throughput, and cost per game (Jev).
- Optional: a Lichess bot account for a real-world blitz rating (this is public. We'd do it only with your go-ahead).

## 6. Phased plan

| Phase | What | Output | Rough effort |
|---|---|---|---|
| 0. Setup | Python env, `python-chess`, Stockfish, Lc0 + nets, fastchess, Laya (MLX port on Mac), Jev API key | Working toolchain | ½ day |
| 1. Zero-shot baselines | Wrap Jev and Laya as "pick a legal move" players (framing A); play vs Stockfish Elo 1320 and random; puzzle test | Floor numbers; confirms the pipeline | 1–2 days |
| 2. Data | Pull Lichess evals + ChessBench subset; build the board encoder; make train/val/test splits (separate by game, not by position) | JSONL in Laya's fine-tune format + a tensor format for option C | 2–3 days |
| 3. Fine-tune Laya (B) | Score-per-move action-value on 1M → 10M positions; Kaggle T4s or a rented A100/H100 | Checkpoint + action accuracy / puzzle curves | 3–7 days |
| 4. Fine-tune backbone (C) | Policy + value heads; compare to B at equal compute | Best no-search model | 3–7 days |
| 5. UCI engine | Wrap the best model as a UCI engine (python-chess `engine` or a small UCI loop) | `laya-chess` engine usable in any GUI | 1 day |
| 6. Tournament | Gauntlet vs the table in §5; Elo report | Comparison report / charts | 2–3 days |
| 7. (Optional) Add search | Use the value head in shallow alpha-beta or MCTS (like Lc0) and measure Elo gain per node | Shows how much search adds | open-ended |

## 7. Risks and open questions

- **Laya's text pretraining may add little.** A board is not language. Baseline: train the same architecture from scratch on the same data, so we know whether Laya's initialisation actually helps.
- **The option budget** (~20) makes framing A awkward for Laya. B and C avoid it.
- **The 512-token context** is fine for one position, but too short to add move history.
- **Jev** is early access, and rate limits and pricing can change. Budget is tiny per move, but a 200-game gauntlet is ~16k calls.
- **Threefold repetition / 50-move rule:** a stateless position model doesn't know about them. Handle them in the engine wrapper, as DeepMind did.
- **Compute:** to approach DeepMind-level strength you need billions of labelled pairs and serious GPU time. With a modest budget, expect roughly 1500–2200 Lichess-equivalent. That's an estimate to be confirmed by phases 3–6.

## Sources
- Laya repo: https://github.com/NandhaKishorM/laya · weights: https://huggingface.co/convaiinnovations/laya
- Jev vs Laya: https://www.orcarouter.ai/blog/jev-vs-laya · https://mer.vin/news/laya-the-33ms-open-source-decision-model-beating-jev/
- Laya explained / limits: https://www.orcarouter.ai/blog/laya-decision-model-explained · https://www.eesel.ai/blog/laya-ai-review
- DeepMind, Amortized Planning with Large-Scale Transformers (NeurIPS 2024): https://arxiv.org/abs/2402.04494 · code + ChessBench: https://github.com/google-deepmind/searchless_chess
