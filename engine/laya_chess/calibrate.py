"""Rate weakened Stockfish opponents on Stockfish's own UCI_Elo scale (anchored at UCI_Elo 1320).

Stockfish cannot go below UCI_Elo 1320 (its lowest skill level). To get rated opponents below that, "sf1320-rNN"
plays a uniformly random legal move NN% of the time and Stockfish 1320's move otherwise. This script plays those
rungs against each other and against plain Stockfish 1320 (CPU only, in parallel) and fits Elo ratings by maximum
likelihood with sf1320 fixed at 1320.

    python -m laya_chess.calibrate --rungs 50 25 10 --games 120 --workers 8
"""
import argparse
import itertools
import json
import math
import multiprocessing as mp
import random
import shutil
import time

import chess
import chess.engine

from .match import OPENINGS

ANCHOR, ANCHOR_ELO = "sf1320", 1320.0


class MixedStockfish:
    """Stockfish UCI_Elo 1320 that plays a random legal move with probability p."""

    def __init__(self, engine, p, seconds, rng):
        self.engine, self.p, self.limit, self.rng = engine, p, chess.engine.Limit(time=seconds), rng

    def move(self, board):
        if self.p > 0 and self.rng.random() < self.p:
            return self.rng.choice(list(board.legal_moves))
        return self.engine.play(board, self.limit).move


def player_p(name):
    return 0.0 if name == ANCHOR else int(name.split("-r")[1]) / 100


def play_one(args):
    """One game; each worker process keeps its own two Stockfish instances."""
    white, black, opening_idx, seed, seconds, stockfish, max_plies = args
    global _ENGINES
    try:
        _ENGINES
    except NameError:
        _ENGINES = []
    while len(_ENGINES) < 2:
        e = chess.engine.SimpleEngine.popen_uci(stockfish)
        e.configure({"UCI_LimitStrength": True, "UCI_Elo": 1320, "Threads": 1, "Hash": 16})
        _ENGINES.append(e)
    rng = random.Random(seed)
    players = {chess.WHITE: MixedStockfish(_ENGINES[0], player_p(white), seconds, rng),
               chess.BLACK: MixedStockfish(_ENGINES[1], player_p(black), seconds, rng)}
    board = chess.Board()
    for uci in OPENINGS[opening_idx % len(OPENINGS)][1].split():
        board.push_uci(uci)
    while not board.is_game_over(claim_draw=True) and board.ply() < max_plies:
        board.push(players[board.turn].move(board))
    outcome = board.outcome(claim_draw=True)
    result = outcome.result() if outcome else "1/2-1/2"
    return {"white": white, "black": black, "result": result, "plies": board.ply()}


def fit_elo(games, anchor=ANCHOR, anchor_elo=ANCHOR_ELO, iters=20000, lr=4.0):
    """Maximum-likelihood Elo (logistic, draws = half point) with one player fixed."""
    names = sorted({g["white"] for g in games} | {g["black"] for g in games})
    R = {n: anchor_elo for n in names}
    for _ in range(iters):
        grad = {n: 0.0 for n in names}
        for g in games:
            s = {"1-0": 1.0, "0-1": 0.0}.get(g["result"], 0.5)
            e = 1 / (1 + 10 ** ((R[g["black"]] - R[g["white"]]) / 400))
            grad[g["white"]] += s - e
            grad[g["black"]] -= s - e
        moved = 0.0
        for n in names:
            if n != anchor:
                step = lr * grad[n]
                R[n] += step
                moved = max(moved, abs(step))
        if moved < 1e-3:
            break
    return R


def bootstrap_ci(games, n=200, seed=0):
    """95% intervals by resampling games."""
    rng = random.Random(seed)
    samples = {}
    for _ in range(n):
        R = fit_elo([rng.choice(games) for _ in games], iters=4000)
        for k, v in R.items():
            samples.setdefault(k, []).append(v)
    return {k: (sorted(v)[int(0.025 * len(v))], sorted(v)[int(0.975 * len(v)) - 1]) for k, v in samples.items()}


def main():
    p = argparse.ArgumentParser(description="Calibrate weakened Stockfish rungs against Stockfish UCI_Elo 1320")
    p.add_argument("--rungs", type=int, nargs="+", default=[50, 25, 10], help="random-move percentages")
    p.add_argument("--games", type=int, default=120, help="games per pairing (even)")
    p.add_argument("--seconds", type=float, default=0.3, help="Stockfish time per move (same as the Laya matches)")
    p.add_argument("--workers", type=int, default=max(1, mp.cpu_count() // 2 - 1))
    p.add_argument("--stockfish", default=shutil.which("stockfish") or "stockfish")
    p.add_argument("--max-plies", type=int, default=300)
    p.add_argument("--out", default="calibration.json")
    a = p.parse_args()

    names = [ANCHOR] + [f"sf1320-r{r}" for r in sorted(a.rungs)]           # strongest first
    pairs = list(zip(names, names[1:])) + [(ANCHOR, n) for n in names[2:]]  # neighbours + everyone vs the anchor
    jobs = []
    for x, y in pairs:
        for g in range(a.games):
            w, b = (x, y) if g % 2 == 0 else (y, x)
            jobs.append((w, b, g // 2, hash((x, y, g)) & 0xFFFFFFFF, a.seconds, a.stockfish, a.max_plies))
    print(f"{len(jobs)} games ({len(pairs)} pairings x {a.games}) on {a.workers} workers", flush=True)

    t0, games = time.time(), []
    with mp.Pool(a.workers) as pool:
        for i, g in enumerate(pool.imap_unordered(play_one, jobs), 1):
            games.append(g)
            if i % 50 == 0 or i == len(jobs):
                R = fit_elo(games, iters=4000)
                print(f"{i}/{len(jobs)} games {time.time() - t0:.0f}s | "
                      + "  ".join(f"{n} {R[n]:.0f}" for n in names if n in R), flush=True)

    R = fit_elo(games)
    ci = bootstrap_ci(games)
    print("\nCalibrated ratings (Stockfish UCI_Elo scale, anchored at sf1320 = 1320):")
    for n in names:
        lo, hi = ci.get(n, (R[n], R[n]))
        print(f"  {n:12s} {R[n]:6.0f}   95% [{lo:.0f}, {hi:.0f}]")
    for x, y in pairs:
        gs = [g for g in games if {g["white"], g["black"]} == {x, y}]
        sx = sum({"1-0": 1.0, "0-1": 0.0}.get(g["result"], 0.5) if g["white"] == x
                 else 1 - {"1-0": 1.0, "0-1": 0.0}.get(g["result"], 0.5) for g in gs)
        print(f"  {x} vs {y}: {sx:g}/{len(gs)}")
    json.dump({"ratings": {n: round(R[n]) for n in names}, "ci95": {n: [round(v) for v in ci[n]] for n in ci},
               "games": games, "seconds": a.seconds, "anchor": {ANCHOR: ANCHOR_ELO}}, open(a.out, "w"), indent=1)
    print(f"saved {a.out}")


if __name__ == "__main__":
    main()
