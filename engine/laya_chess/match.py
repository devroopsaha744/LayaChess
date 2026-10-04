"""LayaChess vs Stockfish (strength-limited) match with an Elo estimate.

    python -m laya_chess.match --games 24 --sf-elo 1320 --laya-nodes 0          # network only
    python -m laya_chess.match --games 24 --sf-elo 1320 --laya-seconds 10       # with search

Each opening is played twice with colours swapped. Games are appended to a PGN as they finish.
"""
import argparse
import datetime
import math
import shutil
import time

import chess
import chess.engine
import chess.pgn

from .model import DEFAULT_CHECKPOINT, LayaChessModel
from .search import MCTS, winprob_to_cp

# short, mainstream openings (UCI), so games start from normal, varied positions
OPENINGS = [
    ("Italian", "e2e4 e7e5 g1f3 b8c6 f1c4"),
    ("Sicilian", "e2e4 c7c5 g1f3 d7d6"),
    ("French", "e2e4 e7e6 d2d4 d7d5"),
    ("Caro-Kann", "e2e4 c7c6 d2d4 d7d5"),
    ("Queen's Gambit Declined", "d2d4 d7d5 c2c4 e7e6"),
    ("King's Indian", "d2d4 g8f6 c2c4 g7g6"),
    ("Nimzo/Queen's Indian setup", "d2d4 g8f6 c2c4 e7e6"),
    ("English", "c2c4 e7e5 b1c3 g8f6"),
    ("Reti", "g1f3 d7d5 g2g3 g8f6"),
    ("Two Knights", "e2e4 e7e5 f1c4 g8f6"),
    ("London", "d2d4 d7d5 c1f4 g8f6"),
    ("Pirc", "e2e4 d7d6 d2d4 g8f6"),
]


def elo_from_score(score):
    score = min(max(score, 1e-3), 1 - 1e-3)
    return -400 * math.log10(1 / score - 1)


def summary(results, sf_elo):
    """results: list of 1 / 0.5 / 0 from Laya's point of view."""
    n = len(results)
    w, d, l = results.count(1.0), results.count(0.5), results.count(0.0)
    s = sum(results) / n
    var = sum((x - s) ** 2 for x in results) / max(1, n - 1)
    se = math.sqrt(var / n) if n > 1 else 0.5
    lo, hi = max(1e-3, s - 1.96 * se), min(1 - 1e-3, s + 1.96 * se)
    diff = elo_from_score(s)
    return (f"{n} games  +{w} ={d} -{l}  score {s:.1%}  Elo diff {diff:+.0f} "
            f"[{elo_from_score(lo):+.0f}, {elo_from_score(hi):+.0f}]  -> performance ~{sf_elo + diff:.0f} "
            f"(Stockfish UCI_Elo scale)")


class LayaPlayer:
    def __init__(self, model, nodes, seconds, batch_leaves):
        self.mcts = MCTS(model, batch_leaves=batch_leaves)
        self.nodes, self.seconds = nodes, seconds

    def new_game(self):
        self.mcts.root = self.mcts.root_board = None

    def play(self, board):
        r = self.mcts.search(board, nodes=self.nodes, seconds=self.seconds)
        return r.move, f"laya {r.value:.1%} cp {winprob_to_cp(r.value)} n {r.nodes} {r.seconds:.1f}s"


def play_game(laya, sf, sf_limit, opening, laya_white, max_plies):
    board = chess.Board()
    for uci in opening[1].split():
        board.push_uci(uci)
    laya.new_game()
    comments = {}
    while not board.is_game_over(claim_draw=True) and board.ply() < max_plies:
        if (board.turn == chess.WHITE) == laya_white:
            mv, note = laya.play(board)
            comments[board.ply()] = note
        else:
            mv = sf.play(board, sf_limit).move
        board.push(mv)
    outcome = board.outcome(claim_draw=True)
    result = outcome.result() if outcome else "1/2-1/2"          # move cap -> adjudicated draw
    return board, result, comments


def to_pgn(board, result, comments, opening, white, black, event):
    game = chess.pgn.Game()
    game.headers.update({"Event": event, "Site": "local", "Date": datetime.date.today().strftime("%Y.%m.%d"),
                         "White": white, "Black": black, "Result": result, "Opening": opening[0]})
    node = game
    replay = chess.Board()
    for mv in board.move_stack:
        ply = replay.ply()
        node = node.add_variation(mv)
        if ply in comments:
            node.comment = comments[ply]
        replay.push(mv)
    return game


def main():
    p = argparse.ArgumentParser(description="LayaChess vs Stockfish match")
    p.add_argument("--games", type=int, default=24, help="even number; each opening twice with colours swapped")
    p.add_argument("--sf-elo", type=int, default=1320, help="Stockfish UCI_Elo (min 1320)")
    p.add_argument("--sf-time", type=float, default=0.5, help="Stockfish seconds per move")
    p.add_argument("--stockfish", default=shutil.which("stockfish") or "stockfish")
    p.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    p.add_argument("--revision", default=None)
    p.add_argument("--device", default="auto")
    p.add_argument("--laya-nodes", type=int, default=None, help="evaluations per move (0 = network only)")
    p.add_argument("--laya-seconds", type=float, default=None, help="thinking time per move (search)")
    p.add_argument("--batch-leaves", type=int, default=1)
    p.add_argument("--max-plies", type=int, default=300)
    p.add_argument("--pgn", default=None, help="output PGN (default: matches/<settings>.pgn)")
    a = p.parse_args()
    if a.laya_nodes is None and a.laya_seconds is None:
        a.laya_nodes = 0

    mode = (f"nodes{a.laya_nodes}" if a.laya_nodes is not None and a.laya_seconds is None else f"{a.laya_seconds:g}s")
    laya_name = f"LayaChess ({mode})"
    sf_name = f"Stockfish (UCI_Elo {a.sf_elo})"
    pgn_path = a.pgn or f"matches/laya-{mode}_vs_sf{a.sf_elo}.pgn"
    import os
    os.makedirs(os.path.dirname(pgn_path) or ".", exist_ok=True)

    model = LayaChessModel(a.checkpoint, revision=a.revision, device=a.device)
    laya = LayaPlayer(model, a.laya_nodes, a.laya_seconds, a.batch_leaves)
    sf = chess.engine.SimpleEngine.popen_uci(a.stockfish)
    sf.configure({"UCI_LimitStrength": True, "UCI_Elo": a.sf_elo, "Threads": 1, "Hash": 16})
    sf_limit = chess.engine.Limit(time=a.sf_time)

    results = []
    t0 = time.time()
    try:
        for g in range(a.games):
            opening = OPENINGS[(g // 2) % len(OPENINGS)]
            laya_white = g % 2 == 0
            tg = time.time()
            board, result, comments = play_game(laya, sf, sf_limit, opening, laya_white, a.max_plies)
            score = {"1-0": 1.0, "0-1": 0.0}.get(result, 0.5)
            results.append(score if laya_white else 1.0 - score)
            white, black = (laya_name, sf_name) if laya_white else (sf_name, laya_name)
            with open(pgn_path, "a") as f:
                print(to_pgn(board, result, comments, opening, white, black, f"{laya_name} vs {sf_name}"), file=f, end="\n\n")
            print(f"game {g + 1}/{a.games} {opening[0]:26s} Laya {'white' if laya_white else 'black'}: {result:7s} "
                  f"{board.ply()} plies {time.time() - tg:.0f}s | {summary(results, a.sf_elo)}", flush=True)
    finally:
        sf.quit()
    print(f"\nFINAL {laya_name} vs {sf_name}: {summary(results, a.sf_elo)}")
    print(f"PGN: {pgn_path} | total {(time.time() - t0) / 60:.0f} min")


if __name__ == "__main__":
    main()
