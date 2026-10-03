"""Analyse a position: `python -m laya_chess.cli --fen "<FEN>" --nodes 50`."""
import argparse
import time

import chess

from .model import DEFAULT_CHECKPOINT, LayaChessModel
from .search import MCTS, winprob_to_cp


def main():
    p = argparse.ArgumentParser(description="Analyse a chess position with LayaChess")
    p.add_argument("--fen", default=chess.STARTING_FEN)
    p.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    p.add_argument("--revision", default=None)
    p.add_argument("--device", default="auto")
    p.add_argument("--nodes", type=int, default=0, help="network evaluations to search (0 = network only)")
    p.add_argument("--seconds", type=float, default=None)
    p.add_argument("--top", type=int, default=8)
    p.add_argument("--batch-leaves", type=int, default=4)
    a = p.parse_args()

    model = LayaChessModel(a.checkpoint, revision=a.revision, device=a.device)
    board = chess.Board(a.fen)
    print(board, "\n")

    t = time.time()
    scored = model.score_moves(board)
    dt = time.time() - t
    print(f"network only ({len(scored)} moves in {dt:.2f}s):")
    for mv, q in scored[:a.top]:
        print(f"  {board.san(mv):8s} win chance {q:5.1%}")

    if a.nodes or a.seconds:
        r = MCTS(model, batch_leaves=a.batch_leaves).search(board, nodes=a.nodes or None, seconds=a.seconds)
        print(f"\nsearch: {r.nodes} positions, {r.simulations} simulations, depth {r.depth}, {r.seconds:.1f}s "
              f"({r.nodes / max(r.seconds, 1e-9):.2f} positions/s)")
        print(f"  best {board.san(r.move)}  value {r.value:.1%} (cp {winprob_to_cp(r.value)})  pv {board.variation_san(r.pv)}")
        for mv, q, n in r.scores[:a.top]:
            print(f"  {board.san(mv):8s} net {q:5.1%}  visits {n}")


if __name__ == "__main__":
    main()
