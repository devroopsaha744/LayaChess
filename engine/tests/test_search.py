"""Search tests with a fast stand-in for Laya (material count one ply ahead). Run: `pytest engine/tests`."""
import math

import chess
import numpy as np
import pytest

from laya_chess.search import MCTS

VAL = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}


class MaterialModel:
    """Same interface as LayaChessModel.evaluate: win chance after each move = logistic(material for the mover)."""

    def __init__(self):
        self.calls = 0

    def evaluate(self, boards):
        out = []
        for b in boards:
            self.calls += 1
            moves, q = list(b.legal_moves), []
            for mv in moves:
                me = b.turn
                b.push(mv)
                m = sum(VAL[p.piece_type] * (1 if p.color == me else -1) for p in b.piece_map().values())
                q.append(1 / (1 + math.exp(-0.6 * m)))
                b.pop()
            out.append((moves, np.array(q, dtype=np.float32)))
        return out


def best(fen, nodes=None, seconds=None):
    board = chess.Board(fen)
    r = MCTS(MaterialModel()).search(board, nodes=nodes, seconds=seconds)
    return board.san(r.move), r


@pytest.mark.parametrize("fen,mate", [
    ("6k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 0 1", "Rd8#"),
    ("r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4", "Qxf7#"),
])
def test_finds_mate_in_one(fen, mate):
    assert best(fen, nodes=20)[0] == mate


@pytest.mark.parametrize("fen,blunder", [
    ("4k3/8/4p3/3p4/8/8/8/3QK3 w - - 0 1", "Qxd5"),   # defended pawn: exd5 wins the queen
    ("4k3/8/8/3q4/3P4/4P3/8/4K3 b - - 0 1", "Qxd4"),  # defended pawn: exd4 wins the queen
])
def test_search_avoids_poisoned_pawn(fen, blunder):
    assert best(fen, nodes=0)[0] == blunder          # the 1-ply "network" alone falls for it
    assert best(fen, nodes=20)[0] != blunder         # a small search does not


def test_avoids_stalemate():
    san, r = best("k7/2K5/8/8/8/8/8/1Q6 w - - 0 1", nodes=50)
    board = chess.Board("k7/2K5/8/8/8/8/8/1Q6 w - - 0 1")
    board.push(r.move)
    assert not board.is_stalemate()


@pytest.mark.parametrize("n", [1, 3, 7, 50])
def test_node_limit_is_exact(n):
    assert best(chess.STARTING_FEN, nodes=n)[1].nodes == n


def test_time_limit():
    _, r = best(chess.STARTING_FEN, seconds=0.3)
    assert r.seconds < 0.6


def test_tree_reuse():
    model = MaterialModel()
    mcts = MCTS(model)
    board = chess.Board()
    mcts.search(board, nodes=30)
    a = mcts.root.moves[int(np.argmax(mcts.root.N))]
    reply = mcts.root.children[a]
    b = reply.moves[int(np.argmax(reply.N))] if reply.N.sum() else reply.moves[0]
    board.push(a)
    board.push(b)
    calls = model.calls
    mcts.search(board, nodes=0)
    assert model.calls == calls or reply.children.get(b) is None   # reused when that position was expanded
