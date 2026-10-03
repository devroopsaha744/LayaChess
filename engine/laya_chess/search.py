"""Monte Carlo tree search (PUCT, AlphaZero/Leela style) on top of Laya's move scores.

Laya gives, for every legal move a of a position s, the win chance Q(s, a) for the side to move.
One evaluation of s therefore yields:
  * priors  P(a) = softmax(Q(s, a) / prior_temp)   -> which moves to explore first
  * a value V(s) = max_a Q(s, a)                    -> how good s is for the side to move
  * per-move initial values Q(s, a)                 -> used as one "virtual visit" per edge
Checkmate / stalemate / draws are scored exactly from the rules, never by the network.
"""
import math
import threading
import time
from dataclasses import dataclass, field

import chess
import numpy as np

WIN, DRAW, LOSS = 1.0, 0.5, 0.0


def terminal_value(board):
    """Exact value for the side to move if the game is over, else None."""
    outcome = board.outcome(claim_draw=True)
    if outcome is None:
        return None
    if outcome.winner is None:
        return DRAW
    return WIN if outcome.winner == board.turn else LOSS


def winprob_to_cp(p):
    """Inverse of the Lichess win% formula ChessBench used (win% = 1 / (1 + exp(-0.00368208 * cp)))."""
    p = min(max(p, 1e-4), 1 - 1e-4)
    return int(round(math.log(p / (1 - p)) / 0.00368208))


@dataclass
class Node:
    moves: list
    q_net: np.ndarray            # network win chance per move (side to move)
    prior: np.ndarray
    N: np.ndarray                # visits per move
    W: np.ndarray                # summed values per move (side to move's view)
    exact: np.ndarray            # NaN, or exact value of the move (mate / stalemate / draw)
    children: dict = field(default_factory=dict)
    visits: int = 0

    def q(self):
        """Mean value per move: the network estimate counts as one visit, exact results override."""
        q = (self.q_net + self.W) / (1.0 + self.N)
        return np.where(np.isnan(self.exact), q, self.exact)


@dataclass
class SearchResult:
    move: chess.Move
    value: float                 # expected win chance for the side to move
    nodes: int                   # positions evaluated by the network
    simulations: int
    depth: int
    seconds: float
    pv: list
    scores: list                 # (move, network win chance, visits) at the root, best first


class MCTS:
    def __init__(self, model, c_puct=1.5, prior_temp=0.03, batch_leaves=4):
        self.model, self.c_puct, self.prior_temp, self.batch_leaves = model, c_puct, prior_temp, batch_leaves
        self.root, self.root_board = None, None

    # ---------- tree building ----------
    def _make_node(self, board, moves, q):
        q = q.astype(np.float64).copy()
        exact = np.full(len(moves), np.nan)
        for i, mv in enumerate(moves):          # exact results one ply ahead (mate-in-1, stalemate, ...)
            board.push(mv)
            v = terminal_value(board)
            board.pop()
            if v is not None:
                exact[i] = 1.0 - v              # value for the side that played mv
        z = (np.where(np.isnan(exact), q, exact) - q.max()) / self.prior_temp
        prior = np.exp(z - z.max()); prior /= prior.sum()
        return Node(moves, q, prior, np.zeros(len(moves)), np.zeros(len(moves)), exact)

    def set_position(self, board):
        """Start from `board`, reusing the subtree from the previous search when the game continued."""
        node = None
        if self.root is not None and self.root_board is not None:
            old, new = self.root_board.move_stack, board.move_stack
            if len(new) >= len(old) and new[:len(old)] == old and board.root() == self.root_board.root():
                node = self.root
                for mv in new[len(old):]:
                    node = node.children.get(mv) if node is not None else None
        self.root_board = board.copy()
        if node is None:
            moves, q = self.model.evaluate([board])[0]
            node = self._make_node(board, moves, q)
        self.root = node
        return node

    # ---------- one batch of simulations ----------
    def _select(self, node):
        q = node.q()
        u = self.c_puct * node.prior * math.sqrt(node.visits + 1) / (1.0 + node.N)
        return int(np.argmax(q + u))

    def _simulate_batch(self, max_leaves=None):
        """Walk up to `batch_leaves` paths with virtual loss, evaluate their leaves in one network call, back up."""
        pending, boards = [], []
        for _ in range(min(self.batch_leaves, max_leaves or self.batch_leaves)):
            board, node, path = self.root_board.copy(), self.root, []
            while True:
                a = self._select(node)
                path.append((node, a))
                node.N[a] += 1; node.visits += 1            # virtual loss: a visit with value 0 until backed up
                if not np.isnan(node.exact[a]):             # game over after this move: exact value
                    pending.append((path, None, node.exact[a], len(path)))
                    break
                mv = node.moves[a]
                board.push(mv)
                child = node.children.get(mv)
                if child is None:
                    tv = terminal_value(board)              # e.g. threefold repetition reached in the tree
                    if tv is not None:
                        pending.append((path, None, 1.0 - tv, len(path)))
                    else:
                        pending.append((path, board, None, len(path)))
                        boards.append(board)
                    break
                node = child
        unique = {}
        for b in boards:                                    # several paths can reach the same new position
            unique.setdefault(b._transposition_key(), b)
        evals = self.model.evaluate(list(unique.values())) if unique else []
        expanded = {key: self._make_node(b, *ev) for (key, b), ev in zip(unique.items(), evals)}
        depth = 0
        for path, board, value, d in pending:
            depth = max(depth, d)
            if board is not None:
                leaf = expanded[board._transposition_key()]
                parent, a = path[-1]
                parent.children.setdefault(parent.moves[a], leaf)
                value = 1.0 - float(np.max(leaf.q()))       # leaf value for the side that moved into it
            for parent, a in reversed(path):                # alternate perspective on the way up
                parent.W[a] += value
                value = 1.0 - value
        return len(unique), len(pending), depth

    # ---------- public API ----------
    def search(self, board, nodes=None, seconds=None, stop_event=None, info=None, info_every=1.0):
        """Search `board` until `nodes` network evaluations or `seconds` elapsed (whichever first).

        nodes=0 -> no search: play the network's best move.
        """
        t0 = time.time()
        root = self.set_position(board)
        if not root.moves:
            raise ValueError("no legal moves")
        n_eval = n_sims = depth = 0
        last_info = t0
        batch_s = 0.0                                       # duration of the last batch, to avoid overrunning the clock
        if nodes != 0 and len(root.moves) > 1:
            while True:
                if stop_event is not None and stop_event.is_set():
                    break
                if nodes is not None and n_eval >= nodes:
                    break
                if seconds is not None and time.time() - t0 + 0.8 * batch_s >= seconds:
                    break
                tb = time.time()
                e, s, d = self._simulate_batch(None if nodes is None else nodes - n_eval)
                batch_s = time.time() - tb
                n_eval += e; n_sims += s; depth = max(depth, d)
                if np.nanmax(np.where(np.isnan(root.exact), -1, root.exact)) == WIN:
                    break                                   # forced mate in one: nothing to think about
                if info is not None and time.time() - last_info >= info_every:
                    info(self._result(t0, n_eval, n_sims, depth)); last_info = time.time()
        res = self._result(t0, n_eval, n_sims, depth)
        if info is not None:
            info(res)
        return res

    def _result(self, t0, n_eval, n_sims, depth):
        root = self.root
        q = root.q()
        if root.N.sum() > 0:
            a = int(np.lexsort((q, root.N))[-1])           # most visits, ties broken by value
        else:
            a = int(np.argmax(q))
        pv, node, b = [root.moves[a]], root.children.get(root.moves[a]), None
        while node is not None and node.N.sum() > 0:
            i = int(np.argmax(node.N)); pv.append(node.moves[i]); node = node.children.get(node.moves[i])
        order = np.lexsort((q, root.N))[::-1]
        scores = [(root.moves[i], float(root.q_net[i]), int(root.N[i])) for i in order]
        return SearchResult(root.moves[a], float(q[a]), n_eval, n_sims, depth, time.time() - t0, pv, scores)


class SearchThread:
    """Runs a search in the background so a UCI `stop` can interrupt it."""

    def __init__(self, mcts):
        self.mcts, self.stop_event, self.thread, self.result = mcts, threading.Event(), None, None

    def start(self, board, done, **kw):
        self.stop_event.clear()

        def run():
            self.result = self.mcts.search(board, stop_event=self.stop_event, **kw)
            done(self.result)

        self.thread = threading.Thread(target=run, daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join()
