"""UCI front end: `python -m laya_chess.uci` works with lichess-bot, cutechess/fastchess, chess GUIs and python-chess."""
import argparse
import sys
import threading

import chess

from .model import DEFAULT_CHECKPOINT
from .search import MCTS, SearchThread, winprob_to_cp

NAME = "LayaChess"


def send(line):
    sys.stdout.write(line + "\n")
    sys.stdout.flush()


class UciEngine:
    def __init__(self, args):
        self.opts = {
            "Checkpoint": args.checkpoint,
            "Revision": args.revision or "",
            "Device": args.device,
            "Nodes": args.nodes,          # >0: fixed network evaluations per move (overrides the clock), 0: no search
            "MaxNodes": args.max_nodes,   # cap when playing on the clock (0 = no cap)
            "CPuct": args.c_puct,
            "PriorTemp": args.prior_temp,
            "BatchLeaves": args.batch_leaves,
            "MoveOverheadMs": args.move_overhead_ms,
            "UseClock": not args.ignore_clock,
        }
        self.model = self.mcts = self.searcher = None
        self.board = chess.Board()
        self.lock = threading.Lock()

    # ---------- lifecycle ----------
    def ensure_loaded(self):
        if self.model is None:
            from .model import LayaChessModel
            self.model = LayaChessModel(self.opts["Checkpoint"], revision=self.opts["Revision"] or None,
                                        device=self.opts["Device"], verbose=False)
            meta = self.model.meta
            send(f"info string loaded {self.opts['Checkpoint']} tag={meta.get('tag', 'base')} "
                 f"step={meta.get('step', 0)} device={self.model.device}")
        if self.mcts is None:
            self.mcts = MCTS(self.model, c_puct=float(self.opts["CPuct"]), prior_temp=float(self.opts["PriorTemp"]),
                             batch_leaves=int(self.opts["BatchLeaves"]))
            self.searcher = SearchThread(self.mcts)

    def cmd_uci(self):
        send(f"id name {NAME}")
        send("id author devroopsaha744")
        send(f"option name Checkpoint type string default {self.opts['Checkpoint']}")
        send("option name Revision type string default <empty>")
        send(f"option name Device type combo default {self.opts['Device']} var auto var cuda var mps var cpu")
        send(f"option name Nodes type spin default {self.opts['Nodes']} min -1 max 1000000")
        send(f"option name MaxNodes type spin default {self.opts['MaxNodes']} min 0 max 1000000")
        send(f"option name CPuct type string default {self.opts['CPuct']}")
        send(f"option name PriorTemp type string default {self.opts['PriorTemp']}")
        send(f"option name BatchLeaves type spin default {self.opts['BatchLeaves']} min 1 max 64")
        send(f"option name MoveOverheadMs type spin default {self.opts['MoveOverheadMs']} min 0 max 10000")
        send(f"option name UseClock type check default {'true' if self.opts['UseClock'] else 'false'}")
        send("uciok")

    def cmd_setoption(self, tokens):
        if "name" not in tokens:
            return
        i = tokens.index("name")
        j = tokens.index("value") if "value" in tokens else len(tokens)
        name, value = " ".join(tokens[i + 1:j]), " ".join(tokens[j + 1:])
        key = next((k for k in self.opts if k.lower() == name.lower()), None)
        if key is None:
            return send(f"info string unknown option {name}")
        old = self.opts[key]
        if isinstance(old, bool):
            value = value.lower() == "true"
        elif isinstance(old, int):
            value = int(value)
        elif isinstance(old, float):
            value = float(value)
        elif value == "<empty>":
            value = ""
        self.opts[key] = value
        if key in ("Checkpoint", "Revision", "Device"):
            self.model = None
        self.mcts = None   # rebuild with new settings at next search

    def cmd_position(self, tokens):
        if not tokens:
            return
        if tokens[0] == "startpos":
            board, rest = chess.Board(), tokens[1:]
        elif tokens[0] == "fen":
            k = tokens.index("moves") if "moves" in tokens else len(tokens)
            board, rest = chess.Board(" ".join(tokens[1:k])), tokens[k:]
        else:
            return
        if rest and rest[0] == "moves":
            for uci in rest[1:]:
                board.push_uci(uci)
        self.board = board

    def budget(self, tokens):
        """-> (nodes, seconds). Fixed Nodes option wins; otherwise a slice of the remaining clock."""
        args = {}
        it = iter(tokens)
        for t in it:
            if t in ("wtime", "btime", "winc", "binc", "movestogo", "movetime", "nodes", "depth", "mate"):
                args[t] = int(next(it, "0"))
            elif t == "infinite":
                args["infinite"] = True
        if args.get("infinite"):
            return None, None
        if "nodes" in args:
            return args["nodes"], None
        if self.opts["Nodes"] >= 0 and (self.opts["Nodes"] > 0 or not self.opts["UseClock"]):
            return self.opts["Nodes"], None
        overhead = self.opts["MoveOverheadMs"] / 1000
        cap = self.opts["MaxNodes"] or None
        if "movetime" in args:
            return cap, max(0.05, args["movetime"] / 1000 - overhead)
        mine = "wtime" if self.board.turn == chess.WHITE else "btime"
        inc = "winc" if self.board.turn == chess.WHITE else "binc"
        if mine in args:
            left, add = args[mine] / 1000, args.get(inc, 0) / 1000
            moves_left = args.get("movestogo") or max(20, 45 - self.board.fullmove_number // 2)
            t = left / moves_left + 0.75 * add
            t = min(t, 0.4 * left)                         # never risk the flag
            return cap, max(0.05, t - overhead)
        return cap or 100, None                            # bare "go": fixed default

    def cmd_go(self, tokens):
        self.ensure_loaded()
        if self.board.is_game_over():
            return send("bestmove 0000")
        nodes, seconds = self.budget(tokens)
        board = self.board.copy()

        def info(r):
            nps = int(r.nodes / r.seconds) if r.seconds > 0 else 0
            send(f"info depth {max(1, r.depth)} nodes {r.nodes} nps {nps} time {int(r.seconds * 1000)} "
                 f"score cp {winprob_to_cp(r.value)} wdl {int(r.value * 1000)} 0 {1000 - int(r.value * 1000)} "
                 f"pv {' '.join(m.uci() for m in r.pv)}")

        def done(r):
            send(f"bestmove {r.move.uci()}")

        self.searcher.start(board, done, nodes=nodes, seconds=seconds, info=info)

    def loop(self):
        for line in sys.stdin:
            tokens = line.strip().split()
            if not tokens:
                continue
            c, rest = tokens[0], tokens[1:]
            if c == "uci":
                self.cmd_uci()
            elif c == "isready":
                self.ensure_loaded()
                send("readyok")
            elif c == "setoption":
                self.cmd_setoption(rest)
            elif c == "ucinewgame":
                if self.mcts is not None:
                    self.mcts.root = self.mcts.root_board = None
            elif c == "position":
                if self.searcher is not None:
                    self.searcher.stop()
                self.cmd_position(rest)
            elif c == "go":
                if self.searcher is not None:
                    self.searcher.stop()
                self.cmd_go(rest)
            elif c == "stop":
                if self.searcher is not None:
                    self.searcher.stop()
            elif c == "quit":
                if self.searcher is not None:
                    self.searcher.stop()
                break
            elif c == "d":
                send(str(self.board))


def main():
    p = argparse.ArgumentParser(description="LayaChess UCI engine")
    p.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT, help="HF repo id or local folder")
    p.add_argument("--revision", default=None, help="HF revision (commit / branch) of the checkpoint")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    p.add_argument("--nodes", type=int, default=0, help="fixed network evaluations per move; 0 = use the clock")
    p.add_argument("--max-nodes", type=int, default=0, help="cap on evaluations per move when using the clock")
    p.add_argument("--ignore-clock", action="store_true", help="with --nodes 0: play the network move instantly (no search)")
    p.add_argument("--c-puct", type=float, default=1.5)
    p.add_argument("--prior-temp", type=float, default=0.03)
    p.add_argument("--batch-leaves", type=int, default=4, help="leaves evaluated per network call (1 on a slow GPU)")
    p.add_argument("--move-overhead-ms", type=int, default=300)
    UciEngine(p.parse_args()).loop()


if __name__ == "__main__":
    main()
