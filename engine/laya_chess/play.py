"""Play LayaChess in your browser: `python -m laya_chess.play` then open http://localhost:8000"""
import argparse
import json
import random
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import chess

from .model import DEFAULT_CHECKPOINT, LayaChessModel
from .search import MCTS, winprob_to_cp

WEB = Path(__file__).parent / "web"


class Game:
    def __init__(self, model):
        self.model, self.lock = model, threading.Lock()
        self.mcts = MCTS(model, batch_leaves=1 if model.device.type != "cuda" else 4)
        self.new("white", 0)

    def new(self, color, think):
        self.board = chess.Board()
        self.human = {"white": chess.WHITE, "black": chess.BLACK}.get(color, random.choice([chess.WHITE, chess.BLACK]))
        self.think = float(think)
        self.analysis = None
        self.mcts.root = self.mcts.root_board = None

    def state(self):
        b = self.board
        legal = {}
        for mv in b.legal_moves:
            legal.setdefault(chess.square_name(mv.from_square), []).append(mv.uci()[2:])
        outcome = b.outcome(claim_draw=True)
        last = b.peek().uci() if b.move_stack else None
        san, replay = [], chess.Board()
        for mv in b.move_stack:
            san.append(replay.san(mv)); replay.push(mv)
        return {
            "fen": b.fen(), "turn": "white" if b.turn else "black",
            "human": "white" if self.human else "black", "think": self.think,
            "legal": legal, "last": last, "san": san,
            "check": chess.square_name(b.king(b.turn)) if b.is_check() else None,
            "over": outcome is not None,
            "result": outcome.result() if outcome else None,
            "reason": outcome.termination.name.replace("_", " ").lower() if outcome else None,
            "analysis": self.analysis,
            "model": {"checkpoint": self.model.checkpoint, "tag": self.model.meta.get("tag", "base"),
                      "step": self.model.meta.get("step", 0), "examples": self.model.meta.get("consumed", 0),
                      "device": str(self.model.device)},
        }

    def human_move(self, uci):
        mv = chess.Move.from_uci(uci)
        if self.board.turn != self.human or mv not in self.board.legal_moves:
            raise ValueError("illegal move")
        self.board.push(mv)

    def engine_move(self):
        b = self.board
        if b.turn == self.human or b.outcome(claim_draw=True):
            return
        r = self.mcts.search(b, nodes=0 if self.think <= 0 else None, seconds=self.think if self.think > 0 else None)
        top = [{"san": b.san(mv), "uci": mv.uci(), "win": round(q, 4), "visits": n} for mv, q, n in r.scores[:6]]
        self.analysis = {"move": b.san(r.move), "uci": r.move.uci(), "win": round(r.value, 4),
                         "cp": winprob_to_cp(r.value), "positions": r.nodes, "seconds": round(r.seconds, 1),
                         "pv": b.variation_san(r.pv) if r.pv else "", "top": top, "searched": self.think > 0}
        b.push(r.move)

    def undo(self):
        if self.board.move_stack:
            self.board.pop()
        while self.board.move_stack and self.board.turn != self.human:
            self.board.pop()
        self.analysis = None


def make_handler(game):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body, ctype="application/json"):
            data = body if isinstance(body, bytes) else json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                return self._send(200, (WEB / "index.html").read_bytes(), "text/html; charset=utf-8")
            if self.path == "/api/state":
                with game.lock:
                    return self._send(200, game.state())
            self._send(404, {"error": "not found"})

        def do_POST(self):
            n = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(n) or b"{}")
            try:
                with game.lock:
                    if self.path == "/api/new":
                        game.new(body.get("color", "white"), body.get("think", 0))
                    elif self.path == "/api/move":
                        game.human_move(body["uci"])
                    elif self.path == "/api/engine":
                        game.engine_move()
                    elif self.path == "/api/undo":
                        game.undo()
                    elif self.path == "/api/settings":
                        game.think = float(body.get("think", game.think))
                    else:
                        return self._send(404, {"error": "not found"})
                    return self._send(200, game.state())
            except (ValueError, KeyError) as e:
                self._send(400, {"error": str(e)})

    return Handler


def main():
    p = argparse.ArgumentParser(description="Play LayaChess in the browser")
    p.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    p.add_argument("--revision", default=None)
    p.add_argument("--device", default="auto")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--no-browser", action="store_true")
    a = p.parse_args()
    game = Game(LayaChessModel(a.checkpoint, revision=a.revision, device=a.device))
    server = ThreadingHTTPServer((a.host, a.port), make_handler(game))
    url = f"http://{'localhost' if a.host in ('127.0.0.1', '0.0.0.0') else a.host}:{a.port}"
    print(f"LayaChess board: {url}  (Ctrl+C to stop)", flush=True)
    if not a.no_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
