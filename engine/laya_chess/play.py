"""Play LayaChess in your browser: `python -m laya_chess.play` then open http://localhost:8000"""
import argparse
import atexit
import json
import random
import shutil
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import chess
import chess.engine

from .book import OpeningBook
from .model import DEFAULT_CHECKPOINT, LayaChessModel
from .search import MCTS, winprob_to_cp

WEB = Path(__file__).parent / "web"


class Game:
    def __init__(self, model, book_path=None, stockfish=None, sf_time=0.3):
        self.model, self.lock = model, threading.Lock()
        self.mcts = MCTS(model, batch_leaves=1 if model.device.type != "cuda" else 4)
        self.book = OpeningBook(book_path)
        self.sf, self.sf_limit = None, chess.engine.Limit(time=sf_time)
        if stockfish:
            try:
                self.sf = chess.engine.SimpleEngine.popen_uci(stockfish)
                atexit.register(self.sf.quit)
            except Exception as e:
                print("Stockfish not available:", e)
        self.new("white", 0, True)

    def new(self, color, think, use_book=True):
        self.board = chess.Board()
        self.human = {"white": chess.WHITE, "black": chess.BLACK}.get(color, random.choice([chess.WHITE, chess.BLACK]))
        self.think = float(think)
        self.book_on = bool(use_book)      # turned off by "Laya takes over" or when the game leaves the book
        self.tags = []                     # per ply: "you", "book" or "laya"
        self.analysis = self.sf_hint = None
        self.mcts.root = self.mcts.root_board = None

    def stockfish_view(self, board):
        """Full-strength Stockfish's preferred move and evaluation (from the side to move's point of view)."""
        if self.sf is None or board.is_game_over():
            return None
        info = self.sf.analyse(board, self.sf_limit)
        mv = info["pv"][0]
        score = info["score"].pov(board.turn)
        return {"move": board.san(mv), "uci": mv.uci(), "cp": score.score(), "mate": score.mate(),
                "line": board.variation_san(info["pv"][:6])}

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
            "analysis": self.analysis, "sf_hint": self.sf_hint, "tags": self.tags,
            "book": {"on": self.book_on, "opening": self.book.name(b), "polyglot": bool(self.book.polyglot_path)},
            "stockfish": self.sf is not None,
            "model": {"checkpoint": self.model.checkpoint, "tag": self.model.meta.get("tag", "base"),
                      "step": self.model.meta.get("step", 0), "examples": self.model.meta.get("consumed", 0),
                      "device": str(self.model.device)},
        }

    def human_move(self, uci):
        mv = chess.Move.from_uci(uci)
        if self.board.turn != self.human or mv not in self.board.legal_moves:
            raise ValueError("illegal move")
        self.board.push(mv); self.tags.append("you"); self.sf_hint = None

    def engine_move(self):
        b = self.board
        if b.turn == self.human or b.outcome(claim_draw=True):
            return
        sf = self.stockfish_view(b)        # what Stockfish would play here, for comparison
        bm = self.book.move(b) if self.book_on else None
        if bm:
            mv = bm[0]
            b.push(mv); opening = self.book.name(b); b.pop()   # named only once the line is unambiguous
            self.analysis = {"book": True, "move": b.san(mv), "uci": mv.uci(), "opening": opening}
            tag = "book"
        else:
            left_book = self.book_on and len(b.move_stack) > 0
            self.book_on = False           # out of book: Laya plays on its own from here
            r = self.mcts.search(b, nodes=0 if self.think <= 0 else None, seconds=self.think if self.think > 0 else None)
            mv = r.move
            top = [{"san": b.san(m), "uci": m.uci(), "win": round(q, 4), "visits": n} for m, q, n in r.scores[:6]]
            self.analysis = {"book": False, "left_book": left_book, "move": b.san(mv), "uci": mv.uci(),
                             "win": round(r.value, 4), "cp": winprob_to_cp(r.value), "positions": r.nodes,
                             "seconds": round(r.seconds, 1), "pv": b.variation_san(r.pv) if r.pv else "",
                             "top": top, "searched": self.think > 0}
            tag = "laya"
        if sf:
            sf["agree"] = sf["uci"] == mv.uci()
        self.analysis["stockfish"] = sf
        b.push(mv); self.tags.append(tag); self.sf_hint = None

    def takeover(self):
        self.book_on = False

    def hint(self):
        self.sf_hint = self.stockfish_view(self.board)

    def undo(self):
        if self.board.move_stack:
            self.board.pop(); self.tags.pop()
        while self.board.move_stack and self.board.turn != self.human:
            self.board.pop(); self.tags.pop()
        self.analysis = self.sf_hint = None


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
                        game.new(body.get("color", "white"), body.get("think", 0), body.get("book", True))
                    elif self.path == "/api/move":
                        game.human_move(body["uci"])
                    elif self.path == "/api/engine":
                        game.engine_move()
                    elif self.path == "/api/takeover":
                        game.takeover()
                    elif self.path == "/api/hint":
                        game.hint()
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
    p.add_argument("--book", default=None, help="Polyglot .bin opening book (default: built-in book of mainstream lines)")
    p.add_argument("--stockfish", default=shutil.which("stockfish"), help="Stockfish binary, for its preferred move")
    p.add_argument("--sf-time", type=float, default=0.3, help="Stockfish thinking time per comparison (seconds)")
    a = p.parse_args()
    game = Game(LayaChessModel(a.checkpoint, revision=a.revision, device=a.device), a.book, a.stockfish, a.sf_time)
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
