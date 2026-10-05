"""LayaChess on a Hugging Face Space (Gradio + ZeroGPU).

The page is the local board (laya_chess/web/index.html) inside an iframe. Its API calls go to Python
through a hidden Gradio textbox, one game per visitor. Only Laya's search runs on the GPU; ZeroGPU runs
it in a separate process, so it gets the game as a list of moves and returns plain data.
"""
import html
import json
import os
import shutil
import threading
import time
import types

import spaces  # before torch, as ZeroGPU requires
import chess
import chess.engine
import gradio as gr

from laya_chess import play
from laya_chess.model import DEFAULT_CHECKPOINT, LayaChessModel
from laya_chess.search import MCTS

MAX_THINK = float(os.environ.get("MAX_THINK", 10))
ON_SPACE = bool(os.environ.get("SPACE_ID"))
MODEL = LayaChessModel(os.environ.get("CHECKPOINT", DEFAULT_CHECKPOINT), device="cuda" if ON_SPACE else "auto")

SF_PATH = shutil.which("stockfish") or shutil.which("stockfish", path="/usr/games")
SF = chess.engine.SimpleEngine.popen_uci(SF_PATH) if SF_PATH else None
SF_LOCK = threading.Lock()
if SF:
    SF.configure({"Threads": 1, "Hash": 64})


@spaces.GPU(duration=40)
def gpu_search(moves, seconds):
    board = chess.Board()
    for m in moves:
        board.push_uci(m)
    r = MCTS(MODEL, batch_leaves=8).search(board, nodes=0 if seconds <= 0 else None,
                                          seconds=seconds if seconds > 0 else None)
    return {"move": r.move.uci(), "value": r.value, "nodes": r.nodes, "seconds": r.seconds,
            "pv": [m.uci() for m in r.pv], "scores": [(m.uci(), q, n) for m, q, n in r.scores]}


class RemoteSearch:
    """Stands in for the game's MCTS: each search is one GPU call from the move list."""

    def search(self, board, nodes=None, seconds=None):
        d = gpu_search([m.uci() for m in board.move_stack], 0.0 if nodes == 0 else float(seconds or 0))
        mv = chess.Move.from_uci
        return types.SimpleNamespace(move=mv(d["move"]), value=d["value"], nodes=d["nodes"], seconds=d["seconds"],
                                     pv=[mv(m) for m in d["pv"]], scores=[(mv(m), q, n) for m, q, n in d["scores"]])


class SpaceGame(play.Game):
    def __init__(self):
        super().__init__(MODEL)
        self.mcts, self.sf, self.sf_limit = RemoteSearch(), SF, chess.engine.Limit(time=0.3)

    def new(self, color, think, use_book=True):
        super().new(color, think, use_book)
        self.mcts = RemoteSearch()

    @property
    def think(self):
        return self._think

    @think.setter
    def think(self, seconds):
        self._think = max(0.0, min(float(seconds), MAX_THINK))

    def stockfish_view(self, board):
        with SF_LOCK:
            return super().stockfish_view(board)

    def state(self):
        s = super().state()
        s["model"]["device"] = "ZeroGPU" if ON_SPACE else s["model"]["device"]
        return s


GAMES, GAMES_LOCK = {}, threading.Lock()


def game_for(sid):
    now = time.time()
    with GAMES_LOCK:
        for k in [k for k, (_, seen) in GAMES.items() if now - seen > 3600]:
            del GAMES[k]
        g = GAMES[sid][0] if sid in GAMES else SpaceGame()
        GAMES[sid] = (g, now)
        return g


def handle(req, request: gr.Request):
    """One board API call: {"id", "path", "body"} -> {"id", "data"} or {"id", "error"}."""
    msg = json.loads(req or "{}")
    rid, path, body = msg.get("id"), msg.get("path", ""), msg.get("body") or {}
    g = game_for(request.session_hash or "local")
    try:
        with g.lock:
            if path == "/api/new":
                g.new(body.get("color", "white"), body.get("think", 0), body.get("book", True))
            elif path == "/api/move":
                g.human_move(body["uci"])
            elif path == "/api/engine":
                g.engine_move()
            elif path == "/api/takeover":
                g.takeover()
            elif path == "/api/hint":
                g.hint()
            elif path == "/api/undo":
                g.undo()
            elif path == "/api/settings":
                g.think = float(body.get("think", g.think))
            elif path != "/api/state":
                raise ValueError("unknown call " + path)
            return json.dumps({"id": rid, "data": g.state()})
    except Exception as e:  # shown to the player as an alert, the board stays usable
        return json.dumps({"id": rid, "error": str(e) or e.__class__.__name__})


# the board page, with fetch() replaced by messages to this Gradio page
BRIDGE = """<script>
let laya_n = 0; const laya_wait = {};
window.addEventListener("message", e => {
  const m = e.data || {};
  if (m.laya !== "resp" || !laya_wait[m.id]) return;
  const p = laya_wait[m.id]; delete laya_wait[m.id];
  m.error ? p.rej(new Error(m.error)) : p.res(m.data);
});
function api(path, body) {
  return new Promise((res, rej) => { const id = ++laya_n; laya_wait[id] = { res, rej };
    parent.postMessage({ laya: "req", id, path, body }, "*"); });
}
new ResizeObserver(() => parent.postMessage({ laya: "height", h: document.documentElement.scrollHeight }, "*"))
  .observe(document.documentElement);
</script>"""


def board_page():
    page = (play.WEB / "index.html").read_text()
    start = page.index("async function api(")
    end = page.index("\n}\n", start) + 3
    page = page[:start] + page[end:]                       # the bridge above defines api()
    page = page.replace("</head>", BRIDGE + "</head>", 1)
    return page.replace('<option value="30">30 s (search)</option>', "")


PARENT_JS = """<script>
(() => {
  const queue = []; let busy = false;
  function send() {
    if (busy || !queue.length) return;
    const ta = document.querySelector("#laya-req textarea");
    if (!ta) return setTimeout(send, 200);
    busy = true;
    ta.value = JSON.stringify(queue.shift());
    ta.dispatchEvent(new Event("input", { bubbles: true }));
    setTimeout(() => document.getElementById("laya-go").click(), 0);
  }
  window.layaReply = (v) => {
    const frame = document.getElementById("laya-frame");
    if (v) { try { frame.contentWindow.postMessage({ laya: "resp", ...JSON.parse(v) }, "*"); } catch (e) {} }
    busy = false; send();
  };
  window.addEventListener("message", e => {
    const m = e.data || {};
    if (m.laya === "req") { queue.push({ id: m.id, path: m.path, body: m.body }); send(); }
    if (m.laya === "height") { const f = document.getElementById("laya-frame"); if (f) f.style.height = (m.h + 8) + "px"; }
  });
})();
</script>"""

CSS = ".laya-hidden { display: none !important; } #laya-frame { width: 100%; height: 900px; border: 0; }"

INTRO = """**LayaChess**: [Laya](https://huggingface.co/convaiinnovations/laya), a 421M System 1 decision model, fine-tuned
on 2M Stockfish-rated moves and wrapped in a tree search. Laya thinks on a shared free GPU, so the first move can take
a while, and you may wait in a queue. [How it works](https://devroopsaha744.github.io/portfolio/blog/laya-chess/) ·
[Code](https://github.com/devroopsaha744/LayaChess) · [Video](https://www.youtube.com/watch?v=bPpAlWArs7E)"""

with gr.Blocks(title="LayaChess") as demo:
    gr.Markdown(INTRO)
    gr.HTML(f'<iframe id="laya-frame" srcdoc="{html.escape(board_page(), quote=True)}"></iframe>')
    req = gr.Textbox(elem_id="laya-req", elem_classes="laya-hidden", show_label=False)
    resp = gr.Textbox(elem_classes="laya-hidden", show_label=False)
    go = gr.Button(elem_id="laya-go", elem_classes="laya-hidden")
    go.click(handle, req, resp, concurrency_limit=16, show_progress="hidden")
    resp.change(None, resp, None, js="(v) => { window.layaReply(v); }")

if __name__ == "__main__":
    demo.launch(head=PARENT_JS, css=CSS)
