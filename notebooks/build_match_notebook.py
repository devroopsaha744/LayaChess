"""Builds notebooks/laya_vs_stockfish_kaggle.ipynb with the engine code embedded (the GitHub repo is private).

    python notebooks/build_match_notebook.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENGINE = ROOT / "engine" / "laya_chess"
OUT = ROOT / "notebooks" / "laya_vs_stockfish_kaggle.ipynb"

cells = []
def md(s): cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})
def code(s): cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n")})

md("""
# LayaChess (fine-tuned + search) vs an opponent ladder

Fine-tuned Laya (`datafreak/laya-chess`) with MCTS search plays a ladder of opponents, from a random mover up to
Stockfish at its lowest official strength (`UCI_Elo` 1320, which is the same as Skill Level 0 in Stockfish's code).
Weaker Stockfish rungs limit how many positions it may search per move.
Both T4s play games in parallel (one engine per GPU). Every finished game is saved immediately to `/kaggle/working/matches/`.

**Run:** GPU **T4 x2**, Internet **ON**, secret `HF_TOKEN` attached (needed for the private model). Then **Save & Run All (Commit)**.
""")

code("""
# ===== Settings =====
# Opponent ladder, weakest first:
#   "random"       random legal moves
#   "greedy"       mate-in-1 if any, else the biggest capture/promotion, else random
#   "sf1320-n50"   Stockfish UCI_Elo 1320 limited to 50 nodes per move (very weak)
#   "sf1320-n500"  Stockfish UCI_Elo 1320 limited to 500 nodes per move
#   "sf1320"       Stockfish UCI_Elo 1320 with SF_SECONDS per move (official 1320 strength)
LEVELS = ["random", "greedy", "sf1320-n50", "sf1320-n500", "sf1320"]
KNOWN_ELO = {"sf1320": 1320}              # opponents with an official rating (for the performance estimate)
GAMES_PER_LEVEL = 24                      # 12 openings x both colours
LAYA_NODES = 32                           # positions Laya searches per move (fixed -> same strength on any GPU)
SF_SECONDS = 0.3                          # Stockfish time per move
TIME_BUDGET_H = 5.5                       # stop starting new games after this (quota left ~6.5 h: setup + last games + upload fit)
CHECKPOINT = "datafreak/laya-chess"
REVISION = None                           # pin a commit for exact reproducibility
MAX_PLIES = 300
UPLOAD_TO_HF = True                       # push PGNs + results to the model repo under matches/ (every 30 min + at the end)
BACKUP_EVERY_MIN = 30
""")

code("""
!pip -q install python-chess "git+https://github.com/NandhaKishorM/laya.git"
!nvidia-smi --query-gpu=name,memory.total --format=csv
""")

code("""
# Official Stockfish build (UCI_Elo down to 1320), pinned. Stops the notebook here if it doesn't work,
# so no GPU time is spent on games that can't be played.
import os, shutil, subprocess, tarfile, urllib.request
SF_BUILDS = [  # (url, path of the binary inside the archive)
    ("https://github.com/official-stockfish/Stockfish/releases/download/sf_19/stockfish-linux-x86-64-universal.tar.gz",
     "stockfish/stockfish-linux-x86-64-universal"),
    ("https://github.com/official-stockfish/Stockfish/releases/download/sf_18/stockfish-ubuntu-x86-64-avx2.tar",
     "stockfish/stockfish-ubuntu-x86-64-avx2"),
]
for url, member in SF_BUILDS:
    try:
        archive = "/tmp/" + url.rsplit("/", 1)[1]
        urllib.request.urlretrieve(url, archive)
        with tarfile.open(archive) as t:
            t.extract(member, "/tmp/sf")
        shutil.copy(f"/tmp/sf/{member}", "/usr/local/bin/stockfish"); os.chmod("/usr/local/bin/stockfish", 0o755)
        out = subprocess.run(["stockfish"], input="uci\\nquit\\n", capture_output=True, text=True, timeout=30).stdout
        assert "uciok" in out and "UCI_Elo" in out
        print([l for l in out.splitlines() if l.startswith("id name") or "UCI_Elo" in l])
        break
    except Exception as e:
        print("Stockfish build failed:", url, e)
else:
    raise SystemExit("No working Stockfish -> stopping before any GPU time is used")
""")

code("""
import os
try:
    from kaggle_secrets import UserSecretsClient
    os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
    print("HF_TOKEN secret found")
except Exception as e:
    raise SystemExit("Attach the HF_TOKEN secret (Add-ons -> Secrets): the model repo is private") from e

# check the token before any GPU work: valid, can see the model, can write (for result backups)
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
try:
    who = api.whoami()
    api.model_info(CHECKPOINT)
except Exception as e:
    raise SystemExit(f"HF_TOKEN doesn't work ({type(e).__name__}): create a new token on huggingface.co "
                     f"and update the Kaggle secret (Add-ons -> Secrets -> HF_TOKEN -> Edit)") from e
role = (who.get("auth", {}).get("accessToken", {}) or {}).get("role", "?")
print(f"Hugging Face: logged in as {who['name']}, token role: {role}, model {CHECKPOINT} reachable")
if UPLOAD_TO_HF and role == "read":
    print("WARNING: read-only token -> games run fine but backups to Hugging Face will fail; use a Write token")
""")

md("## Engine code (same files as `engine/laya_chess/` in the repo)")
files = {name: (ENGINE / name).read_text() for name in ["encoding.py", "model.py", "search.py", "match.py"]}
files["__init__.py"] = '"""LayaChess engine (embedded copy)."""\n'
code("import os\nos.makedirs('/kaggle/working/engine/laya_chess', exist_ok=True)\nFILES = " + repr(files) + """
for name, src in FILES.items():
    open(f'/kaggle/working/engine/laya_chess/{name}', 'w').write(src)
print('wrote', sorted(FILES))""")

md("## Worker: one per GPU, plays its share of the games")
code('''
WORKER = r"""
import json, random, sys, time, types, chess, chess.engine
sys.path.insert(0, "/kaggle/working/engine")
from laya_chess.model import LayaChessModel
from laya_chess.match import LayaPlayer, OPENINGS, play_game, to_pgn

cfg = json.loads(sys.argv[1]); wid = cfg["worker"]
model = LayaChessModel(cfg["checkpoint"], revision=cfg["revision"], device="cuda", batch_size=192)
laya = LayaPlayer(model, cfg["nodes"], None, batch_leaves=4)
sf = chess.engine.SimpleEngine.popen_uci("stockfish")
sf.configure({"UCI_LimitStrength": True, "UCI_Elo": 1320, "Threads": 1, "Hash": 16})
VAL = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}

class FuncPlayer:   # same .play(board, limit).move interface as a python-chess engine
    def __init__(self, f): self.f = f
    def play(self, board, limit): return types.SimpleNamespace(move=self.f(board))

def random_move(b):
    return random.choice(list(b.legal_moves))

def greedy_move(b):
    best, best_v = [], -1
    for mv in b.legal_moves:
        b.push(mv); mate = b.is_checkmate(); b.pop()
        if mate:
            return mv
        v = VAL[b.piece_type_at(mv.to_square)] if b.piece_type_at(mv.to_square) else (1 if b.is_en_passant(mv) else 0)
        v += 8 if mv.promotion == chess.QUEEN else 0
        if v > best_v: best, best_v = [mv], v
        elif v == best_v: best.append(mv)
    return random.choice(best)

def opponent(name):
    if name == "random": return FuncPlayer(random_move), None
    if name == "greedy": return FuncPlayer(greedy_move), None
    if name.startswith("sf1320-n"): return sf, chess.engine.Limit(nodes=int(name.split("-n")[1]))
    if name == "sf1320": return sf, chess.engine.Limit(time=cfg["sf_seconds"])
    raise ValueError(name)

deadline = time.time() + cfg["budget_s"]
out = open(f"/kaggle/working/matches/results_w{wid}.jsonl", "a")
for level, g in cfg["tasks"]:
    if time.time() > deadline:
        print(f"[w{wid}] time budget reached", flush=True); break
    opp, limit = opponent(level); random.seed(1000 * g + sum(map(ord, level)))
    opening = OPENINGS[(g // 2) % len(OPENINGS)]; laya_white = g % 2 == 0
    t0, n0 = time.time(), model.n_positions
    board, result, comments = play_game(laya, opp, limit, opening, laya_white, cfg["max_plies"])
    s = {"1-0": 1.0, "0-1": 0.0}.get(result, 0.5); score = s if laya_white else 1 - s
    dt = time.time() - t0
    rec = {"level": level, "game": g, "opening": opening[0], "laya_white": laya_white, "result": result, "score": score,
           "plies": board.ply(), "seconds": round(dt, 1), "positions_per_s": round((model.n_positions - n0) / dt, 2)}
    out.write(json.dumps(rec) + "\\n"); out.flush()
    names = ("LayaChess+search", level) if laya_white else (level, "LayaChess+search")
    with open(f"/kaggle/working/matches/laya_vs_{level}_w{wid}.pgn", "a") as f:
        print(to_pgn(board, result, comments, opening, *names, f"LayaChess (nodes {cfg['nodes']}) vs {level}"), file=f, end="\\n\\n")
    print(f"[w{wid}] {level:12s} game {g:2d} {opening[0]:26s} Laya {'W' if laya_white else 'B'} {result:7s} "
          f"{board.ply()//2} moves {dt/60:.1f} min ({rec['positions_per_s']} pos/s)", flush=True)
sf.quit()
"""
open("/kaggle/working/engine/worker.py", "w").write(WORKER)
print("worker written")
''')

md("## Run both GPUs and show the scoreboard while it plays")
code("""
import json, glob, math, os, subprocess, sys, time
os.makedirs("/kaggle/working/matches", exist_ok=True)

# interleave levels so every level gets games even if the time budget runs out
tasks = [(lvl, g) for g in range(GAMES_PER_LEVEL) for lvl in LEVELS]
n_gpu = 2
procs = []
for w in range(n_gpu):
    cfg = {"worker": w, "tasks": tasks[w::n_gpu], "nodes": LAYA_NODES, "sf_seconds": SF_SECONDS,
           "budget_s": TIME_BUDGET_H * 3600, "checkpoint": CHECKPOINT, "revision": REVISION, "max_plies": MAX_PLIES}
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(w), TOKENIZERS_PARALLELISM="false")
    log = open(f"/kaggle/working/matches/worker{w}.log", "w")
    procs.append(subprocess.Popen([sys.executable, "/kaggle/working/engine/worker.py", json.dumps(cfg)], env=env, stdout=log, stderr=subprocess.STDOUT))
print(f"{len(tasks)} games queued on {n_gpu} GPUs")

def results():
    out = []
    for p in glob.glob("/kaggle/working/matches/results_w*.jsonl"):
        out += [json.loads(l) for l in open(p) if l.strip()]
    return out

def elo(s): s = min(max(s, 1e-3), 1 - 1e-3); return -400 * math.log10(1 / s - 1)

def scoreboard(rs):
    lines = []
    for lvl in LEVELS:
        r = [x["score"] for x in rs if x["level"] == lvl]
        if not r: continue
        n, s = len(r), sum(r) / len(r)
        perf = f"  -> perf ~{KNOWN_ELO[lvl] + elo(s):.0f}" if lvl in KNOWN_ELO else ""
        lines.append(f"  {lvl:12s}: {n:2d} games +{r.count(1.0)} ={r.count(0.5)} -{r.count(0.0)}  score {s:5.1%}{perf}")
    return "\\n".join(lines)

def backup(note):
    # copy every finished game to the HF repo, so a killed session loses at most the games in progress
    if not UPLOAD_TO_HF:
        return
    try:
        from huggingface_hub import HfApi
        HfApi(token=os.environ["HF_TOKEN"]).upload_folder(
            folder_path="/kaggle/working/matches", repo_id=CHECKPOINT, path_in_repo="matches",
            allow_patterns=["*.pgn", "*.json", "*.jsonl", "*.png", "*.log"], commit_message=note)
        print(f"[backup] {note} -> https://huggingface.co/{CHECKPOINT}/tree/main/matches", flush=True)
    except Exception as e:
        print("[backup] failed (will retry next time):", e, flush=True)

t0, seen, last_backup = time.time(), 0, time.time()
while any(p.poll() is None for p in procs):
    time.sleep(120)
    rs = results()
    if len(rs) != seen:
        seen = len(rs)
        pps = [x["positions_per_s"] for x in rs]
        print(f"--- {(time.time() - t0) / 3600:.2f} h | {len(rs)}/{len(tasks)} games | ~{sum(pps)/len(pps):.1f} positions/s per GPU ---")
        print(scoreboard(rs), flush=True)
    if time.time() - last_backup > BACKUP_EVERY_MIN * 60 and rs:
        backup(f"matches (in progress): {len(rs)} games"); last_backup = time.time()
backup(f"matches (all workers done): {len(results())} games")
for w in range(n_gpu):
    print(f"worker {w} exit code {procs[w].returncode}; last log lines:")
    print("".join(open(f"/kaggle/working/matches/worker{w}.log").readlines()[-3:]))
""")

md("## Final results: score per level + overall performance rating")
code("""
rs = results()
print(f"{len(rs)} games played\\n" + scoreboard(rs))

# performance rating vs the rated opponent(s) only: the R at which expected score equals the actual score
rated = [x for x in rs if x["level"] in KNOWN_ELO]
def expected(R): return sum(1 / (1 + 10 ** ((KNOWN_ELO[x["level"]] - R) / 400)) for x in rated)
actual = sum(x["score"] for x in rated)
lo, hi = 0.0, 4000.0
for _ in range(60):
    mid = (lo + hi) / 2
    lo, hi = (mid, hi) if expected(mid) < actual else (lo, mid)
perf = (lo + hi) / 2
if rated:
    print(f"\\nPerformance vs Stockfish 1320: ~{perf:.0f} on Stockfish's UCI_Elo scale "
          f"({actual:g}/{len(rated)} points, Laya searching {LAYA_NODES} positions per move)"
          + ("  [0 or 100% score: only a bound, not an estimate]" if actual in (0, len(rated)) else ""))

summary = {"levels": {}, "performance_vs_sf1320": round(perf) if rated else None, "games": len(rs), "laya_nodes": LAYA_NODES,
           "sf_seconds": SF_SECONDS, "checkpoint": CHECKPOINT, "revision": REVISION}
for lvl in LEVELS:
    r = [x["score"] for x in rs if x["level"] == lvl]
    if r:
        summary["levels"][lvl] = {"games": len(r), "wins": r.count(1.0), "draws": r.count(0.5), "losses": r.count(0.0),
                                  "score": round(sum(r) / len(r), 3)}
json.dump(summary, open("/kaggle/working/matches/summary.json", "w"), indent=1)

import matplotlib.pyplot as plt
lv = [l for l in LEVELS if l in summary["levels"]]
sc = [summary["levels"][l]["score"] * 100 for l in lv]
fig, ax = plt.subplots(figsize=(7, 4))
NICE = {"random": "Random\\nmover", "greedy": "Greedy\\ncapture bot", "sf1320-n50": "Stockfish\\n50 nodes",
        "sf1320-n500": "Stockfish\\n500 nodes", "sf1320": "Stockfish\\nElo 1320"}
ax.bar([NICE.get(l, str(l)) for l in lv], sc, color="#769656")
ax.axhline(50, color="#888", lw=1, ls="--")
for i, v in enumerate(sc):
    ax.text(i, v + 1.5, f"{v:.0f}%", ha="center")
ax.set_ylim(0, 105); ax.set_xlabel("opponent (weakest to strongest)"); ax.set_ylabel("LayaChess score (%)")
ax.set_title(f"LayaChess + search ({LAYA_NODES} positions/move) vs opponent ladder")
for s in ("top", "right"): ax.spines[s].set_visible(False)
plt.tight_layout(); plt.savefig("/kaggle/working/matches/score_by_level.png", dpi=160); plt.show()
""")

code("""
if UPLOAD_TO_HF:
    try:
        from huggingface_hub import HfApi
        HfApi(token=os.environ["HF_TOKEN"]).upload_folder(
            folder_path="/kaggle/working/matches", repo_id=CHECKPOINT, path_in_repo="matches",
            allow_patterns=["*.pgn", "*.json", "*.jsonl", "*.png"],
            commit_message=f"matches: {len(rs)} games vs opponent ladder")
        print(f"uploaded to https://huggingface.co/{CHECKPOINT}/tree/main/matches")
    except Exception as e:
        print("HF upload failed (results are still in /kaggle/working/matches):", e)
""")

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
      "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
OUT.write_text(json.dumps(nb, indent=1))
print("wrote", OUT)
