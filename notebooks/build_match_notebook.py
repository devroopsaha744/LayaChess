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
# LayaChess (fine-tuned + search) vs Stockfish ladder

Fine-tuned Laya (`datafreak/laya-chess`) with MCTS search plays Stockfish at several `UCI_Elo` levels.
Both T4s play games in parallel (one engine per GPU). Every finished game is saved immediately to `/kaggle/working/matches/`.

**Run:** GPU **T4 x2**, Internet **ON**, secret `HF_TOKEN` attached (needed for the private model). Then **Save & Run All (Commit)**.
""")

code("""
# ===== Settings =====
LEVELS = [1320, 1500, 1700, 1900, 2100]   # Stockfish UCI_Elo levels
GAMES_PER_LEVEL = 24                      # 12 openings x both colours
LAYA_NODES = 32                           # positions Laya searches per move (fixed -> same strength on any GPU)
SF_SECONDS = 0.3                          # Stockfish time per move
TIME_BUDGET_H = 5.75                      # stop starting new games after this (~7 h quota left: setup + last games + upload fit)
CHECKPOINT = "datafreak/laya-chess"
REVISION = None                           # pin a commit for exact reproducibility
MAX_PLIES = 300
UPLOAD_TO_HF = True                       # push PGNs + results to the model repo under matches/ (every 30 min + at the end)
BACKUP_EVERY_MIN = 30
""")

code("""
!pip -q install python-chess "git+https://github.com/NandhaKishorM/laya.git"
# official Stockfish build (UCI_Elo down to 1320)
!wget -q -O /tmp/sf.tar https://github.com/official-stockfish/Stockfish/releases/latest/download/stockfish-ubuntu-x86-64-avx2.tar && tar -xf /tmp/sf.tar -C /tmp
!cp $(find /tmp/stockfish -type f -name 'stockfish-ubuntu*' | head -1) /usr/local/bin/stockfish && chmod +x /usr/local/bin/stockfish
!printf 'uci\\nquit\\n' | stockfish | grep -E '^id name|UCI_Elo'
!nvidia-smi --query-gpu=name,memory.total --format=csv
""")

code("""
import os
try:
    from kaggle_secrets import UserSecretsClient
    os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
    print("HF_TOKEN secret found")
except Exception as e:
    raise SystemExit("Attach the HF_TOKEN secret (Add-ons -> Secrets): the model repo is private") from e
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
import json, sys, time, chess, chess.engine
sys.path.insert(0, "/kaggle/working/engine")
from laya_chess.model import LayaChessModel
from laya_chess.match import LayaPlayer, OPENINGS, play_game, to_pgn

cfg = json.loads(sys.argv[1]); wid = cfg["worker"]
model = LayaChessModel(cfg["checkpoint"], revision=cfg["revision"], device="cuda", batch_size=192)
laya = LayaPlayer(model, cfg["nodes"], None, batch_leaves=4)
sf = chess.engine.SimpleEngine.popen_uci("stockfish")
deadline = time.time() + cfg["budget_s"]
out = open(f"/kaggle/working/matches/results_w{wid}.jsonl", "a")
for level, g in cfg["tasks"]:
    if time.time() > deadline:
        print(f"[w{wid}] time budget reached", flush=True); break
    sf.configure({"UCI_LimitStrength": True, "UCI_Elo": level, "Threads": 1, "Hash": 16})
    opening = OPENINGS[(g // 2) % len(OPENINGS)]; laya_white = g % 2 == 0
    t0, n0 = time.time(), model.n_positions
    board, result, comments = play_game(laya, sf, chess.engine.Limit(time=cfg["sf_seconds"]), opening, laya_white, cfg["max_plies"])
    s = {"1-0": 1.0, "0-1": 0.0}.get(result, 0.5); score = s if laya_white else 1 - s
    dt = time.time() - t0
    rec = {"level": level, "game": g, "opening": opening[0], "laya_white": laya_white, "result": result, "score": score,
           "plies": board.ply(), "seconds": round(dt, 1), "positions_per_s": round((model.n_positions - n0) / dt, 2)}
    out.write(json.dumps(rec) + "\\n"); out.flush()
    names = ("LayaChess+search", f"Stockfish {level}") if laya_white else (f"Stockfish {level}", "LayaChess+search")
    with open(f"/kaggle/working/matches/laya_vs_sf{level}_w{wid}.pgn", "a") as f:
        print(to_pgn(board, result, comments, opening, *names, f"LayaChess (nodes {cfg['nodes']}) vs Stockfish UCI_Elo {level}"), file=f, end="\\n\\n")
    print(f"[w{wid}] SF{level} game {g:2d} {opening[0]:26s} Laya {'W' if laya_white else 'B'} {result:7s} "
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
        lines.append(f"  SF {lvl}: {n:2d} games +{r.count(1.0)} ={r.count(0.5)} -{r.count(0.0)}  score {s:5.1%}  -> perf ~{lvl + elo(s):.0f}")
    return "\\n".join(lines)

def backup(note):
    """Copy every finished game to the HF repo, so a killed session loses at most the games in progress."""
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

# overall performance rating: the rating R at which expected score vs all opponents equals the actual score
def expected(R): return sum(1 / (1 + 10 ** ((x["level"] - R) / 400)) for x in rs)
actual = sum(x["score"] for x in rs)
lo, hi = 0.0, 4000.0
for _ in range(60):
    mid = (lo + hi) / 2
    lo, hi = (mid, hi) if expected(mid) < actual else (lo, mid)
perf = (lo + hi) / 2
print(f"\\nOverall performance rating: ~{perf:.0f} on Stockfish's UCI_Elo scale "
      f"({actual:g}/{len(rs)} points, Laya searching {LAYA_NODES} positions per move)")

summary = {"levels": {}, "performance": round(perf), "games": len(rs), "laya_nodes": LAYA_NODES,
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
ax.bar([str(l) for l in lv], sc, color="#769656")
ax.axhline(50, color="#888", lw=1, ls="--")
for i, v in enumerate(sc):
    ax.text(i, v + 1.5, f"{v:.0f}%", ha="center")
ax.set_ylim(0, 105); ax.set_xlabel("Stockfish UCI_Elo"); ax.set_ylabel("LayaChess score (%)")
ax.set_title(f"LayaChess + search ({LAYA_NODES} positions/move) vs Stockfish — perf ~{perf:.0f}")
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
            commit_message=f"matches: {len(rs)} games vs Stockfish, perf ~{perf:.0f}")
        print(f"uploaded to https://huggingface.co/{CHECKPOINT}/tree/main/matches")
    except Exception as e:
        print("HF upload failed (results are still in /kaggle/working/matches):", e)
""")

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
      "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
OUT.write_text(json.dumps(nb, indent=1))
print("wrote", OUT)
