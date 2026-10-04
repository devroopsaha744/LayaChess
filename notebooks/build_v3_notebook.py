"""Builds notebooks/laya_chess_v3_kaggle.ipynb: Laya encoder + move-query head (one pass per position).

    python notebooks/build_v3_notebook.py
"""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent / "laya_chess_v3_kaggle.ipynb"
cells = []
def md(s): cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})
def code(s): cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n")})

md("""
# LayaChess v3: one pass per position

v2 asks Laya one question per legal move (~35 passes of ~194 tokens per position).
v3 reads **the board once** (~80 tokens) with the Laya encoder (warm-started from the fine-tuned v2 checkpoint) and a
small **move-query head** scores any set of moves against that encoding: each move (from, to, piece, promotion) is a
query that cross-attends to the board tokens and predicts its win chance over `N_BINS` levels. Queries don't attend to
each other, so training on one move per position (ChessBench train shards are shuffled globally: ~1 move per
position) matches scoring all legal moves at once when playing.

Same data (ChessBench action values) and the same 300 validation positions as v2, so top-move accuracy is comparable.

**Run:** GPU **T4 x2**, Internet **ON**, input dataset `chessbench-av`, secret `HF_TOKEN` (write).
First `QUICK_TEST = True` (~25 min), then `QUICK_TEST = False` → **Save & Run All (Commit)**.
""")

code("""
# ===== Settings =====
import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
QUICK_TEST = True

CFG = dict(
    INIT="v2",                    # "v2" = encoder from datafreak/laya-chess (fine-tuned), "base" = convaiinnovations/laya
    V2_REPO="datafreak/laya-chess",
    BASE_REPO="convaiinnovations/laya",
    HF_REPO_NAME="laya-chess-v3", # checkpoints -> <your HF user>/laya-chess-v3 (private)
    N_TRAIN_SHARDS=2,
    MAX_TRAIN_RECORDS=None,
    N_BINS=32,
    LABEL_SIGMA=0.75,             # soft-label width in bins
    HEAD_LAYERS=2,
    HEAD_HEADS=16,
    HEAD_FF=2048,
    BATCH_SIZE=128,               # records per step over both GPUs
    GRAD_ACCUM=1,
    LR_ENCODER=3e-5,
    LR_HEAD=5e-4,
    WARMUP_STEPS=300,
    MIN_LR_FRAC=0.05,
    WEIGHT_DECAY=0.01,
    NUM_WORKERS=4,
    TIME_BUDGET_H=11.0,
    CKPT_EVERY_MIN=45,
    EVAL_EVERY_STEPS=2000,
    LOG_EVERY_STEPS=50,
    N_VAL_RECORDS=4000,
    N_VAL_POSITIONS=300,          # same 300 positions as v2 (same selection code)
    N_VAL_POSITIONS_BIG=2000,     # tighter estimate, cheap with one pass per position
    SAVE_OPTIMIZER=True,
    RESUME_FROM=None,             # "hf" = latest v3 checkpoint on Hugging Face
    RESUME_LR_SCALE=0.5,
    SEED=0,
)
if QUICK_TEST:
    CFG.update(TIME_BUDGET_H=0.35, CKPT_EVERY_MIN=12, EVAL_EVERY_STEPS=300, N_VAL_RECORDS=1000,
               N_VAL_POSITIONS_BIG=500, SAVE_OPTIMIZER=False, HF_REPO_NAME="laya-chess-v3-quicktest")
BASE_DIR, CKPT_DIR = "/tmp/laya_base", "/kaggle/working/ckpt_v3"

HF_TOKEN = None
try:
    from kaggle_secrets import UserSecretsClient
    HF_TOKEN = UserSecretsClient().get_secret("HF_TOKEN")
    os.environ["HF_TOKEN"] = HF_TOKEN
except Exception as e:
    print("no HF_TOKEN secret:", type(e).__name__)
if CFG["INIT"] == "v2" and not HF_TOKEN:
    raise SystemExit("INIT='v2' needs the HF_TOKEN secret (the v2 model is private)")
print(CFG)
""")

code("""
!pip -q install python-chess "git+https://github.com/NandhaKishorM/laya.git"
!nvidia-smi --query-gpu=name,memory.total --format=csv
""")

md("## 1. ChessBench data (attached `chessbench-av` dataset)")
code("""
import glob, mmap, struct, chess
found = glob.glob("/kaggle/input/**/test_action_value.bag", recursive=True)
assert found, "attach the chessbench-av dataset"
DATA_DIR = os.path.dirname(found[0])

class BagReader:
    def __init__(self, path): self.path = path; self._open()
    def _open(self):
        f = open(self.path, "rb"); self.mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        size = len(self.mm); self.index_start = struct.unpack("<q", self.mm[size - 8:])[0]
        self.n = (size - self.index_start) // 8
    def __len__(self): return self.n
    def __getitem__(self, i):
        end = struct.unpack_from("<q", self.mm, self.index_start + 8 * i)[0]
        start = 0 if i == 0 else struct.unpack_from("<q", self.mm, self.index_start + 8 * (i - 1))[0]
        return self.mm[start:end]
    def __getstate__(self): return {"path": self.path}
    def __setstate__(self, s): self.path = s["path"]; self._open()

def _varint(b, p):
    r = s = 0
    while True:
        x = b[p]; p += 1; r |= (x & 0x7F) << s
        if x < 0x80: return r, p
        s += 7

def decode_av(b):
    n, p = _varint(b, 0); fen = b[p:p + n].decode(); p += n
    n, p = _varint(b, p); move = b[p:p + n].decode(); p += n
    return fen, move, struct.unpack(">d", b[p:p + 8])[0]

test_bag = BagReader(f"{DATA_DIR}/test_action_value.bag")
train_bags = [BagReader(f"{DATA_DIR}/train_action_value_{i:05d}.bag") for i in range(CFG["N_TRAIN_SHARDS"])]
print("test records:", len(test_bag), "| train records:", [len(b) for b in train_bags])
""")

md("## 2. Laya encoder (warm start) + move-query head")
code("""
import json, math, random, shutil, time
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from huggingface_hub import snapshot_download, HfApi
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer
from laya.agent import _fix_tokenizer_config
from laya.common import build_model, serialize_state

random.seed(CFG["SEED"]); np.random.seed(CFG["SEED"]); torch.manual_seed(CFG["SEED"])
snapshot_download(CFG["BASE_REPO"], local_dir=BASE_DIR,
                  allow_patterns=["model.safetensors", "rl_agent_config.json", "encoder/*", "tokenizer/*"])
MODEL_DIR = Path(BASE_DIR); _fix_tokenizer_config(str(MODEL_DIR))
lcfg = json.loads((MODEL_DIR / "rl_agent_config.json").read_text())
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR / "tokenizer")
dm = build_model(lcfg, encoder_dir=str(MODEL_DIR / "encoder"), pretrained=False)
init_dir = MODEL_DIR if CFG["INIT"] == "base" else Path(snapshot_download(
    CFG["V2_REPO"], token=HF_TOKEN, allow_patterns=["model.safetensors", "chess_meta.json"]))
dm.load_state_dict({k: v.float() for k, v in load_file(str(init_dir / "model.safetensors")).items()}, strict=True)
encoder = dm.encoder
del dm
H = encoder.config.hidden_size
print(f"encoder from {CFG['INIT']} ({init_dir}) | hidden {H} | params {sum(p.numel() for p in encoder.parameters())/1e6:.0f}M")

PROMOS = {None: 0, chess.KNIGHT: 1, chess.BISHOP: 2, chess.ROOK: 3, chess.QUEEN: 4}

class CrossBlock(nn.Module):
    \"\"\"Move queries attend to the board tokens only (never to each other).\"\"\"
    def __init__(self, h, heads, ff):
        super().__init__()
        self.n1, self.attn = nn.LayerNorm(h), nn.MultiheadAttention(h, heads, batch_first=True)
        self.n2, self.ff = nn.LayerNorm(h), nn.Sequential(nn.Linear(h, ff), nn.GELU(), nn.Linear(ff, h))
    def forward(self, q, mem, mem_pad):
        x = self.n1(q)
        q = q + self.attn(x, mem, mem, key_padding_mask=mem_pad, need_weights=False)[0]
        return q + self.ff(self.n2(q))

class MoveHead(nn.Module):
    def __init__(self, h, n_bins, layers, heads, ff):
        super().__init__()
        self.frm, self.dst = nn.Embedding(64, h), nn.Embedding(64, h)   # not `self.to`: that's nn.Module.to
        self.piece, self.promo = nn.Embedding(13, h), nn.Embedding(5, h)
        self.blocks = nn.ModuleList(CrossBlock(h, heads, ff) for _ in range(layers))
        self.out = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, n_bins))
    def forward(self, mem, mem_pad, mv):     # mv: [B, M, 4] (from, to, piece, promo)
        q = self.frm(mv[..., 0]) + self.dst(mv[..., 1]) + self.piece(mv[..., 2]) + self.promo(mv[..., 3])
        for blk in self.blocks:
            q = blk(q, mem, mem_pad)
        return self.out(q)                    # [B, M, n_bins]

class LayaChessV3(nn.Module):
    def __init__(self, encoder, head):
        super().__init__(); self.encoder, self.head = encoder, head
    def forward(self, ids, att, mv):
        mem = self.encoder(input_ids=ids, attention_mask=att).last_hidden_state
        return self.head(mem.float(), att == 0, mv)

head = MoveHead(H, CFG["N_BINS"], CFG["HEAD_LAYERS"], CFG["HEAD_HEADS"], CFG["HEAD_FF"])
model = LayaChessV3(encoder, head)
print(f"head params {sum(p.numel() for p in head.parameters())/1e6:.1f}M")

HF_REPO = None
if HF_TOKEN:
    api = HfApi(token=HF_TOKEN)
    HF_REPO = f"{api.whoami()['name']}/{CFG['HF_REPO_NAME']}"
    api.create_repo(HF_REPO, private=True, exist_ok=True)
    print("checkpoints ->", f"https://huggingface.co/{HF_REPO}")

meta = None
if CFG["RESUME_FROM"] == "hf":
    CFG["RESUME_FROM"] = snapshot_download(HF_REPO, token=HF_TOKEN, local_dir="/tmp/resume_v3")
if CFG["RESUME_FROM"]:
    meta = json.loads((Path(CFG["RESUME_FROM"]) / "chess_meta.json").read_text())
    model.load_state_dict({k: v.float() for k, v in load_file(str(Path(CFG["RESUME_FROM"]) / "model.safetensors")).items()})
    print("resumed weights:", meta["tag"], "step", meta["step"], "examples", meta["consumed"])
""")

md("## 3. Encoding: board text (same state format as v2) + move features")
code("""
NAMES_ORDER = [chess.KING, chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT, chess.PAWN]
def piece_list(board, color):
    return " ".join(chess.piece_symbol(pt).upper() + chess.square_name(sq)
                    for pt in NAMES_ORDER for sq in sorted(board.pieces(pt, color)))
def board_text(board):
    return serialize_state({
        "to_move": "white" if board.turn else "black",
        "white": piece_list(board, chess.WHITE), "black": piece_list(board, chess.BLACK),
        "castling": board.castling_xfen() if board.castling_rights else "-",
        "en_passant": chess.square_name(board.ep_square) if board.has_legal_en_passant() else "-"})
def encode_board(board):
    return tokenizer(board_text(board), add_special_tokens=True, truncation=True, max_length=192)["input_ids"]
def move_feat(board, mv):
    p = board.piece_at(mv.from_square)
    return [mv.from_square, mv.to_square, p.piece_type - 1 + (0 if p.color == chess.WHITE else 6), PROMOS[mv.promotion]]

N_BINS = CFG["N_BINS"]
CENTERS = (np.arange(N_BINS) + 0.5) / N_BINS
def soft_target(wp):
    t = np.exp(-0.5 * ((np.arange(N_BINS) - (wp * N_BINS - 0.5)) / CFG["LABEL_SIGMA"]) ** 2)
    return (t / t.sum()).astype(np.float32)

fen, mv, wp = decode_av(train_bags[0][0]); b = chess.Board(fen)
ids = encode_board(b)
print(len(ids), "tokens:", tokenizer.decode(ids)); print("move features", move_feat(b, chess.Move.from_uci(mv)), "win", round(wp, 3))
lens = [len(encode_board(chess.Board(decode_av(train_bags[0][i])[0]))) for i in range(0, 20000, 250)]
print("board tokens min/mean/max:", min(lens), int(np.mean(lens)), max(lens))
""")

md("## 4. Datasets (same fixed order as v2: file 1 shuffled, then file 2 shuffled)")
code("""
from torch.utils.data import Dataset, DataLoader

class RecordDataset(Dataset):      # one (position, move, win chance) per item
    def __init__(self, refs, bags): self.refs, self.bags = refs, bags
    def __len__(self): return len(self.refs)
    def __getitem__(self, i):
        k, r = self.refs[i]; fen, mv, wp = decode_av(self.bags[k][r]); b = chess.Board(fen)
        return encode_board(b), [move_feat(b, chess.Move.from_uci(mv))], soft_target(wp)[None], np.array([wp], np.float32)

PAD = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
def collate(items):
    L = max(len(x[0]) for x in items); M = max(len(x[1]) for x in items); B = len(items)
    ids = torch.full((B, L), PAD, dtype=torch.long); att = torch.zeros((B, L), dtype=torch.long)
    mv = torch.zeros((B, M, 4), dtype=torch.long); mmask = torch.zeros((B, M), dtype=torch.bool)
    tgt = torch.zeros((B, M, N_BINS)); wp = torch.zeros((B, M))
    for j, (i, m, t, w) in enumerate(items):
        ids[j, :len(i)] = torch.tensor(i); att[j, :len(i)] = 1
        mv[j, :len(m)] = torch.tensor(m); mmask[j, :len(m)] = True
        tgt[j, :len(m)] = torch.from_numpy(t); wp[j, :len(m)] = torch.from_numpy(w)
    return ids, att, mv, mmask, tgt, wp

rng = np.random.default_rng(CFG["SEED"])
def shard_refs(k):
    return np.stack([np.full(len(train_bags[k]), k, dtype=np.int32), np.arange(len(train_bags[k]), dtype=np.int32)], 1)
parts = [shard_refs(0)]; rng.shuffle(parts[0])
val_idx = rng.choice(len(test_bag), CFG["N_VAL_RECORDS"], replace=False)
for k in range(1, len(train_bags)):
    p = shard_refs(k); np.random.default_rng(CFG["SEED"] + k).shuffle(p); parts.append(p)
train_refs = np.concatenate(parts); del parts
if CFG["MAX_TRAIN_RECORDS"]: train_refs = train_refs[:CFG["MAX_TRAIN_RECORDS"]]
val_ds = RecordDataset(np.stack([np.zeros_like(val_idx), val_idx], 1), [test_bag])

by_fen = {}
for i in range(len(test_bag)):
    fen, mv, wp = decode_av(test_bag[i]); by_fen.setdefault(fen, []).append((mv, wp))
all_pos = [(f, m) for f, m in by_fen.items() if len(m) >= 2]
random.Random(CFG["SEED"]).shuffle(all_pos)
val_positions = all_pos[:CFG["N_VAL_POSITIONS"]]                                   # identical to v2's 300
val_positions_big = all_pos[:CFG["N_VAL_POSITIONS_BIG"]]
del by_fen, all_pos
print("train records:", len(train_refs), "| val records:", len(val_ds), "| val positions:", len(val_positions), len(val_positions_big))
""")

md("## 5. Train")
code("""
device = "cuda"
for m in model.modules():
    if hasattr(m, "config") and hasattr(m.config, "reference_compile"): m.config.reference_compile = False
from transformers.modeling_utils import PreTrainedModel
def _first_float_tensor(mod):
    for p in mod.parameters():
        if p.is_floating_point(): return p
    for sub in mod.modules():
        for v in vars(sub).values():
            if torch.is_tensor(v) and v.is_floating_point(): return v
    raise StopIteration
PreTrainedModel.dtype = property(lambda self: _first_float_tensor(self).dtype)
PreTrainedModel.device = property(lambda self: _first_float_tensor(self).device)

model.float().to(device)
CENTERS_T = torch.tensor(CENTERS, dtype=torch.float32, device=device)
lr_scale = CFG["RESUME_LR_SCALE"] if CFG["RESUME_FROM"] else 1.0
groups = [{"params": list(model.encoder.parameters()), "peak": CFG["LR_ENCODER"] * lr_scale},
          {"params": list(model.head.parameters()), "peak": CFG["LR_HEAD"] * lr_scale}]
for g in groups: g["lr"] = g["peak"]
opt = torch.optim.AdamW(groups, weight_decay=CFG["WEIGHT_DECAY"])
scaler = torch.amp.GradScaler("cuda")

def loss_fn(logits, mmask, tgt):
    logp = torch.log_softmax(logits.float(), -1)
    per_move = -(tgt * logp).sum(-1)
    return (per_move * mmask).sum() / mmask.sum().clamp(min=1)

def expected_wp(logits):
    return (torch.softmax(logits.float(), -1) * CENTERS_T).sum(-1)

probe_items = sorted((val_ds[i] for i in range(min(512, len(val_ds)))), key=lambda x: -len(x[0]))
def probe(net_, bs):
    ids, att, mv, mmask, tgt, _ = [t.to(device) for t in collate(probe_items[:bs])]
    with torch.autocast("cuda", dtype=torch.float16):
        logits = net_(ids, att, mv)
    loss_fn(logits, mmask, tgt).backward()
    for g in opt.param_groups: g["lr"] = 0.0
    opt.step(); opt.zero_grad(set_to_none=True)
    for g in opt.param_groups: g["lr"] = g["peak"]
    torch.cuda.synchronize()

net = model
if torch.cuda.device_count() > 1:
    try:
        dp = torch.nn.DataParallel(model)
        ids, att, mv, *_ = [t.to(device) for t in collate(probe_items[:2])]
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16): dp(ids, att, mv)
        net = dp; print("using DataParallel on", torch.cuda.device_count(), "GPUs")
    except Exception as e:
        print("DataParallel failed -> single GPU:", type(e).__name__, str(e)[:200])
        CFG["BATCH_SIZE"] //= 2; CFG["GRAD_ACCUM"] *= 2
grad_ckpt = False
while True:
    try:
        probe(net, CFG["BATCH_SIZE"]); break
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        if "out of memory" not in str(e).lower(): raise
        opt.zero_grad(set_to_none=True); torch.cuda.empty_cache()
        if not grad_ckpt:
            model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            grad_ckpt = True; print("OOM -> gradient checkpointing on")
        elif CFG["BATCH_SIZE"] > 8:
            CFG["BATCH_SIZE"] //= 2; CFG["GRAD_ACCUM"] *= 2; print("OOM -> batch", CFG["BATCH_SIZE"])
        else: raise
opt.state.clear(); torch.cuda.empty_cache()
print(f"setup: batch {CFG['BATCH_SIZE']} x accum {CFG['GRAD_ACCUM']} | grad ckpt {grad_ckpt} | "
      f"GPU0 peak {torch.cuda.max_memory_allocated(0) / 2**30:.1f} GB")

step, consumed = 0, 0
if meta:
    step, consumed = meta["step"], meta["consumed"]
    st_path = Path(CFG["RESUME_FROM"]) / "train_state.pt"
    if st_path.exists():
        st = torch.load(st_path, map_location="cpu", weights_only=False)
        opt.load_state_dict(st["opt"]); scaler.load_state_dict(st["scaler"])
        for g, src in zip(opt.param_groups, groups): g["peak"] = src["peak"]
        print("optimizer state restored")
    print(f"resumed at step {step}, {consumed} records consumed")
if consumed >= len(train_refs): consumed = 0

train_dl = DataLoader(RecordDataset(train_refs[consumed:], train_bags), batch_size=CFG["BATCH_SIZE"], shuffle=False,
                      collate_fn=collate, num_workers=CFG["NUM_WORKERS"], persistent_workers=True, prefetch_factor=4, drop_last=True)
BUDGET_S = CFG["TIME_BUDGET_H"] * 3600
def set_lr(s, elapsed):
    f = (s + 1) / CFG["WARMUP_STEPS"] if s < CFG["WARMUP_STEPS"] else \\
        CFG["MIN_LR_FRAC"] + (1 - CFG["MIN_LR_FRAC"]) * 0.5 * (1 + math.cos(math.pi * min(1.0, elapsed / BUDGET_S)))
    for g in opt.param_groups: g["lr"] = g["peak"] * f

@torch.no_grad()
def score_positions(positions, bs=32):
    \"\"\"One forward pass per position: win chance for every listed move.\"\"\"
    net.eval(); out = []
    for k in range(0, len(positions), bs):
        items = []
        for fen, mvs in positions[k:k + bs]:
            b = chess.Board(fen)
            items.append((encode_board(b), [move_feat(b, chess.Move.from_uci(m)) for m, _ in mvs],
                          np.zeros((len(mvs), N_BINS), np.float32), np.zeros(len(mvs), np.float32)))
        ids, att, mv, mmask, _, _ = [t.to(device) for t in collate(items)]
        with torch.autocast("cuda", dtype=torch.float16):
            v = expected_wp(net(ids, att, mv)).cpu().numpy()
        out += [v[j, :len(positions[k + j][1])] for j in range(len(items))]
    net.train(); return out

def top_move_acc(positions):
    preds = score_positions(positions)
    return float(np.mean([np.argmax(p) == np.argmax([w for _, w in mvs]) for p, (_, mvs) in zip(preds, positions)]))

@torch.no_grad()
def evaluate():
    net.eval(); tot = err = n = 0.0
    for ids, att, mv, mmask, tgt, wp in DataLoader(val_ds, batch_size=256, collate_fn=collate, num_workers=2):
        ids, att, mv, mmask, tgt, wp = [t.to(device) for t in (ids, att, mv, mmask, tgt, wp)]
        with torch.autocast("cuda", dtype=torch.float16):
            logits = net(ids, att, mv)
        c = mmask.sum().item()
        tot += loss_fn(logits, mmask, tgt).item() * c; n += c
        err += ((expected_wp(logits) - wp).abs() * mmask).sum().item()
    net.train()
    return {"val_loss": round(tot / n, 4), "val_wp_mae": round(err / n, 4),
            "top_move_acc": round(top_move_acc(val_positions), 4),
            "top_move_acc_big": round(top_move_acc(val_positions_big), 4)}

MODEL_CARD = '''---
license: apache-2.0
base_model: convaiinnovations/laya
tags: [chess, laya]
---
# LayaChess v3

Laya encoder (warm-started from `{init}`) + move-query head: one forward pass per position scores every legal move
(win chance over {n} levels). Trained on DeepMind ChessBench action values.

- checkpoint `{tag}`: step {step}, {consumed:,} training examples
- latest eval: `{last_eval}`
- code: github.com/devroopsaha744/LayaChess (notebooks/build_v3_notebook.py)
'''

def save_ckpt(tag, final=False):
    tmp = Path(CKPT_DIR + ".tmp"); shutil.rmtree(tmp, ignore_errors=True); tmp.mkdir(parents=True)
    save_file({k: v.detach().half().cpu().contiguous() for k, v in model.state_dict().items()}, str(tmp / "model.safetensors"))
    shutil.copytree(MODEL_DIR / "encoder", tmp / "encoder", ignore=shutil.ignore_patterns("*.safetensors", "*.bin"))
    shutil.copytree(MODEL_DIR / "tokenizer", tmp / "tokenizer")
    shutil.copy(MODEL_DIR / "rl_agent_config.json", tmp / "rl_agent_config.json")
    if CFG["SAVE_OPTIMIZER"] and final:
        torch.save({"opt": opt.state_dict(), "scaler": scaler.state_dict()}, tmp / "train_state.pt")
    last_eval = next((h for h in reversed(history) if "top_move_acc" in h), {})
    json.dump({"format": "laya-chess-v3", "tag": tag, "final": final, "step": step, "consumed": consumed,
               "n_bins": N_BINS, "head": {"layers": CFG["HEAD_LAYERS"], "heads": CFG["HEAD_HEADS"], "ff": CFG["HEAD_FF"]},
               "state_format": "piece lists v2 (serialize_state)", "init": CFG["INIT"], "cfg": CFG,
               "history": history[-80:]}, open(tmp / "chess_meta.json", "w"), indent=1)
    (tmp / "README.md").write_text(MODEL_CARD.format(init=CFG["V2_REPO"] if CFG["INIT"] == "v2" else CFG["BASE_REPO"],
                                   n=N_BINS, tag=tag, step=step, consumed=consumed, last_eval=last_eval))
    shutil.rmtree(CKPT_DIR, ignore_errors=True); tmp.rename(CKPT_DIR)
    print(f"[ckpt] saved {tag} at step {step}")
    if HF_REPO:
        try:
            api.upload_folder(folder_path=CKPT_DIR, repo_id=HF_REPO,
                              commit_message=f"{tag}: step {step}, {consumed} examples, eval {last_eval}")
            print(f"[hf] uploaded {tag} -> https://huggingface.co/{HF_REPO}")
        except Exception as e:
            print("[hf] upload failed (training continues):", e)

history = (meta or {}).get("history", [])
m = evaluate(); m["step"] = step; history.append(m); print("baseline (before training):", m)
t0 = last_ckpt = time.time(); run_loss = 0.0; micro = log_micro = session_step = 0
net.train()
for ids, att, mv, mmask, tgt, wp in train_dl:
    ids, att, mv, mmask, tgt = [t.to(device, non_blocking=True) for t in (ids, att, mv, mmask, tgt)]
    with torch.autocast("cuda", dtype=torch.float16):
        logits = net(ids, att, mv)
    loss = loss_fn(logits, mmask, tgt) / CFG["GRAD_ACCUM"]
    scaler.scale(loss).backward()
    run_loss += loss.item(); micro += 1; log_micro += 1; consumed += len(wp)
    if micro % CFG["GRAD_ACCUM"]: continue
    set_lr(session_step, time.time() - t0)
    scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
    step += 1; session_step += 1
    if step % CFG["LOG_EVERY_STEPS"] == 0:
        el = time.time() - t0; avg = run_loss * CFG["GRAD_ACCUM"] / max(1, log_micro)
        print(f"step {step} loss {avg:.4f} lr {opt.param_groups[0]['lr']:.2e} | {consumed} recs | "
              f"{session_step * CFG['BATCH_SIZE'] * CFG['GRAD_ACCUM'] / el * 3600 / 1e3:.0f}k recs/h | {el / 3600:.2f} h", flush=True)
        history.append({"step": step, "loss": round(avg, 4)}); run_loss = 0.0; log_micro = 0
    if step % CFG["EVAL_EVERY_STEPS"] == 0:
        m = evaluate(); m["step"] = step; history.append(m); print("EVAL", m, flush=True)
    if time.time() - last_ckpt > CFG["CKPT_EVERY_MIN"] * 60:
        save_ckpt(f"step{step}"); last_ckpt = time.time()
    if time.time() - t0 > BUDGET_S:
        print("time budget reached"); break

m = evaluate(); m["step"] = step; history.append(m); print("FINAL EVAL", m)
save_ckpt("final", final=True)
""")

md("## 6. Speed check: one pass per position")
code("""
pos = [(chess.STARTING_FEN, [(m.uci(), 0.0) for m in chess.Board().legal_moves])] * 64
torch.cuda.synchronize(); t = time.time(); score_positions(pos, bs=64); torch.cuda.synchronize()
print(f"{64 / (time.time() - t):.0f} positions/s (all legal moves each) on {torch.cuda.device_count()} GPU(s)")
b = chess.Board("r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4")
mvs = list(b.legal_moves); v = score_positions([(b.fen(), [(m.uci(), 0.0) for m in mvs])])[0]
print("Scholar's mate position, top 5:", [(b.san(mvs[i]), round(float(v[i]), 3)) for i in np.argsort(-v)[:5]])
""")

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
      "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
OUT.write_text(json.dumps(nb, indent=1))
print("wrote", OUT)
