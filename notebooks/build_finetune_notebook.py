import json, sys

cells = []
def md(s): cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})
def code(s): cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n")})

md("""
# Fine-tune Laya to play chess (ChessBench action-values)

Every training example: **(position, legal move) → how good is this move?**, asked as a Laya `score` question
with `N_LEVELS` win-probability levels. The engine plays the move with the highest expected win probability.

Data: DeepMind ChessBench `action_value` shards (Stockfish win prob for every legal move) — attach the `chessbench-av` dataset.

**How to run**
1. Settings: Accelerator **GPU T4 x2**, Internet **ON**.
2. Hugging Face upload: **Add-ons → Secrets → Add secret** named `HF_TOKEN` (a *write* token from huggingface.co/settings/tokens), and switch it on for this notebook. Without it training still runs, it just doesn't upload.
3. First run with `QUICK_TEST = True` (~30 min): check sanity cells, `k recs/h` and loss going down.
4. Then set `QUICK_TEST = False` and use **Save Version → Save & Run All (Commit)** (~11 h).
5. To continue training: set `RESUME_FROM = "hf"` (pulls the latest checkpoint from your Hugging Face repo) and run again.
""")

code("""
# ===== Settings =====
import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"   # must be set before CUDA starts
QUICK_TEST = True

CFG = dict(
    MODEL_ID="convaiinnovations/laya",
    N_TRAIN_SHARDS=2,            # each ChessBench shard ~1.5 GB, ~16.7M (fen, move, win_prob); used one after another
    MAX_TRAIN_RECORDS=None,      # None = all records of all shards (~33.5M); each run is limited by TIME_BUDGET_H
    N_VAL_RECORDS=4_000,         # for val loss / win-prob error
    N_VAL_POSITIONS=300,         # for top-move accuracy (scores every legal move)
    N_LEVELS=10,                 # win-probability levels (10% each)
    LABEL_SIGMA=0.6,             # soft-label width, in levels
    BATCH_SIZE=32,               # per step, total over both GPUs
    GRAD_ACCUM=2,                # effective batch = 64
    LR_ENCODER=4e-5,
    LR_HEAD=2e-4,
    WARMUP_STEPS=150,
    MIN_LR_FRAC=0.05,            # cosine decays to 5% of peak by the end of TIME_BUDGET_H
    WEIGHT_DECAY=0.01,
    NUM_WORKERS=4,
    TIME_BUDGET_H=11.0,          # stop + save before Kaggle's 12 h limit
    CKPT_EVERY_MIN=45,
    EVAL_EVERY_STEPS=2000,
    LOG_EVERY_STEPS=50,
    SAVE_OPTIMIZER=True,         # ~3.4 GB extra, needed for exact resume
    RESUME_FROM=None,            # "hf" = latest checkpoint on Hugging Face | "auto" = attached previous run output | a path
    RESUME_LR_SCALE=0.5,         # later sessions restart the cosine at half the peak LR
    HF_REPO_NAME="laya-chess",   # uploaded to <your HF username>/laya-chess (private)
    SEED=0,
)
if QUICK_TEST:
    CFG.update(N_VAL_RECORDS=1_000, N_VAL_POSITIONS=100, TIME_BUDGET_H=0.4, CKPT_EVERY_MIN=15,
               EVAL_EVERY_STEPS=300, SAVE_OPTIMIZER=False, HF_REPO_NAME="laya-chess-quicktest")

DATA_DIR = "/tmp/chessbench"
BASE_DIR = "/tmp/laya_base"
CKPT_DIR = "/kaggle/working/ckpt"

# Hugging Face token from Kaggle Secrets (never paste tokens into the notebook itself)
HF_TOKEN = None
try:
    from kaggle_secrets import UserSecretsClient
    HF_TOKEN = UserSecretsClient().get_secret("HF_TOKEN")
    os.environ["HF_TOKEN"] = HF_TOKEN          # also speeds up the Laya download
    print("HF_TOKEN secret found -> checkpoints will be uploaded")
except Exception as e:
    print("no HF_TOKEN secret -> no Hugging Face upload:", type(e).__name__)
print(CFG)
""")

code("""
!pip -q install python-chess zstandard
!pip -q install "git+https://github.com/NandhaKishorM/laya.git"
!nvidia-smi --query-gpu=name,memory.total --format=csv
""")

md("## 1. ChessBench data\nUses the Kaggle dataset if attached (any folder under `/kaggle/input` containing `test_action_value.bag`), otherwise downloads.")
code("""
import os, subprocess, glob
found = glob.glob("/kaggle/input/**/test_action_value.bag", recursive=True)
if found:
    DATA_DIR = os.path.dirname(found[0])
    print("using attached Kaggle dataset:", DATA_DIR, sorted(os.listdir(DATA_DIR)))
    n_have = len(glob.glob(f"{DATA_DIR}/train_action_value_*.bag"))
    assert n_have >= CFG["N_TRAIN_SHARDS"], f"dataset has {n_have} train shards, CFG wants {CFG['N_TRAIN_SHARDS']}"
os.makedirs(DATA_DIR, exist_ok=True)
BASE = "https://storage.googleapis.com/searchless_chess/data"
urls = {"test_action_value.bag": f"{BASE}/test/action_value_data.bag"}
for i in range(CFG["N_TRAIN_SHARDS"]):
    urls[f"train_action_value_{i:05d}.bag"] = f"{BASE}/train/action_value-{i:05d}-of-02148_data.bag"
for name, url in urls.items():
    out = os.path.join(DATA_DIR, name)
    if not os.path.exists(out):
        subprocess.run(["wget", "-q", "-O", out + ".part", url], check=True)
        os.rename(out + ".part", out)
    print(f"{name}: {os.path.getsize(out)/1e6:.0f} MB")
""")

md("## 2. Read `.bag` files (no apache_beam needed)\nBag = records back to back, then an index of int64 end offsets. Record = Beam TupleCoder(fen str, move str, win_prob float).")
code("""
import mmap, struct

class BagReader:
    def __init__(self, path):
        self.path = path
        self._open()
    def _open(self):
        f = open(self.path, "rb")
        self.mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        size = len(self.mm)
        index_start = struct.unpack("<q", self.mm[size - 8:])[0]
        # bagz: the last index entry (= end of last record) doubles as the index start pointer
        n_with = (size - index_start) // 8
        last = struct.unpack_from("<q", self.mm, size - 8)[0]
        self.index_start, self.n = index_start, n_with
        assert last == index_start, "unexpected bag layout"
    def __len__(self): return self.n
    def __getitem__(self, i):
        end = struct.unpack_from("<q", self.mm, self.index_start + 8 * i)[0]
        start = 0 if i == 0 else struct.unpack_from("<q", self.mm, self.index_start + 8 * (i - 1))[0]
        return self.mm[start:end]
    def __getstate__(self): return {"path": self.path}      # re-open in DataLoader workers
    def __setstate__(self, s): self.path = s["path"]; self._open()

def _varint(b, p):
    r = s = 0
    while True:
        x = b[p]; p += 1
        r |= (x & 0x7F) << s
        if x < 0x80: return r, p
        s += 7

def decode_av(b):
    n, p = _varint(b, 0); fen = b[p:p + n].decode(); p += n
    n, p = _varint(b, p); move = b[p:p + n].decode(); p += n
    (wp,) = struct.unpack(">d", b[p:p + 8])
    return fen, move, wp

import chess
test_bag = BagReader(f"{DATA_DIR}/test_action_value.bag")
train_bags = [BagReader(f"{DATA_DIR}/train_action_value_{i:05d}.bag") for i in range(CFG["N_TRAIN_SHARDS"])]
print("test records:", len(test_bag), "| train records:", [len(b) for b in train_bags])

# sanity: records decode, moves legal, win probs in [0, 1]
for bag in [test_bag] + train_bags:
    for i in [0, 1, len(bag) // 2, len(bag) - 1]:
        fen, mv, wp = decode_av(bag[i])
        assert chess.Move.from_uci(mv) in chess.Board(fen).legal_moves and 0.0 <= wp <= 1.0, (fen, mv, wp)
print("example:", decode_av(train_bags[0][0]))
""")

md("## 3. Load Laya (+ Hugging Face repo)")
code("""
import json, glob, shutil, math, random, time
from pathlib import Path
import numpy as np, torch
from huggingface_hub import snapshot_download, HfApi
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer
from laya.agent import _fix_tokenizer_config
from laya.common import QTYPES, build_model, build_sequence

random.seed(CFG["SEED"]); np.random.seed(CFG["SEED"]); torch.manual_seed(CFG["SEED"])

snapshot_download(CFG["MODEL_ID"], local_dir=BASE_DIR)
cfg_files = sorted(glob.glob(f"{BASE_DIR}/**/rl_agent_config.json", recursive=True), key=len)
print("configs found:", cfg_files)
MODEL_DIR = Path(cfg_files[0]).parent          # shortest path = main English model
_fix_tokenizer_config(str(MODEL_DIR))
with open(MODEL_DIR / "rl_agent_config.json") as f:
    lcfg = json.load(f)
print("laya cfg:", {k: lcfg[k] for k in lcfg if k in ("max_len", "head_max_len")})

HF_REPO = None
if HF_TOKEN:
    try:
        hf_api = HfApi(token=HF_TOKEN)
        HF_REPO = f"{hf_api.whoami()['name']}/{CFG['HF_REPO_NAME']}"
        hf_api.create_repo(HF_REPO, private=True, exist_ok=True)
        print("Hugging Face repo:", f"https://huggingface.co/{HF_REPO}")
    except Exception as e:
        print("Hugging Face setup failed -> no upload:", e); HF_REPO = None

if CFG["RESUME_FROM"] == "hf":       # latest checkpoint from the Hugging Face repo
    assert HF_REPO, "RESUME_FROM='hf' needs the HF_TOKEN secret attached"
    CFG["RESUME_FROM"] = snapshot_download(HF_REPO, token=HF_TOKEN, local_dir="/tmp/resume_ckpt")
elif CFG["RESUME_FROM"] == "auto":   # ckpt/ from an attached previous run's output
    found = sorted(glob.glob("/kaggle/input/**/ckpt/chess_meta.json", recursive=True))
    assert found, "RESUME_FROM='auto' but no ckpt/ found -> Add Input: the previous run's notebook output"
    CFG["RESUME_FROM"] = os.path.dirname(found[0])
if CFG["RESUME_FROM"]:
    meta = json.load(open(Path(CFG["RESUME_FROM"]) / "chess_meta.json"))
    print("resuming from", CFG["RESUME_FROM"], "| tag", meta["tag"], "| step", meta["step"], "| examples", meta["consumed"],
          "| optimizer state:", (Path(CFG["RESUME_FROM"]) / "train_state.pt").exists())

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR / "tokenizer")
model = build_model(lcfg, encoder_dir=MODEL_DIR / "encoder")
weights_dir = Path(CFG["RESUME_FROM"]) if CFG["RESUME_FROM"] else MODEL_DIR
model.load_state_dict({k: v.float() for k, v in load_file(weights_dir / "model.safetensors").items()})
print("loaded weights from", weights_dir, "| params:", sum(p.numel() for p in model.parameters()) / 1e6, "M")
""")

md("## 4. Position + move → Laya `score` question\nCompact text: piece lists instead of an 8×8 grid, short level labels → ~half the tokens of v1, so ~2× more examples per hour.")
code("""
N_LEVELS = CFG["N_LEVELS"]
EDGES = np.linspace(0, 1, N_LEVELS + 1)
CENTERS = (EDGES[:-1] + EDGES[1:]) / 2
CRIT = [f"{EDGES[i]*100:.0f}-{EDGES[i+1]*100:.0f}%" for i in range(N_LEVELS)]
INS_TMPL = "{side} plays {san} ({piece} {frm}-{to}{extra}). Win chance for {side}?"
NAMES = {chess.PAWN: "pawn", chess.KNIGHT: "knight", chess.BISHOP: "bishop",
         chess.ROOK: "rook", chess.QUEEN: "queen", chess.KING: "king"}
ORDER = [chess.KING, chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT, chess.PAWN]

def piece_list(board, color):
    out = []
    for pt in ORDER:
        for sq in sorted(board.pieces(pt, color)):
            out.append(chess.piece_symbol(pt).upper() + chess.square_name(sq))
    return " ".join(out)

def board_state(board):
    return {
        "to_move": "white" if board.turn else "black",
        "white": piece_list(board, chess.WHITE),
        "black": piece_list(board, chess.BLACK),
        "castling": board.castling_xfen() if board.castling_rights else "-",
        "en_passant": chess.square_name(board.ep_square) if board.has_legal_en_passant() else "-",
    }

def question(board, move):
    piece = board.piece_at(move.from_square)
    extra = ""
    if board.is_capture(move):
        cap = board.piece_at(move.to_square)
        extra += f", takes {NAMES[cap.piece_type] if cap else 'pawn'}"
    if move.promotion: extra += f", promotes to {NAMES[move.promotion]}"
    if board.gives_check(move): extra += ", check"
    side = "white" if board.turn else "black"
    ins = INS_TMPL.format(side=side, san=board.san(move), piece=NAMES[piece.piece_type],
                          frm=chess.square_name(move.from_square), to=chess.square_name(move.to_square), extra=extra)
    return {"t": "score", "ins": ins, "crit": CRIT}

def encode(fen, uci):
    board = chess.Board(fen)
    ids, markers = build_sequence(tokenizer, board_state(board), question(board, chess.Move.from_uci(uci)),
                                  lcfg["max_len"], lcfg["head_max_len"])[:2]
    return ids, markers

def soft_target(wp):
    x = (wp * N_LEVELS) - 0.5                      # position in level units
    t = np.exp(-0.5 * ((np.arange(N_LEVELS) - x) / CFG["LABEL_SIGMA"]) ** 2)
    return (t / t.sum()).astype(np.float32)

# sanity: look at one encoded example
fen, mv, wp = decode_av(train_bags[0][0])
ids, markers = encode(fen, mv)
print("tokens:", len(ids), "| markers:", len(markers), "| win prob:", round(wp, 3))
assert len(markers) == N_LEVELS, "level options got truncated -> lower N_LEVELS or shorten text"
print(tokenizer.decode(ids))
lens = [len(encode(*decode_av(train_bags[0][i])[:2])[0]) for i in range(0, 20000, 250)]
print("token length min/mean/max:", min(lens), int(np.mean(lens)), max(lens))
""")

md("## 5. Datasets")
code("""
from torch.utils.data import Dataset, DataLoader

class AVDataset(Dataset):
    def __init__(self, refs, bags):           # refs: array of (bag_idx, rec_idx)
        self.refs, self.bags = refs, bags
    def __len__(self): return len(self.refs)
    def __getitem__(self, i):
        b, r = self.refs[i]
        fen, mv, wp = decode_av(self.bags[b][r])
        ids, markers = encode(fen, mv)
        return {"ids": ids, "markers": markers, "target": soft_target(wp), "wp": wp}

PAD = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
def collate(items):
    L = max(len(x["ids"]) for x in items); M = max(len(x["markers"]) for x in items)
    B = len(items)
    ids = torch.full((B, L), PAD, dtype=torch.long); att = torch.zeros((B, L), dtype=torch.long)
    pos = torch.zeros((B, M), dtype=torch.long); mask = torch.zeros((B, M), dtype=torch.bool)
    tgt = torch.zeros((B, M), dtype=torch.float32)
    for j, x in enumerate(items):
        n, m = len(x["ids"]), len(x["markers"])
        ids[j, :n] = torch.tensor(x["ids"]); att[j, :n] = 1
        pos[j, :m] = torch.tensor(x["markers"]); mask[j, :m] = True
        tgt[j, :m] = torch.from_numpy(x["target"][:m])
    qtype = torch.full((B,), QTYPES["score"], dtype=torch.long)
    wp = torch.tensor([x["wp"] for x in items], dtype=torch.float32)
    return ids, att, pos, mask, tgt, qtype, wp

# Fixed training order across runs: file 1 shuffled, then file 2 shuffled, then file 3 ...
# Resumed runs skip the first `consumed` records, so nothing repeats until every file is used.
# (File 1's order + the validation set are identical to the first run, so its checkpoint resumes correctly.)
rng = np.random.default_rng(CFG["SEED"])
def shard_refs(k):
    return np.stack([np.full(len(train_bags[k]), k, dtype=np.int32), np.arange(len(train_bags[k]), dtype=np.int32)], 1)
parts = [shard_refs(0)]
rng.shuffle(parts[0])

# validation from the separate ChessBench test split (different games)
val_idx = rng.choice(len(test_bag), CFG["N_VAL_RECORDS"], replace=False)

for k in range(1, len(train_bags)):
    p = shard_refs(k); np.random.default_rng(CFG["SEED"] + k).shuffle(p); parts.append(p)
train_refs = np.concatenate(parts); del parts
if CFG["MAX_TRAIN_RECORDS"]:
    train_refs = train_refs[:CFG["MAX_TRAIN_RECORDS"]]
val_ds = AVDataset(np.stack([np.zeros_like(val_idx), val_idx], 1), [test_bag])

# positions for top-move accuracy: group the test set by FEN, keep positions with >= 2 moves
by_fen = {}
for i in range(len(test_bag)):
    fen, mv, wp = decode_av(test_bag[i])
    by_fen.setdefault(fen, []).append((mv, wp))
val_positions = [(f, mv) for f, mv in by_fen.items() if len(mv) >= 2]
random.Random(CFG["SEED"]).shuffle(val_positions)
val_positions = val_positions[:CFG["N_VAL_POSITIONS"]]
del by_fen
print("train records:", len(train_refs), "| val records:", len(val_ds), "| val positions:", len(val_positions))
""")

md("## 6. Train (fp16, 2× T4, time-based LR schedule, checkpoints → Hugging Face)")
code("""
device = "cuda"
# ModernBERT's built-in torch.compile path breaks DataParallel replicas -> turn it off
for m in model.modules():
    if hasattr(m, "config") and hasattr(m.config, "reference_compile"):
        m.config.reference_compile = False

# DataParallel replicas have no registered parameters, so HF's `self.dtype` / `self.device`
# raise StopIteration. Fall back to the tensors stored on the replica's submodules.
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

def enable_grad_ckpt():
    for m in model.modules():
        if isinstance(m, PreTrainedModel):
            m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            print("gradient checkpointing on:", type(m).__name__)

model.float().to(device)
CENTERS_T = torch.tensor(CENTERS, dtype=torch.float32, device=device)

enc_params = [p for n, p in model.named_parameters() if n.startswith("encoder")]
head_params = [p for n, p in model.named_parameters() if not n.startswith("encoder")]
print(f"encoder params {sum(p.numel() for p in enc_params)/1e6:.1f}M | head params {sum(p.numel() for p in head_params)/1e6:.1f}M")
lr_scale = CFG["RESUME_LR_SCALE"] if CFG["RESUME_FROM"] else 1.0
groups = [{"params": enc_params, "lr": CFG["LR_ENCODER"] * lr_scale, "peak": CFG["LR_ENCODER"] * lr_scale}]
if head_params: groups.append({"params": head_params, "lr": CFG["LR_HEAD"] * lr_scale, "peak": CFG["LR_HEAD"] * lr_scale})
opt = torch.optim.AdamW(groups, weight_decay=CFG["WEIGHT_DECAY"])
scaler = torch.amp.GradScaler("cuda")

def ce_loss(logits, mask, tgt):
    logp = torch.log_softmax(logits.float().masked_fill(~mask, -1e4), -1)
    return -(tgt * logp).sum(-1).mean()

# --- pick the fastest setup that fits: 2 GPUs if DataParallel works, checkpointing only if needed ---
probe_items = sorted((val_ds[i] for i in range(min(256, len(val_ds)))), key=lambda x: -len(x["ids"]))
def probe(net_, bs):
    # forward+backward on the longest examples, plus one optimizer step at lr=0 so AdamW state is allocated too
    b = [t.to(device) for t in collate(probe_items[:bs])]
    with torch.autocast("cuda", dtype=torch.float16):
        logits, _ = net_(b[0], b[1], b[2], b[3], b[5])
    ce_loss(logits, b[3], b[4]).backward()
    for g in opt.param_groups: g["lr"] = 0.0
    opt.step(); opt.zero_grad(set_to_none=True)
    for g in opt.param_groups: g["lr"] = g["peak"]
    torch.cuda.synchronize()

net = model
if torch.cuda.device_count() > 1:
    try:
        dp = torch.nn.DataParallel(model)
        b = [t.to(device) for t in collate(probe_items[:2])]
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            dp(b[0], b[1], b[2], b[3], b[5])
        net = dp
        print("using DataParallel on", torch.cuda.device_count(), "GPUs")
    except Exception as e:
        print("DataParallel failed -> single GPU:", type(e).__name__, str(e)[:200])
        CFG["BATCH_SIZE"] //= 2; CFG["GRAD_ACCUM"] *= 2

grad_ckpt = False
while True:
    try:
        probe(net, CFG["BATCH_SIZE"])
        break
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:   # DataParallel may re-raise OOM as RuntimeError
        if "out of memory" not in str(e).lower(): raise
        opt.zero_grad(set_to_none=True); torch.cuda.empty_cache()
        if not grad_ckpt:
            print(f"OOM at batch {CFG['BATCH_SIZE']} -> turning on gradient checkpointing")
            enable_grad_ckpt(); grad_ckpt = True
        elif CFG["BATCH_SIZE"] > 4:
            CFG["BATCH_SIZE"] //= 2; CFG["GRAD_ACCUM"] *= 2
            print(f"still OOM -> batch {CFG['BATCH_SIZE']} x accum {CFG['GRAD_ACCUM']}")
        else:
            raise
opt.state.clear()   # drop the probe's AdamW moments; real training starts from a clean optimizer
torch.cuda.empty_cache()
print(f"setup: batch {CFG['BATCH_SIZE']} x accum {CFG['GRAD_ACCUM']} | grad checkpointing {grad_ckpt} | "
      f"GPU0 peak {torch.cuda.max_memory_allocated(0) / 2**30:.1f} GB")

step, consumed = 0, 0
if CFG["RESUME_FROM"]:
    step, consumed = meta["step"], meta["consumed"]
    st_path = Path(CFG["RESUME_FROM"]) / "train_state.pt"
    if st_path.exists():
        st = torch.load(st_path, map_location="cpu", weights_only=False)
        if "opt" in st:
            opt.load_state_dict(st["opt"]); scaler.load_state_dict(st["scaler"])
            for g, peak in zip(opt.param_groups, [x["peak"] for x in groups]): g["peak"] = peak
            print("optimizer state restored")
    else:
        print("no train_state.pt (mid-run checkpoint) -> fresh optimizer, warmup handles it")
    print(f"resumed at step {step}, {consumed} records consumed")
if consumed >= len(train_refs):
    print("all records used -> starting a new pass over the data"); consumed = 0

train_ds = AVDataset(train_refs[consumed:], train_bags)
train_dl = DataLoader(train_ds, batch_size=CFG["BATCH_SIZE"], shuffle=False, collate_fn=collate,
                      num_workers=CFG["NUM_WORKERS"], persistent_workers=True, prefetch_factor=4, drop_last=True)

BUDGET_S = CFG["TIME_BUDGET_H"] * 3600
def set_lr(session_step, elapsed):
    # linear warmup, then cosine over the *time* budget, so the LR anneals however fast the GPUs are
    if session_step < CFG["WARMUP_STEPS"]:
        f = (session_step + 1) / CFG["WARMUP_STEPS"]
    else:
        f = CFG["MIN_LR_FRAC"] + (1 - CFG["MIN_LR_FRAC"]) * 0.5 * (1 + math.cos(math.pi * min(1.0, elapsed / BUDGET_S)))
    for g in opt.param_groups: g["lr"] = g["peak"] * f

def expected_wp(logits, mask):
    p = torch.softmax(logits.float().masked_fill(~mask, -1e4), -1)
    return (p[:, :N_LEVELS] * CENTERS_T).sum(-1)

@torch.no_grad()
def score_moves(fen, ucis, bs=64):
    net.eval(); out = []
    for k in range(0, len(ucis), bs):
        items = []
        for u in ucis[k:k + bs]:
            ids, markers = encode(fen, u)
            items.append({"ids": ids, "markers": markers, "target": np.zeros(N_LEVELS, np.float32), "wp": 0.0})
        ids, att, pos, mask, _, qt, _ = [t.to(device) for t in collate(items)]
        with torch.autocast("cuda", dtype=torch.float16):
            logits, _ = net(ids, att, pos, mask, qt)
        out += expected_wp(logits, mask).tolist()
    net.train(); return out

@torch.no_grad()
def evaluate():
    net.eval(); tot, n, err = 0.0, 0, 0.0
    for ids, att, pos, mask, tgt, qt, wp in DataLoader(val_ds, batch_size=64, collate_fn=collate, num_workers=2):
        ids, att, pos, mask, tgt, qt, wp = [t.to(device) for t in (ids, att, pos, mask, tgt, qt, wp)]
        with torch.autocast("cuda", dtype=torch.float16):
            logits, _ = net(ids, att, pos, mask, qt)
        tot += ce_loss(logits, mask, tgt).item() * len(wp); n += len(wp)
        err += (expected_wp(logits, mask) - wp).abs().sum().item()
    hits = 0
    for fen, mvs in val_positions:
        pred = score_moves(fen, [m for m, _ in mvs])
        hits += int(np.argmax(pred) == int(np.argmax([w for _, w in mvs])))
    net.train()
    return {"val_loss": round(tot / n, 4), "val_wp_mae": round(err / n, 4), "top_move_acc": hits / len(val_positions)}

MODEL_CARD = '''---
license: apache-2.0
base_model: convaiinnovations/laya
tags: [chess, laya, decision-model]
---
# Laya fine-tuned for chess

[Laya](https://huggingface.co/convaiinnovations/laya) fine-tuned on DeepMind ChessBench action-values:
for each legal move, a Laya `score` question predicts the side to move's win chance over {n} levels.
The engine plays the move with the highest expected win chance (no search).

- checkpoint: `{tag}` (step {step}, {consumed:,} training examples)
- latest eval: `{last_eval}`
- encoding / levels / prompt template: see `chess_meta.json`
- code: github.com/devroopsaha744/LayaChess
'''

def upload_ckpt(tag, final):
    if not HF_REPO: return
    try:
        last_eval = next((h for h in reversed(history) if "top_move_acc" in h), {})
        (Path(CKPT_DIR) / "README.md").write_text(MODEL_CARD.format(n=N_LEVELS, tag=tag, step=step, consumed=consumed, last_eval=last_eval))
        ignore = [] if final else ["train_state.pt"]           # optimizer state (~3.4 GB) only with the final upload
        hf_api.upload_folder(folder_path=CKPT_DIR, repo_id=HF_REPO, ignore_patterns=ignore,
                             commit_message=f"{tag}: step {step}, {consumed} examples, eval {last_eval}")
        print(f"[hf] uploaded {tag} -> https://huggingface.co/{HF_REPO}")
    except Exception as e:
        print("[hf] upload failed (training continues):", e)

def save_ckpt(tag, final=False):
    tmp = Path(CKPT_DIR + ".tmp"); shutil.rmtree(tmp, ignore_errors=True); tmp.mkdir(parents=True)
    save_file({k: v.detach().half().cpu().contiguous() for k, v in model.state_dict().items()}, str(tmp / "model.safetensors"))
    shutil.copy(MODEL_DIR / "rl_agent_config.json", tmp / "rl_agent_config.json")
    shutil.copytree(MODEL_DIR / "encoder", tmp / "encoder", ignore=shutil.ignore_patterns("*.safetensors", "*.bin"))
    shutil.copytree(MODEL_DIR / "tokenizer", tmp / "tokenizer")
    st = {"step": step, "consumed": consumed}
    if CFG["SAVE_OPTIMIZER"]:
        st.update(opt=opt.state_dict(), scaler=scaler.state_dict())
    torch.save(st, tmp / "train_state.pt")
    json.dump({"tag": tag, "final": final, "step": step, "consumed": consumed, "n_levels": N_LEVELS,
               "crit": CRIT, "ins_template": INS_TMPL, "state_format": "piece lists v2", "cfg": CFG,
               "history": history[-50:]}, open(tmp / "chess_meta.json", "w"), indent=1)
    shutil.rmtree(CKPT_DIR, ignore_errors=True); tmp.rename(CKPT_DIR)
    print(f"[ckpt] saved {tag} at step {step}")
    upload_ckpt(tag, final)

history = []
m = evaluate(); m["step"] = step; history.append(m); print("baseline (before training):", m)
t0 = last_ckpt = time.time(); run_loss = 0.0; micro = 0; session_step = 0; log_micro = 0
net.train()
for ids, att, pos, mask, tgt, qt, wp in train_dl:
    ids, att, pos, mask, tgt, qt = [t.to(device, non_blocking=True) for t in (ids, att, pos, mask, tgt, qt)]
    with torch.autocast("cuda", dtype=torch.float16):
        logits, _ = net(ids, att, pos, mask, qt)
    loss = ce_loss(logits, mask, tgt) / CFG["GRAD_ACCUM"]
    scaler.scale(loss).backward()
    run_loss += loss.item(); micro += 1; consumed += len(wp); log_micro += 1
    if micro % CFG["GRAD_ACCUM"]: continue
    set_lr(session_step, time.time() - t0)
    scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
    step += 1; session_step += 1

    if step % CFG["LOG_EVERY_STEPS"] == 0:
        el = time.time() - t0
        avg = run_loss * CFG["GRAD_ACCUM"] / max(1, log_micro)   # mean over the micro-batches since the last log line
        print(f"step {step} loss {avg:.4f} lr {opt.param_groups[0]['lr']:.2e} | "
              f"{consumed} recs | {session_step * CFG['BATCH_SIZE'] * CFG['GRAD_ACCUM'] / el * 3600 / 1e3:.0f}k recs/h | {el / 3600:.2f} h")
        history.append({"step": step, "loss": round(avg, 4)}); run_loss = 0.0; log_micro = 0
    if step % CFG["EVAL_EVERY_STEPS"] == 0:
        m = evaluate(); m["step"] = step; history.append(m); print("EVAL", m)
    if time.time() - last_ckpt > CFG["CKPT_EVERY_MIN"] * 60:
        save_ckpt(f"step{step}"); last_ckpt = time.time()
    if time.time() - t0 > BUDGET_S:
        print("time budget reached"); break

m = evaluate(); m["step"] = step; history.append(m); print("FINAL EVAL", m)
save_ckpt("final", final=True)
""")

md("## 7. Play: pick a move for any position")
code("""
def pick_move(fen):
    board = chess.Board(fen)
    ucis = [m.uci() for m in board.legal_moves]
    scores = score_moves(fen, ucis)
    best = int(np.argmax(scores))
    return ucis[best], sorted(zip(ucis, [round(s, 3) for s in scores]), key=lambda x: -x[1])[:5]

for fen in [chess.STARTING_FEN,
            "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4",   # Qxf7# available
            "6k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 0 1"]:                                # Rd8# available
    print(fen, "->", pick_move(fen))
""")

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
      "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
json.dump(nb, open(sys.argv[1], "w"), indent=1)
