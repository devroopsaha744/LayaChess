"""Load a Laya chess checkpoint and score every legal move of a position."""
import json
import os
import time
from collections import OrderedDict
from pathlib import Path

import chess
import numpy as np
import torch

from .encoding import MoveEncoder, board_state

DEFAULT_CHECKPOINT = "datafreak/laya-chess"
# only what inference needs (skips the ~3.4 GB optimizer state and other Laya variants)
ALLOW = ["model.safetensors", "rl_agent_config.json", "chess_meta.json", "encoder/*", "tokenizer/*"]


def pick_device(name="auto"):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def position_key(board):
    # position + side + castling + en passant; move counters don't change the encoding
    return board._transposition_key()


class LayaChessModel:
    """`evaluate(boards)` -> for each board: (legal moves, win chance for the side to move after each move).

    `checkpoint` is a Hugging Face repo id (private repos use your `hf auth login` token) or a local folder.
    """

    def __init__(self, checkpoint=DEFAULT_CHECKPOINT, revision=None, device="auto", dtype=None,
                 batch_size=96, cache_size=200_000, verbose=True):
        from huggingface_hub import snapshot_download
        from laya.agent import _fix_tokenizer_config
        from laya.common import QTYPES, build_model
        from safetensors.torch import load_file
        from transformers import AutoTokenizer

        t0 = time.time()
        path = Path(checkpoint) if os.path.isdir(checkpoint) else Path(
            snapshot_download(checkpoint, revision=revision, allow_patterns=ALLOW))
        _fix_tokenizer_config(str(path))
        laya_cfg = json.loads((path / "rl_agent_config.json").read_text())
        meta_file = path / "chess_meta.json"
        self.meta = json.loads(meta_file.read_text()) if meta_file.exists() else {}
        self.checkpoint = checkpoint

        self.tokenizer = AutoTokenizer.from_pretrained(path / "tokenizer")
        self.encoder = MoveEncoder(self.tokenizer, laya_cfg, self.meta)
        model = build_model(laya_cfg, encoder_dir=str(path / "encoder"), pretrained=False)
        for m in model.modules():  # no torch.compile inside ModernBERT (slow first call, breaks on MPS)
            if hasattr(m, "config") and hasattr(m.config, "reference_compile"):
                m.config.reference_compile = False
        model.load_state_dict({k: v.float() for k, v in load_file(str(path / "model.safetensors")).items()}, strict=True)

        self.device = pick_device(device)
        self.dtype = dtype or (torch.float16 if self.device.type in ("cuda", "mps") else torch.float32)
        self.model = model.to(self.device, self.dtype).eval()
        self.qtype = QTYPES["score"]
        self.pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        self.centers = torch.tensor(self.encoder.centers, device=self.device)
        self.batch_size = batch_size
        self.cache, self.cache_size = OrderedDict(), cache_size
        self.n_positions = self.n_sequences = 0
        if verbose:
            tag = self.meta.get("tag", "base Laya (untrained for chess)")
            print(f"[laya-chess] {checkpoint} ({tag}, step {self.meta.get('step', 0)}, "
                  f"{self.meta.get('consumed', 0):,} examples) on {self.device}/{str(self.dtype).split('.')[-1]} "
                  f"in {time.time() - t0:.1f}s", flush=True)

    @torch.no_grad()
    def _forward(self, items):
        L = max(len(ids) for ids, _ in items)
        M = max(len(mk) for _, mk in items)
        B = len(items)
        ids = torch.full((B, L), self.pad_id, dtype=torch.long)
        att = torch.zeros((B, L), dtype=torch.long)
        pos = torch.zeros((B, M), dtype=torch.long)
        mask = torch.zeros((B, M), dtype=torch.bool)
        for j, (i, mk) in enumerate(items):
            ids[j, :len(i)] = torch.tensor(i)
            att[j, :len(i)] = 1
            pos[j, :len(mk)] = torch.tensor(mk)
            mask[j, :len(mk)] = True
        qt = torch.full((B,), self.qtype, dtype=torch.long)
        ids, att, pos, mask, qt = (t.to(self.device) for t in (ids, att, pos, mask, qt))
        logits, _ = self.model(ids, att, pos, mask, qt)
        p = torch.softmax(logits.float().masked_fill(~mask, -1e4), -1)[:, :len(self.centers)]
        return (p * self.centers).sum(-1).cpu().numpy()

    def evaluate(self, boards):
        """Batch-score all legal moves of several positions in as few forward passes as possible."""
        out, todo = [None] * len(boards), []
        for bi, board in enumerate(boards):
            key = position_key(board)
            if key in self.cache:
                self.cache.move_to_end(key)
                out[bi] = self.cache[key]
            else:
                todo.append((bi, key, board, list(board.legal_moves)))
        items, owners = [], []
        for bi, key, board, moves in todo:
            state = board_state(board)
            for mv in moves:
                items.append(self.encoder.encode(board, mv, state))
                owners.append(bi)
        values = np.concatenate([self._forward(items[k:k + self.batch_size])
                                 for k in range(0, len(items), self.batch_size)]) if items else np.zeros(0)
        k = 0
        for bi, key, board, moves in todo:
            q = values[k:k + len(moves)].astype(np.float32); k += len(moves)
            out[bi] = (moves, q)
            self.cache[key] = out[bi]
            if len(self.cache) > self.cache_size:
                self.cache.popitem(last=False)
        self.n_positions += len(todo); self.n_sequences += len(items)
        return out

    def score_moves(self, board):
        """Moves sorted best-first with the model's win chance for the side to move (no search)."""
        moves, q = self.evaluate([board])[0]
        order = np.argsort(-q)
        return [(moves[i], float(q[i])) for i in order]
