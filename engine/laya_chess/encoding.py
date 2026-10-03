"""Position + move -> Laya `score` question. Must match the training notebook exactly."""
import chess
import numpy as np

NAMES = {chess.PAWN: "pawn", chess.KNIGHT: "knight", chess.BISHOP: "bishop",
         chess.ROOK: "rook", chess.QUEEN: "queen", chess.KING: "king"}
ORDER = [chess.KING, chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT, chess.PAWN]

DEFAULT_N_LEVELS = 10
DEFAULT_INS_TMPL = "{side} plays {san} ({piece} {frm}-{to}{extra}). Win chance for {side}?"


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


class MoveEncoder:
    """Builds the token ids for "how good is `move` in `board`?" exactly as in training.

    Level labels and the instruction template come from the checkpoint's chess_meta.json, so the
    engine always asks the question the model was trained on.
    """

    def __init__(self, tokenizer, laya_cfg, meta=None):
        from laya.common import build_sequence
        meta = meta or {}
        if meta.get("state_format", "piece lists v2") != "piece lists v2":
            raise ValueError(f"checkpoint uses state format {meta['state_format']!r}, engine supports 'piece lists v2'")
        self.n_levels = int(meta.get("n_levels", DEFAULT_N_LEVELS))
        edges = np.linspace(0, 1, self.n_levels + 1)
        self.centers = ((edges[:-1] + edges[1:]) / 2).astype(np.float32)
        self.crit = meta.get("crit") or [f"{edges[i]*100:.0f}-{edges[i+1]*100:.0f}%" for i in range(self.n_levels)]
        self.ins_tmpl = meta.get("ins_template") or DEFAULT_INS_TMPL
        self._build_sequence = build_sequence
        self.tokenizer, self.max_len, self.head_max_len = tokenizer, laya_cfg["max_len"], laya_cfg["head_max_len"]

    def question(self, board, move):
        piece = board.piece_at(move.from_square)
        extra = ""
        if board.is_capture(move):
            cap = board.piece_at(move.to_square)
            extra += f", takes {NAMES[cap.piece_type] if cap else 'pawn'}"
        if move.promotion:
            extra += f", promotes to {NAMES[move.promotion]}"
        if board.gives_check(move):
            extra += ", check"
        side = "white" if board.turn else "black"
        ins = self.ins_tmpl.format(side=side, san=board.san(move), piece=NAMES[piece.piece_type],
                                   frm=chess.square_name(move.from_square), to=chess.square_name(move.to_square),
                                   extra=extra)
        return {"t": "score", "ins": ins, "crit": self.crit}

    def encode(self, board, move, state=None):
        state = state if state is not None else board_state(board)
        ids, markers = self._build_sequence(self.tokenizer, state, self.question(board, move),
                                            self.max_len, self.head_max_len)[:2]
        return ids, markers
