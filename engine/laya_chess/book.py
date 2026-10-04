"""Opening book: a small built-in book of mainstream lines, or any Polyglot .bin book.

The engine plays a book move while the current game is still on one of the book's lines; once the position leaves the
book (or the book is switched off) the model decides on its own.
"""
import random

import chess
import chess.polyglot

# Mainstream lines in UCI (6-10 plies). Any position reached by following a prefix of a line is "in book".
LINES = {
    "Italian Game": "e2e4 e7e5 g1f3 b8c6 f1c4 f8c5 c2c3 g8f6 d2d3 d7d6",
    "Two Knights": "e2e4 e7e5 g1f3 b8c6 f1c4 g8f6 d2d3 f8e7 e1g1 e8g8",
    "Ruy Lopez": "e2e4 e7e5 g1f3 b8c6 f1b5 a7a6 b5a4 g8f6 e1g1 f8e7",
    "Ruy Lopez, Berlin": "e2e4 e7e5 g1f3 b8c6 f1b5 g8f6 e1g1 f6e4 d2d4 e4d6",
    "Scotch Game": "e2e4 e7e5 g1f3 b8c6 d2d4 e5d4 f3d4 g8f6 d4c6 b7c6",
    "Petrov": "e2e4 e7e5 g1f3 g8f6 f3e5 d7d6 e5f3 f6e4 d2d4 d6d5",
    "Sicilian Najdorf": "e2e4 c7c5 g1f3 d7d6 d2d4 c5d4 f3d4 g8f6 b1c3 a7a6",
    "Sicilian Classical": "e2e4 c7c5 g1f3 b8c6 d2d4 c5d4 f3d4 g8f6 b1c3 d7d6",
    "Sicilian Alapin": "e2e4 c7c5 c2c3 g8f6 e4e5 f6d5 d2d4 c5d4",
    "French Defence": "e2e4 e7e6 d2d4 d7d5 b1c3 g8f6 c1g5 f8e7 e4e5 f6d7",
    "French Advance": "e2e4 e7e6 d2d4 d7d5 e4e5 c7c5 c2c3 b8c6 g1f3 d8b6",
    "Caro-Kann": "e2e4 c7c6 d2d4 d7d5 b1c3 d5e4 c3e4 c8f5 e4g3 f5g6",
    "Caro-Kann Advance": "e2e4 c7c6 d2d4 d7d5 e4e5 c8f5 g1f3 e7e6 f1e2 c6c5",
    "Scandinavian": "e2e4 d7d5 e4d5 d8d5 b1c3 d5a5 d2d4 g8f6 g1f3 c8f5",
    "Pirc": "e2e4 d7d6 d2d4 g8f6 b1c3 g7g6 g1f3 f8g7 f1e2 e8g8",
    "Queen's Gambit Declined": "d2d4 d7d5 c2c4 e7e6 b1c3 g8f6 c1g5 f8e7 e2e3 e8g8",
    "Queen's Gambit Accepted": "d2d4 d7d5 c2c4 d5c4 g1f3 g8f6 e2e3 e7e6 f1c4 c7c5",
    "Slav Defence": "d2d4 d7d5 c2c4 c7c6 g1f3 g8f6 b1c3 d5c4 a2a4 c8f5",
    "London System": "d2d4 d7d5 c1f4 g8f6 e2e3 e7e6 g1f3 c7c5 c2c3 b8c6",
    "King's Indian": "d2d4 g8f6 c2c4 g7g6 b1c3 f8g7 e2e4 d7d6 g1f3 e8g8",
    "Nimzo-Indian": "d2d4 g8f6 c2c4 e7e6 b1c3 f8b4 e2e3 e8g8 f1d3 d7d5",
    "Queen's Indian": "d2d4 g8f6 c2c4 e7e6 g1f3 b7b6 g2g3 c8b7 f1g2 f8e7",
    "Grünfeld": "d2d4 g8f6 c2c4 g7g6 b1c3 d7d5 c4d5 f6d5 e2e4 d5c3",
    "English Opening": "c2c4 e7e5 b1c3 g8f6 g1f3 b8c6 g2g3 d7d5 c4d5 f6d5",
    "Reti Opening": "g1f3 d7d5 g2g3 g8f6 f1g2 e7e6 e1g1 f8e7 d2d3 e8g8",
}


class OpeningBook:
    """`move(board)` -> (move, opening name) if the game is still in the book, else None."""

    def __init__(self, polyglot_path=None, seed=None):
        self.polyglot_path = polyglot_path
        self.rng = random.Random(seed)
        self.lines = {name: line.split() for name, line in LINES.items()}

    def move(self, board):
        if self.polyglot_path:
            with chess.polyglot.open_reader(self.polyglot_path) as reader:
                try:
                    entry = reader.weighted_choice(board, random=self.rng)
                    return entry.move, "book"
                except IndexError:
                    return None
        # built-in book: only from the standard start, following the moves played so far
        if board.root() != chess.Board():
            return None
        played = [mv.uci() for mv in board.move_stack]
        options = [(line[len(played)], name) for name, line in self.lines.items()
                   if len(line) > len(played) and line[:len(played)] == played]
        if not options:
            return None
        uci, name = self.rng.choice(options)
        mv = chess.Move.from_uci(uci)
        return (mv, name) if mv in board.legal_moves else None

    def name(self, board):
        """Name of the book line the game is on (for display), if any."""
        played = [mv.uci() for mv in board.move_stack]
        names = [n for n, line in self.lines.items() if played and line[:len(played)] == played]
        return names[0] if len(names) == 1 else None
