"""Chess board as a PIL image (no browser code needed), and click position -> square."""
import os

import chess
import matplotlib
from PIL import Image, ImageDraw, ImageFont

SQ = 64
MARGIN = 20
SIZE = 8 * SQ + 2 * MARGIN
LIGHT, DARK = (240, 217, 181), (181, 136, 99)
SELECT, LAST = (246, 246, 105), (205, 210, 106)
GLYPH = {chess.KING: "♚", chess.QUEEN: "♛", chess.ROOK: "♜",
         chess.BISHOP: "♝", chess.KNIGHT: "♞", chess.PAWN: "♟"}
_FONT = os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data", "fonts", "ttf", "DejaVuSans.ttf")
_fonts = {}


def _font(size):
    if size not in _fonts:
        _fonts[size] = ImageFont.truetype(_FONT, size)
    return _fonts[size]


def _xy(square, flip):
    f, r = chess.square_file(square), chess.square_rank(square)
    if flip:
        f, r = 7 - f, 7 - r
    return MARGIN + f * SQ, MARGIN + (7 - r) * SQ


def render(board, flip=False, selected=None, last_move=None, targets=()):
    img = Image.new("RGB", (SIZE, SIZE), (48, 46, 43))
    d = ImageDraw.Draw(img)
    marked = {s: LAST for s in ((last_move.from_square, last_move.to_square) if last_move else ())}
    if selected is not None:
        marked[selected] = SELECT
    for sq in chess.SQUARES:
        x, y = _xy(sq, flip)
        base = LIGHT if (chess.square_file(sq) + chess.square_rank(sq)) % 2 else DARK
        d.rectangle([x, y, x + SQ - 1, y + SQ - 1], fill=marked.get(sq, base))
    for sq in targets:  # legal destinations of the selected piece
        x, y = _xy(sq, flip)
        r = 9 if board.piece_at(sq) is None else 26
        d.ellipse([x + SQ / 2 - r, y + SQ / 2 - r, x + SQ / 2 + r, y + SQ / 2 + r],
                  outline=(90, 90, 90), width=3 if r > 10 else 0, fill=None if r > 10 else (90, 90, 90))
    if board.is_check():
        x, y = _xy(board.king(board.turn), flip)
        d.rectangle([x, y, x + SQ - 1, y + SQ - 1], outline=(220, 40, 40), width=4)
    font = _font(int(SQ * 0.82))
    for sq, piece in board.piece_map().items():
        x, y = _xy(sq, flip)
        white = piece.color == chess.WHITE
        d.text((x + SQ / 2, y + SQ / 2 + 2), GLYPH[piece.piece_type], font=font, anchor="mm",
               fill=(250, 250, 250) if white else (20, 20, 20), stroke_width=2,
               stroke_fill=(20, 20, 20) if white else (230, 230, 230))
    small = _font(13)
    for i in range(8):
        f = chr(ord("a") + (7 - i if flip else i))
        r = str(i + 1 if flip else 8 - i)
        d.text((MARGIN + i * SQ + SQ / 2, SIZE - MARGIN / 2), f, font=small, anchor="mm", fill=(200, 200, 200))
        d.text((MARGIN / 2, MARGIN + i * SQ + SQ / 2), r, font=small, anchor="mm", fill=(200, 200, 200))
    return img


def square_at(x, y, flip=False, size=SIZE):
    """Square under a click at pixel (x, y) of an image displayed at `size` pixels, or None."""
    s = SIZE / size
    x, y = x * s - MARGIN, y * s - MARGIN
    if not (0 <= x < 8 * SQ and 0 <= y < 8 * SQ):
        return None
    f, r = int(x // SQ), 7 - int(y // SQ)
    if flip:
        f, r = 7 - f, 7 - r
    return chess.square(f, r)
