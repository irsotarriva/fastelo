"""Move vocabulary in side-to-move perspective.

Every position is stored as if white were to move (black-to-move positions are
mirrored vertically with colors swapped), so the vocabulary only needs moves a
white player could make: queen-like moves and knight moves from every square,
plus white promotions. That gives 1880 moves, the same size Maia uses.
"""
import chess
import numpy as np


def _build():
    moves = set()
    for sq in chess.SQUARES:
        f, r = chess.square_file(sq), chess.square_rank(sq)
        for df, dr in [(1, 1), (1, -1), (-1, 1), (-1, -1), (0, 1), (0, -1), (1, 0), (-1, 0)]:
            for d in range(1, 8):
                nf, nr = f + df * d, r + dr * d
                if not (0 <= nf < 8 and 0 <= nr < 8):
                    break
                moves.add(chess.Move(sq, chess.square(nf, nr)).uci())
        for df, dr in [(1, 2), (2, 1), (-1, 2), (-2, 1), (1, -2), (2, -1), (-1, -2), (-2, -1)]:
            nf, nr = f + df, r + dr
            if 0 <= nf < 8 and 0 <= nr < 8:
                moves.add(chess.Move(sq, chess.square(nf, nr)).uci())
    for f in range(8):
        for df in (-1, 0, 1):
            if 0 <= f + df < 8:
                for p in "qrbn":
                    moves.add(chess.Move(chess.square(f, 6), chess.square(f + df, 7),
                                         promotion=chess.Piece.from_symbol(p).piece_type).uci())
    return sorted(moves)


MOVES = _build()
MOVE_TO_IDX = {m: i for i, m in enumerate(MOVES)}
VOCAB_SIZE = len(MOVES)
assert VOCAB_SIZE == 1880, VOCAB_SIZE

# (from, to, promotion piece type or 0) per vocab index, for vectorized legality checks.
MOVE_FROM = np.array([chess.Move.from_uci(m).from_square for m in MOVES], dtype=np.int16)
MOVE_TO = np.array([chess.Move.from_uci(m).to_square for m in MOVES], dtype=np.int16)
MOVE_PROMO = np.array([chess.Move.from_uci(m).promotion or 0 for m in MOVES], dtype=np.int8)


def mirror_move(move: chess.Move) -> chess.Move:
    return chess.Move(chess.square_mirror(move.from_square), chess.square_mirror(move.to_square),
                      promotion=move.promotion)


def encode_move(move: chess.Move, turn: chess.Color) -> int:
    """Vocab index of a real move played by `turn`."""
    if turn == chess.BLACK:
        move = mirror_move(move)
    return MOVE_TO_IDX[move.uci()]


def decode_move(idx: int, turn: chess.Color) -> chess.Move:
    """Real move for side `turn` from a vocab index."""
    move = chess.Move.from_uci(MOVES[idx])
    return mirror_move(move) if turn == chess.BLACK else move


def legal_mask(board: chess.Board) -> np.ndarray:
    """Boolean [VOCAB_SIZE] mask of legal moves in `board` (mover perspective)."""
    mask = np.zeros(VOCAB_SIZE, dtype=bool)
    for mv in board.legal_moves:
        mask[encode_move(mv, board.turn)] = True
    return mask
