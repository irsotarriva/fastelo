"""Leela Chess Zero networks (ONNX, converted with `lc0 leela2onnx`) as a teacher.

Only the INPUT_CLASSICAL_112_PLANE format is supported (T1, T82, BT3, BT4 nets):
  planes 13*t + 0..5   current mover's P N B R Q K, t = 0 (now) .. 7 (7 plies ago)
  planes 13*t + 6..11  opponent's pieces
  plane  13*t + 12     position t had occurred before (repetition)
  104..107             castling: we O-O-O, we O-O, they O-O-O, they O-O
  108                  side to move is black (all ones)
  109                  rule-50 ply counter (raw value)
  110                  zeros
  111                  ones
Every history board is seen from the *current* mover: flipped vertically when black is to
move, exactly like our records. Missing history (game start) stays zero.

Policy: 1858 logits in lc0's move order, in the mover's frame. A plain rank-7 -> rank-8 pawn
move in that list is the knight promotion.
"""
import chess
import numpy as np

from .vocab import MOVES, VOCAB_SIZE, encode_move


def _lc0_move_order():
    """lc0's 1858-move policy order: queen-like and knight moves grouped by from-square, each
    group sorted by to-square (a1..h8), then rank-7 promotions to q/r/b. Generated here rather
    than vendored; verified identical to lczero-training's tf/policy_index.py."""
    out = []
    for f in range(64):
        for t in range(64):
            df, dr = abs(t % 8 - f % 8), abs(t // 8 - f // 8)
            if t != f and (df == 0 or dr == 0 or df == dr or (df, dr) in ((1, 2), (2, 1))):
                out.append(chess.square_name(f) + chess.square_name(t))
    for f in range(48, 56):
        for t in (f + 7, f + 8, f + 9):
            if 56 <= t < 64 and abs(t % 8 - f % 8) <= 1:
                out += [chess.square_name(f) + chess.square_name(t) + p for p in "qrb"]
    return out


LC0_MOVES = _lc0_move_order()
assert len(LC0_MOVES) == 1858

LC0_SIZE = len(LC0_MOVES)
_LC0_POS = {m: i for i, m in enumerate(LC0_MOVES)}
# our vocab index -> lc0 index (knight promotions share the plain move's slot)
OURS_TO_LC0 = np.array([_LC0_POS[m[:4] if m.endswith("n") and len(m) == 5 else m] for m in MOVES], np.int64)
assert len(OURS_TO_LC0) == VOCAB_SIZE


def _masks(board: chess.Board, us: chess.Color, flip: bool):
    out = []
    for color in (us, not us):
        for pt in range(1, 7):
            m = board.pieces_mask(pt, color)
            out.append(chess.flip_vertical(m) if flip else m)
    return out


def _bits_to_plane(mask: int) -> np.ndarray:
    return np.unpackbits(np.frombuffer(mask.to_bytes(8, "little"), np.uint8), bitorder="little").reshape(8, 8)


def encode_planes(board: chess.Board) -> np.ndarray:
    """Reference encoder: [112, 8, 8] float32 from a python-chess board with its move stack."""
    planes = np.zeros((112, 8, 8), np.float32)
    us = board.turn
    flip = us == chess.BLACK
    hist = board.copy()
    for t in range(8):
        for k, m in enumerate(_masks(hist, us, flip)):
            planes[13 * t + k] = _bits_to_plane(m)
        if hist.is_repetition(2):  # this position occurred at least once before
            planes[13 * t + 12] = 1.0
        if not hist.move_stack:
            break
        hist.pop()
    them = not us
    planes[104] = float(board.has_queenside_castling_rights(us))
    planes[105] = float(board.has_kingside_castling_rights(us))
    planes[106] = float(board.has_queenside_castling_rights(them))
    planes[107] = float(board.has_kingside_castling_rights(them))
    planes[108] = float(us == chess.BLACK)
    planes[109] = float(board.halfmove_clock)
    planes[111] = 1.0
    return planes


_LC0_CASTLE = {}  # (king from, king to) in the mover frame -> lc0 "king takes rook" index
for _k_to, _r_from in ((chess.G1, chess.H1), (chess.C1, chess.A1)):
    _LC0_CASTLE[_k_to] = _LC0_POS[chess.square_name(chess.E1) + chess.square_name(_r_from)]


def lc0_index(board: chess.Board, move: chess.Move) -> int:
    """lc0 policy index of a legal move. Castling is king-takes-own-rook in lc0 (O-O = e1h1)."""
    if board.is_castling(move):
        to = move.to_square if board.turn == chess.WHITE else chess.square_mirror(move.to_square)
        return _LC0_CASTLE[to]
    return int(OURS_TO_LC0[encode_move(move, board.turn)])


def lc0_uci(board: chess.Board, move: chess.Move) -> str:
    """How lc0 prints a move (castling as king-takes-rook, real coordinates)."""
    if board.is_castling(move):
        rook_file = 7 if chess.square_file(move.to_square) == 6 else 0
        return chess.square_name(move.from_square) + chess.square_name(
            chess.square(rook_file, chess.square_rank(move.from_square)))
    return move.uci()


def legal_lc0_indices(board: chess.Board):
    """(our vocab indices, lc0 indices) of the legal moves, mover frame."""
    moves = list(board.legal_moves)
    ours = np.array([encode_move(m, board.turn) for m in moves], np.int64)
    return ours, np.array([lc0_index(board, m) for m in moves], np.int64)


def board_with_history(games, positions, idx) -> chess.Board:
    """Replay the game containing dataset position `idx` (one stage-2 shard: games table and
    positions array) up to that position, so the board carries its real move history."""
    from .board_codec import encode_board
    from .vocab import decode_move
    gi = int(np.searchsorted(games["start"], idx, "right")) - 1
    start = int(games["start"][gi])
    board = chess.Board()
    for k in range(start, idx):
        board.push(decode_move(int(positions["move"][k]), board.turn))
    occ, pieces, _, _ = encode_board(board)
    if occ != int(positions["occ"][idx]) or pieces != positions["pieces"][idx].tobytes():
        raise ValueError(f"replay of game {gi} does not reach stored position {idx}")
    return board


class LeelaONNX:
    def __init__(self, path, providers=None, threads=None):
        import onnxruntime as ort
        opts = ort.SessionOptions()
        if threads:
            opts.intra_op_num_threads = threads
        self.session = ort.InferenceSession(path, opts, providers=providers or ["CPUExecutionProvider"])
        if providers and "CUDAExecutionProvider" in providers and \
                "CUDAExecutionProvider" not in self.session.get_providers():
            raise RuntimeError("CUDA was requested but onnxruntime fell back to CPU "
                               "(CUDA/cuDNN version mismatch? see the onnxruntime log above)")
        self.input_type = self.session.get_inputs()[0].type  # tensor(float) or tensor(float16)

    def run(self, planes: np.ndarray):
        """planes [B,112,8,8] -> policy logits [B,1858], wdl probabilities [B,3]."""
        x = planes.astype(np.float16 if "float16" in self.input_type else np.float32)
        policy, wdl = self.session.run(["/output/policy", "/output/wdl"], {"/input/planes": x})
        return policy.astype(np.float32), wdl.astype(np.float32)

    def move_distribution(self, board: chess.Board, temperature=1.0):
        """{chess.Move: probability} over legal moves, plus WDL, for one position."""
        logits, wdl = self.run(encode_planes(board)[None])
        ours, lc0 = legal_lc0_indices(board)
        z = logits[0, lc0] / temperature
        p = np.exp(z - z.max())
        p /= p.sum()
        moves = list(board.legal_moves)
        return dict(zip(moves, p)), wdl[0]
