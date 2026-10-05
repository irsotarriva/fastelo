"""Compact position encoding (~20 bytes/board) and its vectorized decoder.

A position is stored from the side-to-move perspective (black-to-move boards
are mirrored vertically with colors swapped):
  occ     uint64      occupancy bitboard
  pieces  uint8[16]   4-bit piece codes of the occupied squares in ascending
                      square order, low nibble first
  flags   uint8       bit0/1 own K/Q castling, bit2/3 opponent K/Q castling,
                      bit4 real side to move is black
  ep      uint8       en-passant file 0-7, or 8 for none
Piece codes: 0 empty, 1-6 own P N B R Q K, 7-12 opponent P N B R Q K.
"""
import chess
import numpy as np

N_PIECE_CODES = 13
EP_NONE = 8


def encode_board(board: chess.Board):
    turn = board.turn
    codes = [0] * 64
    for pt in range(1, 7):
        for color in (chess.WHITE, chess.BLACK):
            m = board.pieces_mask(pt, color)
            if turn == chess.BLACK:
                m = chess.flip_vertical(m)
            code = pt if color == turn else pt + 6
            while m:
                lsb = m & -m
                codes[lsb.bit_length() - 1] = code
                m ^= lsb
    occ = 0
    nibbles = []
    for sq in range(64):
        if codes[sq]:
            occ |= 1 << sq
            nibbles.append(codes[sq])
    nibbles += [0] * (32 - len(nibbles))
    pieces = bytes(nibbles[i] | (nibbles[i + 1] << 4) for i in range(0, 32, 2))

    own, opp = turn, not turn
    flags = (int(board.has_kingside_castling_rights(own))
             | int(board.has_queenside_castling_rights(own)) << 1
             | int(board.has_kingside_castling_rights(opp)) << 2
             | int(board.has_queenside_castling_rights(opp)) << 3
             | int(turn == chess.BLACK) << 4)
    ep = chess.square_file(board.ep_square) if board.has_legal_en_passant() else EP_NONE
    return occ, pieces, flags, ep


def decode_boards(occ: np.ndarray, pieces: np.ndarray) -> np.ndarray:
    """occ [N] uint64/int64, pieces [N,16] uint8 -> [N,64] uint8 piece codes."""
    occ = occ.view(np.int64) if occ.dtype == np.uint64 else occ
    bits = (occ[:, None] >> np.arange(64, dtype=np.int64)) & 1
    nib = np.empty((len(pieces), 32), dtype=np.uint8)
    nib[:, 0::2] = pieces & 0xF
    nib[:, 1::2] = pieces >> 4
    idx = np.clip(np.cumsum(bits, axis=1) - 1, 0, 31)
    return (np.take_along_axis(nib, idx, axis=1) * bits).astype(np.uint8)


def board_from_codes(codes, flags: int, ep: int, halfmove: int = 0) -> chess.Board:
    """Rebuild the mover-perspective board (always white to move)."""
    board = chess.Board(None)
    for sq in range(64):
        c = int(codes[sq])
        if c:
            board.set_piece_at(sq, chess.Piece((c - 1) % 6 + 1, chess.WHITE if c <= 6 else chess.BLACK))
    board.turn = chess.WHITE
    cr = 0
    if flags & 1: cr |= chess.BB_H1
    if flags & 2: cr |= chess.BB_A1
    if flags & 4: cr |= chess.BB_H8
    if flags & 8: cr |= chess.BB_A8
    board.castling_rights = cr
    board.ep_square = chess.square(ep, 5) if ep != EP_NONE else None
    board.halfmove_clock = halfmove
    return board
