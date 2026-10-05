import random

import chess
import numpy as np

from fastelo.build_dataset import block_index
from fastelo.board_codec import board_from_codes, decode_boards, encode_board
from fastelo.extract import POS_DTYPE, process_game
from fastelo.pgn_stream import EVAL_NONE, MATE_SCORE, FilterConfig, check_headers, parse_movetext, tc_category
from fastelo.vocab import MOVES, VOCAB_SIZE, decode_move, encode_move, legal_mask

SAMPLE_MOVETEXT = ("1. e4 { [%clk 0:03:00] } 1... e6 { [%clk 0:03:00] } 2. d4 { [%clk 0:02:59] } "
                   "2... d5 { [%clk 0:02:58] } 3. exd5 { [%clk 0:02:55] } 3... exd5 { [%clk 0:02:50] } "
                   "4. Nf3 { [%clk 0:02:54] } 4... Nf6 { [%clk 0:02:49] } 5. Bd3 { [%clk 0:02:50] } "
                   "5... Bd6 { [%clk 0:02:40] } 6. O-O { [%clk 0:02:49] } 1-0")
SAMPLE_HEADERS = {"Site": "https://lichess.org/abcdEFGH", "WhiteElo": "1500", "BlackElo": "1600",
                  "Result": "1-0", "TimeControl": "180+0", "Termination": "Normal"}


def random_boards(n_games=30, seed=0):
    rng = random.Random(seed)
    for _ in range(n_games):
        board = chess.Board()
        for _ in range(rng.randint(1, 150)):
            moves = list(board.legal_moves)
            if not moves:
                break
            board.push(rng.choice(moves))
            yield board.copy()


def test_vocab_unique_and_roundtrip():
    assert len(set(MOVES)) == VOCAB_SIZE
    for board in random_boards(5):
        for mv in board.legal_moves:
            assert decode_move(encode_move(mv, board.turn), board.turn) == mv


def test_codec_roundtrip_preserves_legal_moves():
    """Decoded mover-perspective board must have exactly the same legal moves
    (in vocab space) as the original board. Checks codec, mirroring and vocab together."""
    boards = list(random_boards())
    enc = [encode_board(b) for b in boards]
    occ = np.array([e[0] for e in enc], dtype=np.uint64)
    pieces = np.frombuffer(b"".join(e[1] for e in enc), np.uint8).reshape(-1, 16)
    codes = decode_boards(occ, pieces)
    for board, (_, _, flags, ep), c in zip(boards, enc, codes):
        rebuilt = board_from_codes(c, flags, ep)
        expected = board.mirror() if board.turn == chess.BLACK else board
        assert rebuilt.board_fen() == expected.board_fen()
        assert np.array_equal(legal_mask(rebuilt), legal_mask(board))


def test_block_index():
    assert block_index([10, 3, 7], [2, 0, 3]).tolist() == [10, 11, 7, 8, 9]
    assert block_index([], []).tolist() == []


def test_headers_and_filters():
    info, reason = check_headers(SAMPLE_HEADERS, FilterConfig())
    assert reason is None and info.game_id == "abcdEFGH" and info.result == 1
    assert check_headers({**SAMPLE_HEADERS, "WhiteTitle": "BOT"}, FilterConfig())[1] == "bot"
    assert check_headers({**SAMPLE_HEADERS, "TimeControl": "60+0"}, FilterConfig())[1] == "time_control"
    assert check_headers({**SAMPLE_HEADERS, "Termination": "Abandoned"}, FilterConfig())[1] == "termination"
    assert [tc_category(t) for t in ("15+0", "120+1", "180+0", "600+5", "1800+20", "-")] == \
        ["ultrabullet", "bullet", "blitz", "rapid", "classical", "correspondence"]


def test_parse_movetext_and_process_game():
    sans, clocks, evals = parse_movetext(SAMPLE_MOVETEXT)
    assert sans[:3] == ["e4", "e6", "d4"] and sans[-1] == "O-O" and len(sans) == len(clocks) == 11
    assert all(e == EVAL_NONE for e in evals)
    assert parse_movetext("1. e4 e5 2. Nf3 1-0")[:2] == (["e4", "e5", "Nf3"], None)
    sans, clocks, evals = parse_movetext(
        "1. e4 { [%eval 0.18] [%clk 0:10:00] } 1... f6?! { [%eval 1.5] [%clk 0:10:00] } "
        "2. d4 { [%eval #4] [%clk 0:09:58] } 2... g5?? { [%eval #-1] [%clk 0:09:50] } 0-1")
    assert sans == ["e4", "f6", "d4", "g5"] and clocks == [600, 600, 598, 590]
    assert evals == [18, 150, MATE_SCORE - 4, -MATE_SCORE + 1]
    # Corrupt mate distance seen in the 2025-11 dump must still fit int16.
    assert parse_movetext("1. e4 { [%eval #-1032399278] [%clk 0:10:00] } 1-0")[2] == [-MATE_SCORE + 999]
    info, _ = check_headers(SAMPLE_HEADERS, FilterConfig())
    game, rows = process_game(info, SAMPLE_MOVETEXT, min_plies=10)
    pos = np.array(rows, dtype=POS_DTYPE)
    assert game[-2] == 11 and len(pos) == 11
    assert pos["clock"][0] == 180 and pos["clock"][2] == 180 and pos["clock"][4] == 179
    assert pos["spent"][4] == 179 - 175  # 3. exd5: 2:59 -> 2:55
    # Black's e7e6 is stored mirrored, as e2e3.
    assert MOVES[pos["move"][1]] == "e2e3"
    assert MOVES[pos["move"][10]] == "e1g1"
