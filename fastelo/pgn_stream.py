"""Stream games out of a Lichess .pgn.zst dump and filter them on headers.

Parsing the moves is the expensive part, so games are split and filtered in
the main process with plain string handling; only kept games go to workers.
"""
import hashlib
import io
import re
from dataclasses import dataclass, field

import zstandard

_HEADER_RE = re.compile(r'^\[(\w+) "(.*)"\]\s*$')

TC_CATEGORIES = ("ultrabullet", "bullet", "blitz", "rapid", "classical", "correspondence")


def iter_raw_games(zst_path):
    """Yield (headers: dict, movetext: str). Stops quietly at a truncated end,
    so a dump that is still downloading can be read."""
    with open(zst_path, "rb") as fh:
        reader = io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(fh),
                                  encoding="utf-8", errors="replace")
        headers, moves = {}, []
        try:
            for line in reader:
                if line.startswith("["):
                    if moves:
                        yield headers, " ".join(moves)
                        headers, moves = {}, []
                    m = _HEADER_RE.match(line)
                    if m:
                        headers[m.group(1)] = m.group(2)
                elif line.strip():
                    moves.append(line.strip())
        except zstandard.ZstdError:
            return  # truncated file; drop the partial last game
        if moves:
            yield headers, " ".join(moves)


def tc_category(time_control: str) -> str:
    """Lichess speed classes, by estimated duration base + 40 * increment."""
    if time_control in ("-", ""):
        return "correspondence"
    base, inc = (int(x) for x in time_control.split("+"))
    est = base + 40 * inc
    if est < 30:
        return "ultrabullet"
    if est < 180:
        return "bullet"
    if est < 480:
        return "blitz"
    if est < 1500:
        return "rapid"
    return "classical"


@dataclass
class FilterConfig:
    keep_tc: tuple = ("blitz", "rapid", "classical")
    keep_terminations: tuple = ("Normal", "Time forfeit")
    min_elo: int = 0
    max_elo: int = 4000
    # "exclude": humans only. "human_vs_bot": also keep games with exactly one bot.
    # "only_human_vs_bot": keep only those (to add them to a month already extracted).
    # Bot-vs-bot is always dropped.
    bots: str = "exclude"
    # (lo, hi): keep a human-vs-human game only if a player is <= lo or >= hi (tail enrichment).
    tail_elo: tuple = None


@dataclass
class HeaderInfo:
    game_id: str
    white_elo: int
    black_elo: int
    base: int
    inc: int
    tc: str
    result: int
    termination: str
    white_diff: int  # rating change from this game; large |diff| = uncertain (new) rating
    black_diff: int
    white_id: int    # 64-bit hash of the username, so we can count games per player
    black_id: int
    white_bot: bool = False
    black_bot: bool = False


DIFF_MISSING = 32767


def player_id(name: str) -> int:
    return int.from_bytes(hashlib.blake2b(name.lower().encode(), digest_size=8).digest(), "little")


def _diff(h, key):
    try:
        return int(h[key])
    except (KeyError, ValueError):
        return DIFF_MISSING


def check_headers(h: dict, cfg: FilterConfig):
    """Return (HeaderInfo, None) for a kept game, else (maybe HeaderInfo, reason)."""
    wb, bb = h.get("WhiteTitle") == "BOT", h.get("BlackTitle") == "BOT"
    if wb and bb:
        return None, "bot_vs_bot"
    if (wb or bb) and cfg.bots == "exclude":
        return None, "bot"
    if not (wb or bb) and cfg.bots == "only_human_vs_bot":
        return None, "human_vs_human"
    if h.get("Variant", "Standard") != "Standard":
        return None, "variant"
    try:
        we, be = int(h["WhiteElo"]), int(h["BlackElo"])
    except (KeyError, ValueError):
        return None, "no_elo"
    tc_str = h.get("TimeControl", "-")
    tc = tc_category(tc_str)
    base, inc = (0, 0) if tc == "correspondence" else (int(x) for x in tc_str.split("+"))
    result = {"1-0": 1, "0-1": -1, "1/2-1/2": 0}.get(h.get("Result"))
    info = HeaderInfo(h.get("Site", "")[-8:], we, be, base, inc, tc,
                      0 if result is None else result, h.get("Termination", ""),
                      _diff(h, "WhiteRatingDiff"), _diff(h, "BlackRatingDiff"),
                      player_id(h.get("White", "")), player_id(h.get("Black", "")), wb, bb)
    if result is None:
        return info, "no_result"
    if info.termination not in cfg.keep_terminations:
        return info, "termination"
    if tc not in cfg.keep_tc:
        return info, "time_control"
    human_elos = [e for e, bot in ((we, wb), (be, bb)) if not bot]
    if not all(cfg.min_elo <= e <= cfg.max_elo for e in human_elos):
        return info, "elo_range"
    if cfg.tail_elo and not (wb or bb):
        lo, hi = cfg.tail_elo
        if not (min(we, be) <= lo or max(we, be) >= hi):
            return info, "not_tail"
    return info, None


# SAN tokens start with a piece letter, a file or O (castling); move numbers and
# results start with digits so they never match. Annotation glyphs (?!, ??) are
# dropped. The { ... } comment after a move carries [%clk h:mm:ss] and, in
# analysed games, [%eval 0.25] or [%eval #-3] for the position after the move.
_COMMENT_SPLIT_RE = re.compile(r"(\{[^}]*\})")
_SAN_RE = re.compile(r"(?:^|\s)([NBRQKa-hO][^\s?!]*)")
_CLK_RE = re.compile(r"%clk (\d+):(\d+):(\d+)")
_EVAL_RE = re.compile(r"%eval (#?)(-?[\d.]+)")

EVAL_NONE = -32768
MATE_SCORE = 30000


def _parse_eval(comment):
    m = _EVAL_RE.search(comment)
    if not m:
        return EVAL_NONE
    if m.group(1):  # mate in n, sign = side that mates (the dump has corrupt n like #1032399278)
        n = int(m.group(2))
        return MATE_SCORE - min(abs(n), 999) if n > 0 else -MATE_SCORE + min(abs(n), 999)
    return max(-MATE_SCORE + 1000, min(MATE_SCORE - 1000, round(float(m.group(2)) * 100)))


def parse_movetext(movetext: str):
    """Return (sans, clocks_after_move_in_seconds or None, evals_white_cp).
    evals[i] is the engine eval after move i (EVAL_NONE when not analysed)."""
    sans, clocks, evals = [], [], []
    for part in _COMMENT_SPLIT_RE.split(movetext):
        if part.startswith("{"):
            if sans and clocks[-1] is None:
                m = _CLK_RE.search(part)
                if m:
                    clocks[-1] = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + int(m.group(3))
                evals[-1] = _parse_eval(part)
        else:
            for san in _SAN_RE.findall(part):
                sans.append(san)
                clocks.append(None)
                evals.append(EVAL_NONE)
    if not sans or None in clocks:
        clocks = None
    return sans, clocks, evals
