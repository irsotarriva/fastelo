"""fastelo demo: play the bot, watch your Elo estimate move after every move.

    python -m fastelo.demo.app --calibration runs/player/calibration.json --rater runs/rater/ckpt.pt
    python -m fastelo.demo.app --player runs/player/ckpt.pt --rater runs/rater/ckpt.pt   # uncalibrated bot

Works the same locally, inside a notebook (`build_app(...).launch()`), and as a Hugging Face Space.
The bot plays at the visitor's current combined estimate (1500 before the first finished game).
"""
import argparse
import os

import chess
import gradio as gr
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ..play import PlayerPolicy, CalibratedPolicy, MoveRater  # noqa: E402
from .board_image import SIZE, render, square_at  # noqa: E402
from .session import RatingSession  # noqa: E402

from ..response import REPORT_RANGE  # noqa: E402

BLUE, ORANGE, GREY = "#3b6ea8", "#d9822b", "#888888"
LO, HI = REPORT_RANGE
YLIM = (LO - 200, HI + 200)


def limits(ax):
    """Shade the ratings the models cannot measure; everything plotted is clipped to the edge of it."""
    ax.axhspan(HI, YLIM[1], color=GREY, alpha=0.18, lw=0)
    ax.axhspan(YLIM[0], LO, color=GREY, alpha=0.18, lw=0)
    ax.text(0.01, 0.99, f"{HI:.0f}+ : not measurable", transform=ax.transAxes, va="top", fontsize=8, color="#555555")
    ax.text(0.01, 0.01, f"below {LO:.0f} : not measurable", transform=ax.transAxes, va="bottom", fontsize=8,
            color="#555555")
    ax.set_ylim(*YLIM)


clip = lambda x: np.clip(x, LO, HI)


def fmt(value, minus, plus, syst):
    """1480 -210/+260 (stat) ±15 (syst); the systematic part is left out when there is none. Outside
    the range the models were validated on, no number is given."""
    if value >= HI:
        return f"{HI:.0f}+ (above what this model can measure)"
    if value <= LO:
        return f"below {LO:.0f} (under what this model can measure)"
    stat = f"{value:.0f} \u2212{minus:.0f}/+{plus:.0f}"
    return f"{stat} (stat) ±{syst:.0f} (syst)" if syst > 0 else stat


# ------------------------------------------------------------------------------ plots
def live_plot(session):
    fig, ax = plt.subplots(figsize=(6, 3.4))
    r = session.rating_curve()
    n, mu = r["n"], r["value"]
    if len(n):
        ax.fill_between(n, clip(r["lo2"]), clip(r["hi2"]), color=BLUE, alpha=0.12, lw=0, label="±2σ (stat.)")
        ax.fill_between(n, clip(r["lo"]), clip(r["hi"]), color=BLUE, alpha=0.25, lw=0, label="±1σ (stat.)")
        ax.plot(n, clip(mu), "o-", color=BLUE, ms=3, label="estimate")
        ax.set_xlim(0.5, max(10, n[-1]) + 0.5)
        ax.set_title(f"This game: {fmt(mu[-1], mu[-1] - r['lo'][-1], r['hi'][-1] - mu[-1], r['syst'][-1])} "
                     f"after {n[-1]} moves", fontsize=10)
    else:
        ax.set_title("This game: make a move", fontsize=11)
        ax.set_xlim(0.5, 10.5)
    g = session.game
    if g is not None and g.bot_elos:
        # bot strength before each of the user's moves (its reply follows the user's move)
        k = len(n) if len(n) else 1
        be = (g.bot_elos[1:] if not g.user_white else g.bot_elos)[:k]
        if len(be):
            ax.step(np.arange(1, len(be) + 1), be, where="mid", color=GREY, ls="--", lw=1.2,
                    label=f"bot strength (now {g.bot_elos[-1]:.0f})")
    limits(ax)
    ax.set_xlabel("your move")
    ax.set_ylabel("Elo")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    plt.close(fig)
    return fig


def games_plot(session):
    fig, ax = plt.subplots(figsize=(6, 3.4))
    R = list(session.results)
    live = session.live_measurement()
    pts = R + ([live] if live else [])
    if pts:
        x = np.arange(1, len(pts) + 1)
        val = lambda ms: clip(np.array([m["mean"] for m in ms]))
        bars = lambda ms: np.array([val(ms) - clip(np.array([m["mean"] - m["minus"] for m in ms])),
                                    clip(np.array([m["mean"] + m["plus"] for m in ms])) - val(ms)])
        if R:
            ax.errorbar(x[:len(R)], val(R), yerr=bars(R), fmt="o", color=BLUE, capsize=4,
                        label="finished game (±1σ stat.)")
        if live:
            ax.errorbar([x[-1]], val([live]), yerr=bars([live]), fmt="o", mfc="white", color=BLUE, capsize=4,
                        label="game in progress (±1σ stat.)")
        c = session.combined_full()
        mu = c["value"]
        lo, hi = mu - np.hypot(c["minus"], c["syst"]), mu + np.hypot(c["plus"], c["syst"])
        ax.axhspan(clip(lo), clip(hi), color=ORANGE, alpha=0.15, lw=0, label="all games: stat. + syst.")
        ax.axhspan(clip(mu - c["minus"]), clip(mu + c["plus"]), color=ORANGE, alpha=0.3, lw=0, label="all games: stat.")
        ax.axhline(clip(mu), color=ORANGE, lw=2)
        ax.set_xlim(0.5, max(5, len(pts)) + 0.5)
        ax.set_xticks(x)
        ax.legend(fontsize=8, loc="lower right")
        ax.set_title(f"Your rating so far: {fmt(mu, c['minus'], c['plus'], c['syst'])} "
                     f"({len(pts)} game{'s' if len(pts) > 1 else ''})", fontsize=10)
    else:
        ax.set_xlim(0.5, 5.5)
        ax.set_title("Your games appear here", fontsize=11)
    limits(ax)
    ax.set_xlabel("game")
    ax.set_ylabel("Elo")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    plt.close(fig)
    return fig


# --------------------------------------------------------------------------------- app
def status(session, note=""):
    g = session.game
    if g is None:
        return "Press **New game** to start."
    lines = []
    if note:
        lines.append(note)
    if g.result is not None:
        won = g.result == ("1-0" if g.user_white else "0-1")
        outcome = "You won" if won else ("Draw" if g.result == "1/2-1/2" else
                                        ("You ran out of time" if g.flagged else
                                         "You resigned" if g.resigned else "You lost"))
        lines.append(f"**{outcome}** ({g.result}). The estimate after your last move is part of your rating.")
        lines.append("Press **New game** for another measurement.")
    else:
        lines.append(f"You play **{'white' if g.user_white else 'black'}**. The bot adapts to your best "
                     f"estimated rating after every move; it is now playing at about **{g.bot_elo:.0f}**. "
                     "Click a piece, then its destination.")
    if g.moves:
        b = chess.Board()
        lines.append("Moves: " + b.variation_san(g.moves))
    return "\n\n".join(lines)


TIME_CONTROLS = {"10+0": (600, 0), "10+5": (600, 5), "15+10": (900, 10)}   # all rapid on Lichess


def clock_text(session):
    t = session.time_left() if session is not None else None
    if t is None:
        return "### \u23f1 --:--"
    warn = " \u26a0" if t < 30 and session.game.result is None else ""
    return f"### \u23f1 {int(t) // 60}:{int(t) % 60:02d}{warn}"


def build_app(policy, rater, title="How strong are you? A chess bot that rates you as you play"):
    def view(session, selected=None, note=""):
        g = session.game
        if g is None:
            img = render(chess.Board())
        else:
            targets = [m.to_square for m in g.board.legal_moves if m.from_square == selected] if selected is not None else []
            img = render(g.board, flip=not g.user_white, selected=selected,
                         last_move=g.moves[-1] if g.moves else None, targets=targets)
        return img, status(session, note), live_plot(session), games_plot(session), clock_text(session)

    blank = (render(chess.Board()), "Press **New game** to start.", None, None, clock_text(None))

    def new_game(session, color, tc):
        session = session or RatingSession(policy, rater)
        white = color == "White" or (color == "Random" and np.random.random() < 0.5)
        base, inc = TIME_CONTROLS[tc]
        session.new_game(user_white=white, base=base, increment=inc)
        return (session, None) + view(session)

    def resign(session):
        if session is None:
            return (session, None) + blank
        session.resign()
        return (session, None) + view(session)

    def tick(session):
        """Once a second: refresh the clock; when the time runs out, end the game and redraw."""
        if session is None or session.game is None:
            return (gr.skip(),) * 6 + (clock_text(session),)
        if session.check_flag():
            return (session, None) + view(session)
        return (gr.skip(),) * 6 + (clock_text(session),)

    def click(session, selected, evt: gr.SelectData):
        if session is None or session.game is None or not session.game.user_turn:
            return (session, selected) + (view(session, selected) if session else blank)
        g = session.game
        sq = square_at(evt.index[0], evt.index[1], flip=not g.user_white, size=SIZE)
        if sq is None:
            return (session, selected) + view(session, selected)
        piece = g.board.piece_at(sq)
        own = piece is not None and piece.color == g.board.turn
        if selected is not None and not own:
            move = chess.Move(selected, sq)
            if g.board.piece_type_at(selected) == chess.PAWN and chess.square_rank(sq) in (0, 7):
                move.promotion = chess.QUEEN  # always promote to a queen
            if session.user_move(move):
                return (session, None) + view(session)
            return (session, None) + view(session, None, "" if g.flagged else "Illegal move.")
        return (session, sq if own else None) + view(session, sq if own else None)

    with gr.Blocks(title="fastelo") as app:
        gr.Markdown(f"## {title}\nA neural network reads your moves and estimates your Lichess-style rating "
                    "after every move, with its uncertainty. Each game is one measurement; "
                    "they are combined into your overall rating (the game in progress counts too). "
                    "The statistical error shrinks as you play more; the systematic error is what remains "
                    "of the network's known bias after correcting for it, and does not. "
                    f"Ratings are only given between {LO:.0f} and {HI:.0f}; outside that range the models "
                    "have not been validated. "
                    "The bot re-targets your current best estimate before every move. Games are timed "
                    "(rapid), like the games the network learned from: running out of time loses the game.")
        session = gr.State(None)
        selected = gr.State(None)
        with gr.Row():
            with gr.Column(scale=5):
                clock = gr.Markdown(clock_text(None))
                board = gr.Image(value=render(chess.Board()), type="pil", show_label=False, interactive=False,
                                 height=SIZE, width=SIZE)
                with gr.Row():
                    color = gr.Radio(["White", "Black", "Random"], value="White", label="Play as")
                    tc = gr.Radio(list(TIME_CONTROLS), value="10+0", label="Your clock (min + s/move)")
                    new = gr.Button("New game", variant="primary")
                    res = gr.Button("Resign")
                info = gr.Markdown("Press **New game** to start.")
            with gr.Column(scale=5):
                live = gr.Plot(label="Live estimate (this game)")
                games = gr.Plot(label="All your games")
        outs = [session, selected, board, info, live, games, clock]
        new.click(new_game, [session, color, tc], outs)
        gr.Timer(1.0).tick(tick, [session], outs, show_progress="hidden")
        res.click(resign, [session], outs)
        board.select(click, [session, selected], outs)
    return app


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibration", default=None, help="calibration.json (bot plays calibrated target Elos)")
    ap.add_argument("--player", default=None, help="player checkpoint (overrides the one named in the calibration)")
    ap.add_argument("--strong", default=None)
    ap.add_argument("--response", default="auto", help="bias correction (fastelo.response fit); auto = "
                    "rating_response.json next to the calibration (response.json next to --rater), none = uncorrected")
    ap.add_argument("--share", action="store_true")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args(argv)
    policy = (CalibratedPolicy(args.calibration, player=args.player, strong=args.strong) if args.calibration
              else PlayerPolicy(args.player))
    response = None if args.response == "none" else args.response
    if response == "auto":
        near = os.path.join(os.path.dirname(os.path.abspath(args.calibration or args.player)), "rating_response.json")
        response = near if os.path.exists(near) else None
    rater = MoveRater(policy, response=response)
    build_app(policy, rater).launch(share=args.share, server_port=args.port)


if __name__ == "__main__":
    main()
