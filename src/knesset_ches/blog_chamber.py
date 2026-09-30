"""The chamber beeswarm and the careers small multiples.

    python -m knesset_ches.blog chamber
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgba

from knesset_ches import blog_style as B
from knesset_ches import tables as T

SRC_POSITIONS = ("Jev model", B.SRC_CHES)
# shorter, to fit the phone footer in two lines
SRC_CORPUS_FIT = "Knesset Corpus (Goldin et al. 2025), extended to July 2026 from Knesset protocols"
KEY_SPACING = (0.15, 0.20)               # tighter, to fit the wide key on one row
TRI_S = 48                               # camp-median triangles


def n_sentences_drawn(dimension: str, drawn: pd.DataFrame) -> int:
    """Sentences behind the drawn estimates, matched on party too."""
    terms = drawn[["speaker_id", "period", "faction_id"]].itertuples(index=False, name=None)
    return B.sentences_behind([dimension], terms)


def one_per_member(df: pd.DataFrame) -> pd.DataFrame:
    """Keep each member's party row with the most speeches in each period."""
    d = df.assign(_suf=(~df["thin"].astype(bool)).to_numpy(), _fid=pd.to_numeric(df["faction_id"], errors="coerce"))
    d = d.sort_values(["n_speeches", "_suf", "_fid"], kind="stable")
    d = d.drop_duplicates(["period", "speaker_id"], keep="last")
    return d.drop(columns=["_suf", "_fid"]).reset_index(drop=True)


# chamber_lrgen

CHAMBER_ROWS = ["13", "15", "17", "20", "21-22-23", "25"]


def _chamber(df: pd.DataFrame, med: pd.DataFrame, n: int, phone: bool) -> B.BlogPage:
    nb = 50 if phone else 80                     # bins across the scale
    count = B.from_sentences(n).capitalize().replace(" ", "\u00a0")     # keep on one line
    title = "The two camps drew together in the Kadima years and are now as far apart as ever"
    sub = ("Each dot is one member, placed by their speeches; the camps are the Right and Orthodox blocs against "
           f"everyone else. {count}")
    pale = to_rgba(B.BLOC["Right"], 0.45)
    key = B.BLOCS + [("other lists", B.OTHER), ("each camp's middle member", B.INK, "triangle"),
                     ("pale: less certain", pale)]
    src = B.sources(SRC_CORPUS_FIT, *SRC_POSITIONS, partial=True)
    page = B.BlogPage(title, sub, src, width=6.0 if phone else 10.0, key=key, max_sub_lines=3 if phone else 2,
                      key_spacing=KEY_SPACING)
    left = page.M + (1.04 if phone else 1.25)                    # row labels
    width = page.W - page.M - (0.50 if phone else 0.80) - left   # gap figures
    bin_in = width / nb
    d_pt = min(6.5, bin_in * 72 * 0.92)          # pt, at most a bin wide
    d_in = d_pt / 72
    head = 0.20                                  # for 'points apart'
    below = 0.24                                 # triangles and bracket
    axis_h = 0.52                                # ticks and direction words
    tallest = 1
    for p in CHAMBER_ROWS:
        g = df[df["period"] == p]
        tallest = max(tallest, int(np.minimum(nb - 1, (g["value"] / 10 * nb).astype(int)).value_counts().max()))
    # rows fit the tallest stack unless the page gets too tall
    ideal_row = below + d_in / 2 + tallest * bin_in + 0.06
    room = 7.5 - page.top - page.foot_h - head - axis_h
    rowh = min(ideal_row, room / len(CHAMBER_ROWS))
    step = min(bin_in, (rowh - below - d_in / 2 - 0.06) / max(1, tallest - 1 + 0.5))
    rows_h = head + rowh * len(CHAMBER_ROWS)
    page.set_body(rows_h + axis_h)
    y0 = page.top
    ax = page.axes(left, y0, width, rows_h)
    ax.set_xlim(0, 10)
    ax.set_ylim(rows_h, 0)
    ax.set_xticks([])
    ax.set_yticks([])
    last_base = head + len(CHAMBER_ROWS) * rowh - below
    for v in (0, 2.5, 5, 7.5, 10):
        ax.plot([v, v], [head, last_base + 0.12], color=B.GRID, lw=0.6, zorder=0, solid_capstyle="butt")
    order = {b: i for i, b in enumerate(B.ROW_ORDER)}
    for i, p in enumerate(CHAMBER_ROWS):
        g = df[df["period"] == p]
        base = head + (i + 1) * rowh - below
        bins: dict[int, list[tuple[str, bool]]] = {}
        for v, b, thin in zip(g["value"], g["bloc"].astype(str), g["thin"]):
            bins.setdefault(min(nb - 1, int(v / 10 * nb)), []).append((b, bool(thin)))
        xs, ys, cs = [], [], []
        for b, members in bins.items():
            # stack by bloc in row order, pale dots above solid
            for k, (bloc, thin) in enumerate(sorted(members, key=lambda m: (order.get(m[0], 99), m[1]))):
                xs.append((b + 0.5) * 10 / nb)
                ys.append(base - d_in / 2 - 0.012 - k * step)
                cs.append(to_rgba(B.BLOC.get(bloc, B.OTHER), 0.45 if thin else 1.0))
        ax.plot([0, 10], [base, base], color=B.INK, lw=0.9, zorder=2, solid_capstyle="butt")
        ax.scatter(xs, ys, s=d_pt ** 2, c=cs, edgecolors=B.PAPER, linewidths=0.45, zorder=3, clip_on=False)
        ma, mb = float(med.loc[p, "median_a"]), float(med.loc[p, "median_b"])
        tri_y = base + 0.065
        ax.plot([mb, ma], [tri_y + 0.035, tri_y + 0.035], color=B.INK, lw=0.9, zorder=4, clip_on=False)
        ax.scatter([mb, ma], [tri_y, tri_y], marker="^", s=TRI_S, c=B.INK, linewidths=0, zorder=5, clip_on=False)
        page.text(page.M, y0 + base - 0.13, B.KNESSET_YEARS[p], size=12.5, weight=600, serif=True)
        page.text(page.M, y0 + base + 0.08, f"{g['speaker_id'].nunique()} members", size=B.T_COUNT, colour=B.CAP)
        page.text(page.W - page.M, y0 + base + 0.03, f"{ma - mb:.1f}", size=15, weight=600, serif=True, ha="right",
                  va="center")
    page.note(page.W - page.M, y0 + head - 0.06, "points apart", ha="right")
    ty = y0 + last_base + below - 0.02
    for v in (0, 2.5, 5, 7.5, 10):
        page.text(left + v / 10 * width, ty, f"{v:g}", size=B.T_TICK, colour=B.CAP, ha="center", va="top")
    page.note(left, ty + 0.22, "← more left-wing", va="top")
    page.note(left + width, ty + 0.22, "more right-wing →", ha="right", va="top")
    return page


def draw_chamber_lrgen(tables: T.Tables) -> list[Path]:
    df = one_per_member(T.prep_chamber(tables, "lrgen", "knesset", CHAMBER_ROWS))
    med = T.chamber_medians(df).set_index("period").reindex(CHAMBER_ROWS)
    n = n_sentences_drawn("lrgen", df)
    left = df[df["bloc"] == "Left"].groupby("period").size().reindex(CHAMBER_ROWS).fillna(0).astype(int)
    g = med["gap"]
    alt = ("Six rows of dots, one per member, for six Knessets from 1992–96 to 2022–26, placed from left-wing to "
           f"right-wing. The middle members of the two camps sat {g['13']:.1f} points apart in 1992–96, {g['17']:.1f} "
           f"in 2006–09 and {g['25']:.1f} in 2022–26, and the red Left dots thin from {left['13']} to {left['25']} by "
           "the last row.")
    paths = []
    for phone in (False, True):
        paths += _chamber(df, med, n, phone).save("chamber_lrgen", variant="_phone" if phone else "", alt=alt)
    return paths


# careers_israel_palestine

CAREERS = [("560", "Ahmad Tibi"), ("23565", "Merav Michaeli"), ("965", "Binyamin Netanyahu"), ("482", "Ariel Sharon"),
           ("469", "Tzipi Livni"), ("1064", "Ehud Olmert")]


def _career_points(tables: T.Tables) -> pd.DataFrame:
    """One sufficient estimate per member and Knesset, placed at mid-term."""
    m = tables.mk
    k = m[(m["dimension"] == "israel_palestine") & (m["period_type"] == "knesset")].copy()
    k = k[k["sufficient"].fillna(False).astype(bool) & np.isfinite(k["estimate"])]
    k = one_per_member(k.assign(thin=False))
    k["speaker_id"] = k["speaker_id"].astype(str)
    k["slot"] = k["period"].map({p: i for i, p in enumerate(B.KNESSET_SLOTS)})
    start, end = pd.to_datetime(k["period_start"]), pd.to_datetime(k["period_end"])
    mid = start + (end - start) / 2
    k["x"] = mid.dt.year + (mid.dt.dayofyear - 1) / 365.25
    return k.sort_values(["speaker_id", "slot"]).reset_index(drop=True)


def _runs(g: pd.DataFrame) -> list[pd.DataFrame]:
    """One member's points split where a Knesset is missing."""
    brk = (g["slot"].diff() != 1).cumsum()
    return [r for _, r in g.groupby(brk)]


def draw_careers_israel_palestine(tables: T.Tables) -> list[Path]:
    pts = _career_points(tables)
    per = pts.groupby("speaker_id").agg(n=("slot", "size"), first=("estimate", "first"), last=("estimate", "last"))
    threads = set(per.index[per["n"] >= 2])
    n = n_sentences_drawn("israel_palestine", pts[pts["speaker_id"].isin(threads | {sid for sid, _ in CAREERS})])
    title = "A few leaders moved far on the conflict; most members ended close to where they began"
    sub = ("Each grey line is one member's position on the conflict, Knesset by Knesset; six careers picked out. "
           f"{B.from_sentences(n).capitalize()}")
    named_blocs = set(pts.loc[pts["speaker_id"].isin([s for s, _ in CAREERS]), "bloc"])
    key = [(b, c) for b, c in B.BLOCS if b in named_blocs]
    page = B.BlogPage(title, sub, B.sources(B.SRC_CORPUS, *SRC_POSITIONS), key=key, key_lead="Party's bloc:",
                      key_spacing=KEY_SPACING)
    gutter = page.width_of("more hawkish", B.T_NOTE, serif=True, italic=True) + 0.08
    col_gap = 0.30
    pw = (page.inner - gutter - col_gap) / 2
    title_h, row_gap, xtick_h = 0.27, 0.20, 0.26
    room = 7.5 - page.top - page.foot_h
    ph = min(1.45, (room - 3 * title_h - 2 * row_gap - xtick_h) / 3)
    page.set_body(3 * (title_h + ph) + 2 * row_gap + xtick_h)
    xlim = (1986.8, 2031.2)                     # room for the end values
    ylim = (-0.45, 10.45)
    segs = [np.column_stack([r["x"], r["estimate"]])
            for _, g in pts[pts["speaker_id"].isin(threads)].groupby("speaker_id") for r in _runs(g) if len(r) >= 2]
    for i, (sid, name) in enumerate(CAREERS):
        row, col = divmod(i, 2)
        left = page.M + gutter + col * (pw + col_gap)
        top = page.top + row * (title_h + ph + row_gap) + title_h
        ax = page.axes(left, top, pw, ph)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_yticks([])
        for v in (0, 5, 10):
            ax.axhline(v, color=B.GRID, lw=0.6, zorder=0)
        if row == 2:
            ax.set_xticks([1992, 2006, 2022], ["1992", "2006", "2022"], fontsize=B.T_TICK, color=B.CAP, family=B.SANS)
            ax.tick_params(axis="x", pad=3)
        else:
            ax.set_xticks([])
        ax.add_collection(LineCollection(segs, colors=B.CONTEXT, linewidths=0.6, alpha=0.42, zorder=1))
        g = pts[pts["speaker_id"] == sid]
        line = dict(solid_capstyle="round", solid_joinstyle="round")
        for r in _runs(g):
            ax.plot(r["x"], r["estimate"], color=B.PAPER, lw=5.0, zorder=3, **line)
        for r in _runs(g):
            ax.plot(r["x"], r["estimate"], color=B.INK, lw=2.4, zorder=4, **line)
        ax.scatter(g["x"], g["estimate"], s=5.5 ** 2, c=[B.BLOC.get(b, B.OTHER) for b in g["bloc"]],
                   edgecolors=B.PAPER, linewidths=0.7, zorder=5)
        value = dict(va="center", fontsize=B.T_VALUE, weight=600, family=B.SANS, color=B.INK, path_effects=B.HALO,
                     zorder=6)
        v0, v1 = g["estimate"].iloc[0], g["estimate"].iloc[-1]
        ax.text(g["x"].iloc[0] - 1.5, v0, f"{v0:.1f}", ha="right", **value)
        ax.text(g["x"].iloc[-1] + 1.5, v1, f"{v1:.1f}", ha="left", **value)
        page.text(left, top - 0.08, name, size=12, weight=600, serif=True)
        if col == 0:
            for v, word in ((10, "more hawkish"), (0, "more dovish")):
                page.note(left - 0.08, top + ph * (ylim[1] - v) / (ylim[1] - ylim[0]), word, ha="right", va="center")

    path = {sid: pts.loc[pts["speaker_id"] == sid, "estimate"] for sid, _ in CAREERS}
    ends = lambda sid: f"from {path[sid].iloc[0]:.1f} to {path[sid].iloc[-1]:.1f}"  # noqa: E731
    net = path["965"]
    three = per[per["n"] >= 3]
    share = ((three["last"] - three["first"]).abs() <= 2).sum() / len(three)
    in_words = "about six in seven" if abs(share - 6 / 7) < 0.02 else f"{share:.0%}"
    alt = ("Six small charts, each showing every member's career on the conflict in grey with one member in black: "
           f"Ariel Sharon falls {ends('482')}, Tzipi Livni {ends('469')} and Ehud Olmert {ends('1064')} as they leave "
           f"or split Likud, Binyamin Netanyahu dips from {net.iloc[0]:.1f} to {net.min():.1f} and returns to "
           f"{net.iloc[-1]:.1f}, and Ahmad Tibi and Merav Michaeli barely move. Most members end close to where they "
           f"began: {in_words} of those who sat for three or more Knessets finished within two points of their first "
           "position.")
    return page.save("careers_israel_palestine", alt=alt)


CHARTS = {"chamber_lrgen": draw_chamber_lrgen, "careers_israel_palestine": draw_careers_israel_palestine}
