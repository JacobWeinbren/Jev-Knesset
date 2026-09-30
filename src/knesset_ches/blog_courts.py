"""Charts on the courts, and on how Likud and Yisrael Beiteinu moved.

    python -m knesset_ches.blog courts
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from matplotlib.patches import Polygon, Rectangle

from knesset_ches import blog_style as B
from knesset_ches import tables as T

BLUE, ORANGE, YELLOW = B.BLOC["Right"], B.BLOC["Orthodox"], B.BLOC["Secular Centre"]
SRC_OWN_COURTS = "Jev model; the courts scale is the project's own item"
R_DOT = 6.5 / 2 / 72                    # dot radius in inches
CAP_H = 0.70                            # sans cap height as a share of font size


def est(t: T.Tables, dimension: str, actor_type: str, period_type: str) -> pd.DataFrame:
    """Hierarchical group estimates that pass the sufficiency check."""
    g = t.group
    return g[(g["estimator"] == "hierarchical") & (g["actor_type"] == actor_type) & (g["period_type"] == period_type)
             & (g["dimension"] == dimension) & g["sufficient"].fillna(False).astype(bool)]


def members(t: T.Tables, dims: Sequence[str], period: str, faction: str | None = None,
            blocs: Sequence[str] | None = None) -> list[str]:
    m = t.mk
    s = m[(m["period_type"] == "knesset") & (m["period"].astype(str) == str(period)) & m["dimension"].isin(list(dims))]
    if faction is not None:
        s = s[s["faction_name"].astype(str) == faction]
    if blocs is not None:
        s = s[s["bloc"].astype(str).isin(list(blocs))]
    return sorted(set(s["speaker_id"].astype(str)))


def n_for(dims: Sequence[str], groups: Sequence[tuple[Sequence[str], str]]) -> int:
    """Sentences behind the estimates drawn."""
    return B.sentences_behind(dims, [(s, k) for ids, k in groups for s in ids])


def page_on_top(*args: Any, **kw: Any) -> B.BlogPage:
    page = B.BlogPage(*args, **kw)
    page.bg.set_zorder(10)
    return page


class Frame:
    """Maps an axes' data coordinates to page inches."""

    def __init__(self, left: float, top: float, width: float, height: float, xlim: tuple[float, float],
                 ylim: tuple[float, float]):
        self.left, self.top, self.width, self.height, self.xlim, self.ylim = left, top, width, height, xlim, ylim

    def X(self, x: float) -> float:
        return self.left + (x - self.xlim[0]) / (self.xlim[1] - self.xlim[0]) * self.width

    def Y(self, y: float) -> float:
        return self.top + (self.ylim[1] - y) / (self.ylim[1] - self.ylim[0]) * self.height


class Obstacles:
    """What is already on the page, in inches, for labels to avoid."""

    def __init__(self) -> None:
        self.segs: list[tuple[float, float, float, float, float]] = []
        self.dots: list[tuple[float, float]] = []
        self.boxes: list[tuple[float, float, float, float]] = []

    def line(self, xs: Sequence[float], ys: Sequence[float], lw_pt: float) -> None:
        for (x0, y0), (x1, y1) in zip(zip(xs, ys), list(zip(xs, ys))[1:]):
            self.segs.append((x0, y0, x1, y1, lw_pt / 2 / 72))

    def dot(self, x: float, y: float) -> None:
        self.dots.append((x, y))

    def box(self, b: tuple[float, float, float, float]) -> None:
        self.boxes.append(b)

    @staticmethod
    def _seg_box(seg: tuple[float, float, float, float, float], b: tuple[float, float, float, float],
                 pad: float) -> bool:
        """Whether a thick segment crosses a box (Liang-Barsky clipping)."""
        x0, y0, x1, y1, half = seg
        bx0, by0, bx1, by1 = b[0] - half - pad, b[1] - half - pad, b[2] + half + pad, b[3] + half + pad
        dx, dy = x1 - x0, y1 - y0
        t0, t1 = 0.0, 1.0
        for p, q in ((-dx, x0 - bx0), (dx, bx1 - x0), (-dy, y0 - by0), (dy, by1 - y0)):
            if abs(p) < 1e-12:
                if q < 0:
                    return False
                continue
            r = q / p
            if p < 0:
                t0 = max(t0, r)
            else:
                t1 = min(t1, r)
            if t0 > t1:
                return False
        return True

    def clearance(self, b: tuple[float, float, float, float]) -> float:
        """Smallest gap between `b` and anything drawn, negative if they overlap."""
        x0, y0, x1, y1 = b

        def to_box(px: float, py: float) -> float:
            return math.hypot(max(x0 - px, 0, px - x1), max(y0 - py, 0, py - y1))

        gaps = [to_box(cx, cy) - R_DOT for cx, cy in self.dots]
        for c in self.boxes:
            gaps.append(max(c[0] - x1, x0 - c[2], c[1] - y1, y0 - c[3]))
        for sx0, sy0, sx1, sy1, half in self.segs:
            n = max(2, int(math.hypot(sx1 - sx0, sy1 - sy0) / 0.004))
            gaps.append(min(to_box(sx0 + (sx1 - sx0) * i / n, sy0 + (sy1 - sy0) * i / n) for i in range(n + 1)) - half)
        return min(gaps) if gaps else 1.0

    def hits(self, b: tuple[float, float, float, float], pad: float = 0.015) -> bool:
        x0, y0, x1, y1 = b
        for c in self.boxes:
            if min(x1, c[2]) - max(x0, c[0]) > -pad and min(y1, c[3]) - max(y0, c[1]) > -pad:
                return True
        for cx, cy in self.dots:
            dx, dy = max(x0 - cx, 0, cx - x1), max(y0 - cy, 0, cy - y1)
            if math.hypot(dx, dy) < R_DOT + pad:
                return True
        return any(self._seg_box(s, b, pad) for s in self.segs)


def place_value(page: B.BlogPage, obs: Obstacles, X: float, Y: float, s: str, prefer: Sequence[str],
                r: float = R_DOT, search: float = 0.0) -> None:
    """Label the mark at (X, Y) with `s` on the first free side in `prefer`."""
    size, weight = B.T_VALUE, 600
    w = page.width_of(s, size, weight=weight)
    cap = CAP_H * size / 72
    d, k = r + 0.035, 0.75
    opts = {"right": (X + d, Y + cap / 2, "left"), "left": (X - d, Y + cap / 2, "right"),
            "above": (X, Y - d, "center"), "below": (X, Y + d + cap, "center"),
            "ur": (X + d * k, Y - d * k, "left"), "ul": (X - d * k, Y - d * k, "right"),
            "lr": (X + d * k, Y + d * k + cap, "left"), "ll": (X - d * k, Y + d * k + cap, "right")}
    steps = [0.0, 0.015, -0.015, 0.03, -0.03, 0.045, -0.045, 0.06, -0.06]

    def cands(name: str):
        x, base, ha = opts[name]
        for st in steps:
            yield (x + st, base, ha) if name in ("above", "below") else (x, base + st, ha)

    def box_of(c: tuple[float, float, str]) -> tuple[float, float, float, float]:
        x, base, ha = c
        x0 = x if ha == "left" else x - w if ha == "right" else x - w / 2
        return (x0, base - cap, x0 + w, base)

    def nudged():
        for side in ("left", "right"):
            for g in np.arange(0.02, search + 1e-9, 0.01):
                for dy in np.arange(-0.08, 0.0801, 0.005):
                    x0 = X - r - g - w if side == "left" else X + r + g
                    b = (x0, Y + dy - cap / 2, x0 + w, Y + dy + cap / 2)
                    yield obs.clearance(b) - 0.25 * g - 0.1 * abs(dy), b

    chosen = None
    for pad in (0.015, 0.006):
        chosen = next((c for n in prefer for c in cands(n) if not obs.hits(box_of(c), pad)), None)
        if chosen:
            break
    if chosen is None and search:
        b = max(nudged(), key=lambda cb: cb[0])[1]
        chosen = (b[0], b[3], "left")
    chosen = chosen or next(cands(prefer[0]))
    x, base, ha = chosen
    obs.box(box_of(chosen))
    page.text(x, base, s, size=size, colour=B.INK, ha=ha, weight=weight, halo=True)


def scale_labels(page: B.BlogPage, f: Frame, x: float, hi: str, lo: str) -> None:
    for v, word in ((10, "↑ " + hi), (5, None), (0, "↓ " + lo)):
        t = page.text(x, f.Y(v) - 0.05, str(v), size=B.T_TICK, colour=B.CAP)
        if word:
            page.note(x + page.width_of(t) + 0.08, f.Y(v) - 0.05, word)


def spread_even(centres: Sequence[float], ups: Sequence[float], downs: Sequence[float], lo: float,
                hi: float) -> list[float]:
    """Push touching label blocks on a vertical line apart evenly, keeping their order within [lo, hi]."""
    order = sorted(range(len(centres)), key=lambda i: centres[i])

    def solve(mem: list[int]) -> tuple[list[float], float]:
        offs = [0.0]
        for a, b in zip(mem, mem[1:]):
            offs.append(offs[-1] + downs[a] + ups[b])
        pos = float(np.mean([centres[m] - o for m, o in zip(mem, offs)]))
        top, bot = pos - ups[mem[0]], pos + offs[-1] + downs[mem[-1]]
        pos += (lo - top) if top < lo else (hi - bot) if bot > hi else 0.0
        return offs, pos

    clusters: list[tuple[list[int], list[float], float]] = []
    for i in order:
        clusters.append(([i], *solve([i])))
        while len(clusters) > 1:
            (m1, o1, p1), (m2, o2, p2) = clusters[-2], clusters[-1]
            if p1 + o1[-1] + downs[m1[-1]] <= p2 - ups[m2[0]] + 1e-9:
                break
            clusters[-2:] = [(m1 + m2, *solve(m1 + m2))]
    out = [0.0] * len(centres)
    for mem, offs, pos in clusters:
        for m, o in zip(mem, offs):
            out[m] = pos + o
    return out


BLOCS = [b for b, _ in B.BLOCS]


def draw_courts_by_bloc(t: T.Tables) -> list[Path]:
    d = est(t, "judicial_power", "bloc", "knesset")
    d = d[d["actor_id"].isin(BLOCS)]
    val = {(r.actor_id, r.period): float(r.estimate) for r in d.itertuples()}
    nmk = {(r.actor_id, r.period): int(r.n_mks) for r in d.itertuples()}
    slots = B.KNESSET_SLOTS
    first, last = slots[0], slots[-1]
    groups = [(members(t, ["judicial_power"], k, blocs=[b for b in BLOCS if (b, k) in val]), k) for k in slots]
    n = n_for(["judicial_power"], groups)

    title = "Right and Left once sounded alike on the courts; now they stand at opposite ends"
    sub = ("Average position of each bloc's members on the power of the courts, Knesset by Knesset. "
           f"{B.from_sentences(n).capitalize()}")
    src = B.sources(B.SRC_CORPUS, SRC_OWN_COURTS, partial=True)
    thin = any(nmk[(b, k)] < 5 for b in BLOCS for k in slots if (b, k) in val)
    page = page_on_top(title, sub, src, key=[("fewer than five members", B.INK, "hollow")] if thin else None)

    ends = {b: val[(b, last)] for b in BLOCS if (b, last) in val}
    names = {b: (["Secular", "Centre"] if b == "Secular Centre" else [b]) for b in ends}
    name_w = {b: page.width_of(names[b][-1], B.T_LABEL, weight=500) for b in ends}
    val_w = {b: page.width_of(f"{v:.1f}", B.T_VALUE, weight=600) for b, v in ends.items()}
    lab_w = max(max(name_w[b] + 0.07 + val_w[b], page.width_of(names[b][0], B.T_LABEL, weight=500)) for b in ends)
    left = page.M
    lab_gap = R_DOT + 0.06
    width = (page.W - page.M - left - lab_gap - lab_w) * 11 / 10.5
    top, height = page.top + 0.30, 3.0
    ax = page.axes(left, top, width, height)
    xs = B.knesset_axis(ax)
    ax.set_ylim(0, 10)
    for v in (0, 2.5, 5, 7.5, 10):
        ax.axhline(v, color=B.INK if v == 0 else B.GRID, lw=0.9 if v == 0 else 0.6, zorder=0 if v else 2)
    ax.set_yticks([])
    ax.tick_params(axis="x", pad=6)
    f = Frame(left, top, width, height, ax.get_xlim(), (0, 10))
    ax.axvspan(xs["24"] - 0.5, xs["24"] + 0.5, color=B.SHADE, lw=0, zorder=-1)

    for z, b in enumerate(["Arab-Israeli", "Secular Centre", "Orthodox", "Left", "Right"]):
        col = B.BLOC[b]
        pts = [(xs[k], val[(b, k)], nmk[(b, k)]) for k in slots if (b, k) in val]
        runs, run = [], []
        for i, (x, y, m) in enumerate(pts):
            joined = i > 0 and x - pts[i - 1][0] == 1 and m >= 4 and pts[i - 1][2] >= 4
            if not joined and run:
                runs.append(run)
                run = []
            run.append((x, y))
        runs.append(run)
        lw = B.LW_STORY if b in ("Right", "Left") else 2.0
        for r in runs:
            if len(r) > 1:
                ax.plot([p[0] for p in r], [p[1] for p in r], color=col, lw=lw, solid_capstyle="round",
                        solid_joinstyle="round", zorder=3 + z)
        for x, y, m in pts:
            if m >= 5:
                ax.scatter([x], [y], s=B.DOT_S, color=col, edgecolors=B.PAPER, linewidths=0.8, zorder=3.5 + z)
            else:
                ax.scatter([x], [y], s=B.DOT_S, facecolors=B.PAPER, edgecolors=col, linewidths=1.4, zorder=20 + z)

    order = list(ends)
    lh = 0.19
    ys = spread_even([f.Y(ends[b]) for b in order], [lh / 2 + lh * (len(names[b]) - 1) for b in order],
                     [lh / 2] * len(order), f.Y(10) - 0.1, f.Y(0) - 0.02)
    x_lab = f.X(xs[last]) + lab_gap
    cap = CAP_H * B.T_LABEL / 72
    for b, y in zip(order, ys):
        base = y + cap / 2
        colour = B.INK if b == "Secular Centre" else B.BLOC[b]
        for j, line in enumerate(names[b]):
            page.text(x_lab, base - (len(names[b]) - 1 - j) * lh, line, size=B.T_LABEL, weight=500, colour=colour,
                      halo=True)
        page.text(x_lab + name_w[b] + 0.07, base, f"{ends[b]:.1f}", size=B.T_VALUE, weight=600, colour=B.INK, halo=True)
    scale_labels(page, f, f.X(-0.5), "curb the courts", "defend the courts")
    page.note(f.X(xs["24"]), f.Y(10) - 0.08, "Bennett–Lapid government", ha="center")
    page.fit(top + height + 0.32)

    now = {b: val[(b, last)] for b in BLOCS}
    alt = (f"Line chart of the five blocs on the power of the courts, 1992–96 to 2022–26. In 1992–96 the Right "
           f"({val[('Right', first)]:.1f}) and the Left ({val[('Left', first)]:.1f}) sat together in the middle, with "
           f"the Orthodox ({val[('Orthodox', first)]:.1f}) already for curbing the courts. By 2022–26 the Right "
           f"({now['Right']:.1f}) and Orthodox ({now['Orthodox']:.1f}) are near the top and the Left "
           f"({now['Left']:.1f}), Arab-Israeli ({now['Arab-Israeli']:.1f}) and Secular Centre "
           f"({now['Secular Centre']:.1f}) near the bottom.")
    return page.save("courts_by_bloc", alt=alt)


DODGE, LEAD = 0.15, 0.13              # side step for merged end markers, room for leaders


def dodge(by_height: list[str], y: dict[str, float], x: float) -> tuple[dict[str, float], list[str]]:
    """Spread end markers that would merge sideways. Returns each marker's x and the first merged group."""
    groups = [[by_height[0]]]
    for q in by_height[1:]:
        if y[q] - y[groups[-1][-1]] < 2 * R_DOT + 0.03:
            groups[-1].append(q)
        else:
            groups.append([q])
    xs = {}
    for grp in groups:
        steps = ([0.0] if len(grp) == 1 else [-DODGE / 2, DODGE / 2] if len(grp) == 2
                 else list(np.linspace(-DODGE, DODGE, len(grp))))
        for q, st in zip(grp, steps):
            xs[q] = x + st
    return xs, next(grp for grp in groups if len(grp) > 1)


POWER_PARTIES = [("Likud", BLUE, "o", "-"), ("Shas", ORANGE, "o", "-"), ("UTJ", ORANGE, "s", (0, (3.2, 1.8))),
                 ("Yesh Atid", YELLOW, "o", "-")]
POWER_COLS = [("21-22-23", "2019–21", ["Netanyahu", "governs"]), ("24", "2021–22", ["Bennett–Lapid", "government"]),
              ("25", "2022–26*", ["Netanyahu", "governs again"])]


def draw_courts_when_power_changed(t: T.Tables) -> list[Path]:
    periods = [p for p, _, _ in POWER_COLS]
    parties = [p for p, *_ in POWER_PARTIES]
    d = est(t, "judicial_power", "party", "knesset")
    d = d[d["period"].isin(periods) & d["actor_name"].isin(parties)]
    val = {(r.actor_name, r.period): float(r.estimate) for r in d.itertuples()}
    gov = {(r.actor_name, r.period): float(r.coalition_share) >= 0.5 for r in d.itertuples()}
    ids = {k: sorted(set().union(*(members(t, ["judicial_power"], k, faction=p) for p in parties))) for k in periods}
    n = n_for(["judicial_power"], [(ids[k], k) for k in periods])

    title = "When power changed hands in 2021, the parties' talk of the courts changed with it"
    sub = f"Each party's average position in three Knessets. {B.from_sentences(n).capitalize()}"
    src = B.sources(B.SRC_CORPUS, SRC_OWN_COURTS, partial=True)
    page = page_on_top(title, sub, src, key=[("in government", B.INK, "dot"), ("in opposition", B.INK, "hollow")])
    colour_of = {q: (B.INK if q == "Yesh Atid" else c) for q, c, *_ in POWER_PARTIES}
    names = {p: (["United Torah", "Judaism"] if p == "UTJ" else [T.display_name(p)]) for p in parties}
    name_w = {p: max(page.width_of(s, B.T_LABEL, weight=500) for s in names[p]) for p in names}
    vw = page.width_of("8.8", B.T_VALUE, weight=600)
    g_left = R_DOT + 0.05 + vw + 0.07 + max(name_w.values())
    g_right = DODGE + R_DOT + LEAD + vw + 0.07 + max(name_w.values())
    x0, x2 = page.M + g_left, page.W - page.M - g_right
    S = (x2 - x0) / 2
    colx = [x0, x0 + S, x2]

    head_top = page.top
    for i, (_, yrs, words) in enumerate(POWER_COLS):
        page.text(colx[i], head_top + 0.15, yrs, size=B.T_PANEL, weight=600, serif=True, ha="center")
        for j, w in enumerate(words):
            page.note(colx[i], head_top + 0.36 + j * 0.17, w, ha="center")
    ptop = head_top + 0.36 + 0.17 + 0.3
    ph = min(4.8, 1.3 * page.W - page.foot_h - ptop - 0.04)     # as tall as a 1.3 x width page allows
    Y = lambda v: ptop + (10 - v) / 10 * ph  # noqa: E731
    page.bg.add_patch(Rectangle((colx[1] - S * 0.42, head_top - 0.06), S * 0.84, ptop + ph + 0.06 - (head_top - 0.06),
                                fc=B.SHADE, ec="none", zorder=0))
    for v in (0, 5, 10):
        page.bg.plot([page.M, page.W - page.M], [Y(v), Y(v)], color=B.INK if v == 0 else B.GRID,
                     lw=0.9 if v == 0 else 0.6, zorder=1, solid_capstyle="butt")
    scale_labels(page, Frame(page.M, ptop, page.inner, ph, (0, 1), (0, 10)), page.M, "curb the courts",
                 "defend the courts")

    last = periods[2]
    endx, cluster = dodge(sorted(parties, key=lambda q: -val[(q, last)]), {q: Y(val[(q, last)]) for q in parties},
                          colx[2])

    obs = Obstacles()
    pts: dict[str, list[tuple[float, float]]] = {}
    for p, col, marker, ls in POWER_PARTIES:
        xs = [colx[0], colx[1], endx[p]]
        ys = [Y(val[(p, k)]) for k in periods]
        pts[p] = list(zip(xs, ys))
        page.bg.plot(xs, ys, color=col, lw=B.LW_DATA, ls=ls, solid_capstyle="round", dash_capstyle="butt", zorder=3)
        obs.line(xs, ys, B.LW_DATA)
        for x, y, k in zip(xs, ys, periods):
            page.bg.scatter([x], [y], s=B.DOT_S * (1.0 if marker == "o" else 0.85), marker=marker, linewidths=1.4,
                            facecolors=col if gov[(p, k)] else B.PAPER, edgecolors=colour_of[p], zorder=4)
            obs.dot(x, y)

    lh, lh2 = 0.19, 0.175                   # label pitch, line pitch within a name
    cap = CAP_H * B.T_LABEL / 72
    # at the left, two-line names wrap upwards to keep the value level with the dot
    k = periods[0]
    ys = spread_even([Y(val[(p, k)]) for p in parties], [lh / 2 + lh2 * (len(names[p]) - 1) for p in parties],
                     [lh / 2] * len(parties), ptop - 0.05, Y(0) - 0.25)
    for p, y in zip(parties, ys):
        base = y + cap / 2
        xv = colx[0] - R_DOT - 0.05
        page.text(xv, base, f"{val[(p, k)]:.1f}", size=B.T_VALUE, weight=600, halo=True, ha="right")
        xn = xv - vw - 0.07
        for j, s in enumerate(names[p]):
            page.text(xn, base - (len(names[p]) - 1 - j) * lh2, s, size=B.T_LABEL, weight=500, colour=colour_of[p],
                      halo=True, ha="right")
        obs.box((xn - name_w[p], base - cap - lh2 * (len(names[p]) - 1), colx[0] - R_DOT, base))

    # at the right, the merged cluster gets a stacked list with leaders
    lead_x = colx[2] + DODGE + R_DOT + LEAD
    for p in [q for q in parties if q not in cluster]:
        base = Y(val[(p, last)]) + cap / 2
        xv = endx[p] + R_DOT + 0.05
        page.text(xv, base, f"{val[(p, last)]:.1f}", size=B.T_VALUE, weight=600, halo=True)
        page.text(xv + vw + 0.07, base, names[p][0], size=B.T_LABEL, weight=500, colour=colour_of[p], halo=True)
        obs.box((endx[p] + R_DOT, base - cap, xv + vw + 0.07 + name_w[p], base))
    ys = spread_even([Y(val[(q, last)]) for q in cluster], [lh / 2] * len(cluster),
                     [lh / 2 + lh2 * (len(names[q]) - 1) for q in cluster], ptop - 0.05, Y(0) - 0.25)
    others = {}
    for q in cluster:
        o = Obstacles()
        for q2 in parties:
            if q2 != q:
                o.dot(endx[q2], Y(val[(q2, last)]))
                o.line([x for x, _ in pts[q2]], [y for _, y in pts[q2]], B.LW_DATA)
        others[q] = o

    def leader(q: str, yl: float) -> tuple[float, float, float, float, float]:
        """Leader from marker to label, and its clearance from other marks."""
        mx, my = endx[q], Y(val[(q, last)])
        ex, ey = lead_x - 0.04, yl
        d = math.hypot(ex - mx, ey - my)
        sx, sy = mx + (ex - mx) / d * (R_DOT + 0.025), my + (ey - my) / d * (R_DOT + 0.025)
        along = [(sx + (ex - sx) * i / 40, sy + (ey - sy) * i / 40) for i in range(41)]
        return sx, sy, ex, ey, min(others[q].clearance((px, py, px, py)) for px, py in along) - 0.9 / 2 / 72

    # nudge the stack to keep leaders clear of other markers
    shift = max(np.arange(-0.10, 0.1001, 0.01),
                key=lambda sh: min(leader(q, y + sh)[4] for q, y in zip(cluster, ys)) - 0.1 * abs(sh))
    ys = [y + float(shift) for y in ys]
    boxes = []
    for q, y in zip(cluster, ys):
        base = y + cap / 2
        page.text(lead_x, base, f"{val[(q, last)]:.1f}", size=B.T_VALUE, weight=600, halo=True)
        for j, s in enumerate(names[q]):
            page.text(lead_x + vw + 0.07, base + j * lh2, s, size=B.T_LABEL, weight=500, colour=colour_of[q], halo=True)
        boxes.append((lead_x, base - cap, lead_x + vw + 0.07 + name_w[q], base + lh2 * (len(names[q]) - 1)))
        sx, sy, ex, ey, _ = leader(q, y)
        page.bg.plot([sx, ex], [sy, ey], color=B.RULE, lw=0.9, solid_capstyle="round", zorder=2)
        obs.line([sx, ex], [sy, ey], 0.9)
    for b in boxes:
        obs.box(b)

    mid = periods[1]
    for p in sorted(parties, key=lambda q: val[(q, mid)]):
        valley = val[(p, mid)] < min(val[(p, periods[0])], val[(p, periods[2])])
        prefer = (["below", "ll", "lr", "left", "right", "above", "ul", "ur"] if valley
                  else ["above", "ul", "ur", "left", "right", "below", "ll", "lr"])
        place_value(page, obs, colx[1], Y(val[(p, mid)]), f"{val[(p, mid)]:.1f}", prefer, search=0.14)
    page.fit(Y(0) + 0.02)

    L, Sh, U, YA = ([val[(p, k)] for k in periods] for p in ("Likud", "Shas", "UTJ", "Yesh Atid"))
    alt = (f"Slope chart of four parties across three Knessets. In opposition in 2021–22, Likud fell from {L[0]:.1f} "
           f"to {L[1]:.1f} on the courts scale, Shas from {Sh[0]:.1f} to {Sh[1]:.1f} and United Torah Judaism from "
           f"{U[0]:.1f} to {U[1]:.1f}, while Yesh Atid, in government, rose from {YA[0]:.1f} to {YA[1]:.1f}. All four "
           f"moved back in 2022–26.")
    return page.save("courts_when_power_changed", alt=alt)


def draw_likud_two_journeys(t: T.Tables) -> list[Path]:
    dims = ["israel_palestine", "judicial_power"]
    series = {}
    for dname in dims:
        d = est(t, dname, "party", "rolling2")
        d = d[d["actor_name"] == "Likud"].assign(x=lambda x: x["period"].astype(int)).sort_values("x")
        series[dname] = d.set_index("x")["estimate"].astype(float)
    cj, ip = series["judicial_power"], series["israel_palestine"]
    first, last = int(min(cj.index.min(), ip.index.min())), int(max(cj.index.max(), ip.index.max()))
    low_x = int(ip[ip.index < 2019].idxmin())
    n = n_for(dims, [(members(t, dims, k, faction="Likud"), k) for k in B.KNESSET_SLOTS])

    title = "Likud swung on the conflict but, over thirty years, turned against the courts"
    sub = f"Average position of Likud members, two-year windows, 1992 to 2026*. {B.from_sentences(n).capitalize()}"
    src = B.sources(B.SRC_CORPUS_SHORT, "Jev model", f"{B.SRC_CHES}, except the courts scale", partial=True)
    page = page_on_top(title, sub, src)
    left = page.M
    width = page.W - page.M - left - (page.width_of("8.8", B.T_VALUE, weight=600) + R_DOT + 0.08)
    ph = 1.75
    xlim = (first - 0.6, last + 0.6)
    panels = [("The conflict", ip, ("more dovish", "more hawkish"),
               [(2005.6, "Gaza disengagement, 2005", "right"), (2009.45, "Bar-Ilan speech, 2009", "left")]),
              ("The courts", cj, ("defend the courts", "curb the courts"),
               [(2023.01, "judicial overhaul, 2023", "right")])]
    y = page.top
    obs = Obstacles()
    for i, (name, s, (lo_w, hi_w), events) in enumerate(panels):
        page.text(page.M, y + 0.16, name, size=B.T_PANEL, weight=600, serif=True)
        top = y + 0.44
        ax = page.axes(left, top, width, ph)
        ax.set_xlim(*xlim)
        ax.set_ylim(0, 10)
        for v in (0, 5, 10):
            ax.axhline(v, color=B.INK if v == 0 else B.GRID, lw=0.9 if v == 0 else 0.6, zorder=2 if v == 0 else 0)
        ax.set_yticks([])
        if i == 0:
            ax.set_xticks([])
        else:
            ax.set_xticks([1995, 2005, 2015, 2026], ["1995", "2005", "2015", "2026*"], fontsize=B.T_TICK, color=B.CAP,
                          family=B.SANS)
            ax.tick_params(axis="x", pad=6)
        f = Frame(left, top, width, ph, xlim, (0, 10))
        scale_labels(page, f, left, hi_w, lo_w)
        for ex, label, ha in events:
            ax.plot([ex, ex], [0, 10], color=B.RULE, lw=1.0, zorder=0)
            obs.line([f.X(ex), f.X(ex)], [f.Y(0), f.Y(10)], 1.0)
            page.note(f.X(ex) + (-0.06 if ha == "right" else 0.06), f.Y(1.9), label, ha=ha, halo=True)
        xs, ys = list(s.index), list(s.values)
        ax.plot(xs, ys, color=BLUE, lw=2.6, solid_capstyle="round", solid_joinstyle="round", zorder=3)
        obs.line([f.X(x) for x in xs], [f.Y(v) for v in ys], 2.6)
        marks = [(first, ["below", "lr", "above"]), (last, ["right"])]
        if s is ip:
            marks.append((low_x, ["below", "ll", "lr"]))
        for x, prefer in marks:
            ax.scatter([x], [s.loc[x]], s=B.DOT_S * 0.75, color=BLUE, edgecolors=B.PAPER, linewidths=0.8, zorder=4)
            place_value(page, obs, f.X(x), f.Y(float(s.loc[x])), f"{float(s.loc[x]):.1f}", prefer, r=R_DOT * 0.87)
        y = top + ph + 0.14
    page.fit(y - 0.14 + 0.3)

    falls = cj.diff()
    run = longest = 0
    for v in falls.iloc[1:]:
        run = run + 1 if v < 0 else 0
        longest = max(longest, run)
    dips = ", with only brief dips" if longest <= 2 and falls.min() > -1 else ""
    alt = (f"Two line charts of Likud, {first} to {last}. On the conflict it fell from {ip.loc[first]:.1f} to "
           f"{ip.loc[low_x]:.1f} by {low_x}, the disengagement era, and climbed back to {ip.loc[last]:.1f}. On the "
           f"courts it rose from {cj.loc[first]:.1f} to {cj.loc[last]:.1f}{dips}.")
    return page.save("likud_two_journeys", alt=alt)


BEITEINU_ROWS = [("judicial_power", "Power of the courts", ("defend courts", "curb courts")),
                 ("democratic_v_jewish_state", "Jewish or democratic state", ("democratic", "Jewish")),
                 ("religious_principles", "Religion in politics", ("secular", "religious")),
                 ("galtan", "Social values", ("liberal", "traditional")),
                 ("israel_palestine", "The conflict", ("dovish", "hawkish"))]
# Knesset 20 only. 19 is too thin, and adding 18 would credit the 2019 break with a secular turn made in 2016-18
BEFORE, AFTER = "20", "25"


def draw_beiteinu_changed_sides(t: T.Tables) -> list[Path]:
    party = "Y. Beiteinu"
    rows = []
    for dname, label, poles in BEITEINU_ROWS:
        a = est(t, dname, "party", "knesset")
        a = a[a["actor_name"] == party].set_index("period")["estimate"]
        rows.append({"dimension": dname, "label": label, "poles": poles, "a": float(a[BEFORE]), "b": float(a[AFTER])})
    r = pd.DataFrame(rows)
    r["move"] = r["b"] - r["a"]
    r = r.reindex(r["move"].abs().sort_values(ascending=False).index).reset_index(drop=True)
    dims = [d for d, *_ in BEITEINU_ROWS]
    n = n_for(dims, [(members(t, dims, k, faction=party), k) for k in (BEFORE, AFTER)])

    title = "Yisrael Beiteinu swung to defending the courts after 2019 but stayed hawkish on the conflict"
    sub = ("Average position of Yisrael Beiteinu members before and after the party broke with Netanyahu in 2019. "
           f"{B.from_sentences(n).capitalize()}")
    src = B.sources(B.SRC_CORPUS_SHORT, "Jev model", f"{B.SRC_CHES}, except the courts scale", partial=True)
    page = page_on_top(title, sub, src, key=[("2015–19, before the break", BLUE, "hollow"),
                                             ("2022–26*, in opposition", BLUE, "left")])
    X0, TW = page.M, page.inner
    X = lambda v: X0 + v / 10 * TW  # noqa: E731
    top = page.top
    for v, ha in ((0, "left"), (5, "center"), (10, "right")):
        page.text(X(v), top + 0.1, str(v), size=B.T_TICK, colour=B.CAP, ha=ha)
    rows_top, RH = top + 0.22, 0.74
    page.bg.plot([X(5), X(5)], [rows_top, rows_top + RH * len(r)], color=B.RULE, lw=0.8, zorder=1)
    note_cap = 0.68 * B.T_NOTE / 72
    cap = CAP_H * B.T_VALUE / 72
    head_w, head_h = 0.12, 0.062
    for i, row in enumerate(r.itertuples()):
        y0 = rows_top + i * RH
        ty = y0 + 0.38
        page.text(X0, y0 + 0.17, row.label, size=B.T_LABEL, weight=500)
        page.bg.plot([X(0), X(10)], [ty, ty], color=B.GRID, lw=1.2, zorder=1, solid_capstyle="butt")
        lo_w, hi_w = row.poles
        page.note(X(0), ty + 0.1 + note_cap, lo_w)
        page.note(X(10), ty + 0.1 + note_cap, hi_w, ha="right")
        # shaft only where it fits between ring and arrowhead
        xa, xb = X(row.a), X(row.b)
        sgn = 1 if row.b >= row.a else -1
        base_x = xb - sgn * head_w
        if abs(xa - base_x) > R_DOT and (base_x - xa) * sgn > 0:
            page.bg.plot([xa + sgn * R_DOT, base_x], [ty, ty], color=BLUE, lw=B.LW_DATA, zorder=3,
                         solid_capstyle="butt")
        page.bg.add_patch(Polygon([(xb, ty), (base_x, ty - head_h), (base_x, ty + head_h)], closed=True, fc=BLUE,
                                  ec="none", zorder=4))
        page.bg.scatter([xa], [ty], s=B.DOT_S, facecolors=B.PAPER, edgecolors=BLUE, linewidths=1.4, zorder=5)
        page.text(xa - sgn * (R_DOT + 0.06), ty + cap / 2, f"{row.a:.1f}", size=B.T_VALUE, weight=600, colour=B.SEC,
                  ha="left" if sgn < 0 else "right", halo=True)
        page.text(xb + sgn * 0.06, ty + cap / 2, f"{row.b:.1f}", size=B.T_VALUE, weight=600, colour=B.INK,
                  ha="right" if sgn < 0 else "left", halo=True)
    page.fit(rows_top + RH * len(r) - 0.08)

    rv = r.set_index("dimension")
    courts, conflict = rv.loc["judicial_power"], rv.loc["israel_palestine"]
    others = rv.drop(["judicial_power", "israel_palestine"])["move"].abs()
    alt = (f"Arrow chart of Yisrael Beiteinu on five scales, 2015–19 against 2022–26. After breaking with Netanyahu it "
           f"swung from {courts['a']:.1f} to {courts['b']:.1f} on the courts, towards defending them, but stayed "
           f"hawkish on the conflict ({conflict['a']:.1f} to {conflict['b']:.1f}); on religion, the Jewish or "
           f"democratic state and social values it moved {others.min():.1f} to {others.max():.1f} points towards the "
           f"liberal end.")
    return page.save("beiteinu_changed_sides", alt=alt)


CHARTS = {
    "courts_by_bloc": draw_courts_by_bloc,
    "courts_when_power_changed": draw_courts_when_power_changed,
    "likud_two_journeys": draw_likud_two_journeys,
    "beiteinu_changed_sides": draw_beiteinu_changed_sides,
}
