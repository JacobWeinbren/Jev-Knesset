"""The opening charts: the 25th Knesset map, the model against the experts, polarisation and the Left's voters.

    python -m knesset_ches.blog opening
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from matplotlib.transforms import Bbox, ScaledTranslation

from knesset_ches import blog_style as B
from knesset_ches import tables as T

EXTERNAL = T.ROOT / "data" / "external"
DOT_R_PT = 3.25                          # radius of a 6.5 pt dot
RINGS = [12, 18, 24, 32, 40, 50, 62, 76, 92]    # label distances in points


class Placer:
    """Puts labels in free spots near their points."""

    def __init__(self, page: B.BlogPage, ax: Any, dots: np.ndarray, pad_pt: float = 1.6):
        self.ax, self.fig = ax, page.fig
        self.r = self.fig.canvas.get_renderer()
        self.px = self.fig.dpi / 72
        self.dots = ax.transData.transform(np.asarray(dots, float))
        self.dot_r = (DOT_R_PT + pad_pt) * self.px
        self.boxes: list[Bbox] = []
        self.area = ax.get_window_extent(self.r)
        self.gap = 0.035 * self.fig.dpi

    def _hits_dots(self, bb: Bbox) -> bool:
        cx = np.clip(self.dots[:, 0], bb.x0, bb.x1)
        cy = np.clip(self.dots[:, 1], bb.y0, bb.y1)
        return bool((np.hypot(self.dots[:, 0] - cx, self.dots[:, 1] - cy) < self.dot_r).any())

    def _segment_hits(self, a: np.ndarray, b: np.ndarray) -> int:
        """Dots and labels that a leader from a to b crosses."""
        hits = 0
        ab = b - a
        L2 = float(ab @ ab) or 1.0
        for p in self.dots:
            if np.hypot(*(p - a)) < 1e-6:
                continue
            t = np.clip(((p - a) @ ab) / L2, 0, 1)
            if np.hypot(*(a + t * ab - p)) < self.dot_r * 0.8:
                hits += 1
        for bb in self.boxes:
            for t in np.linspace(0, 1, 25):
                q = a + t * ab
                if bb.x0 <= q[0] <= bb.x1 and bb.y0 <= q[1] <= bb.y1:
                    hits += 5
                    break
        return hits

    def free(self, bb: Bbox) -> bool:
        a = self.area
        if bb.x0 < a.x0 or bb.x1 > a.x1 or bb.y0 < a.y0 or bb.y1 > a.y1:
            return False
        g = self.gap
        for o in self.boxes:
            if min(bb.x1, o.x1) - max(bb.x0, o.x0) > -g and min(bb.y1, o.y1) - max(bb.y0, o.y0) > -g:
                return False
        return not self._hits_dots(bb)

    @staticmethod
    def candidates(arc: tuple[float, float] = (0, 360)) -> list[tuple[float, float, str, str]]:
        """Offsets in points on each ring within `arc`, in degrees."""
        out = []
        lo, hi = arc
        for r in RINGS:
            for k in range(36):
                deg = 360 * k / 36
                if not (lo <= deg <= hi or lo <= deg + 360 <= hi):
                    continue
                a = np.radians(deg)
                dx, dy = r * np.cos(a), r * np.sin(a)
                va = "bottom" if dy > 0.3 * r else ("top" if dy < -0.3 * r else "center")
                if dx > 0.3 * r:
                    out.append((dx, dy, "left", va))
                elif dx < -0.3 * r:
                    out.append((dx, dy, "right", va))
                else:                                     # above or below, also try either side
                    out += [(dx, dy, "center", va), (dx + 4, dy, "right", va), (dx - 4, dy, "left", va)]
        return out

    def _text(self, x: float, y: float, s: str, dx: float, dy: float, ha: str, va: str, **kw: Any):
        # Text rather than Annotation, whose extent includes the arrow
        off = self.ax.transData + ScaledTranslation(dx / 72, dy / 72, self.fig.dpi_scale_trans)
        return self.ax.text(x, y, B.visual(s), transform=off, ha=ha, va=va, path_effects=B.HALO, **kw)

    def place(self, x: float, y: float, s: str, arc: tuple[float, float], leader_from: float, **kw: Any) -> None:
        """Label a point, in `arc` if there is room."""
        for cands in (self.candidates(arc), self.candidates()):
            if self._place(x, y, s, cands, leader_from, **kw):
                return
        raise RuntimeError(f"no room for {s}")

    def _place(self, x: float, y: float, s: str, cands: Sequence[tuple[float, float, str, str]], leader_from: float,
               **kw: Any) -> bool:
        own = self.ax.transData.transform((x, y))
        best = None
        for dx, dy, ha, va in cands:
            dist = float(np.hypot(dx, dy))
            t = self._text(x, y, s, dx, dy, ha, va, **kw)
            bb = t.get_window_extent(self.r)
            t.remove()
            trim = 0.14 * bb.height
            bb = Bbox.from_extents(bb.x0, bb.y0 + trim, bb.x1, bb.y1 - trim)
            if not self.free(bb):
                continue
            cost = dist
            if dist > leader_from:
                q = np.array([np.clip(own[0], bb.x0, bb.x1), np.clip(own[1], bb.y0, bb.y1)])
                cost += 40 * self._segment_hits(own, q)
            if best is None or cost < best[0]:
                best = (cost, dx, dy, ha, va, bb, dist)
            if cost <= dist and best[0] <= dist:
                break
        if best is None:
            return False
        _, dx, dy, ha, va, bb, dist = best
        self._text(x, y, s, dx, dy, ha, va, **kw)
        if dist > leader_from:
            q = np.array([np.clip(own[0], bb.x0 - 2 * self.px, bb.x1 + 2 * self.px),
                          np.clip(own[1], bb.y0 - 1 * self.px, bb.y1 + 1 * self.px)])
            v = q - own
            L = float(np.hypot(*v))
            if L > (DOT_R_PT + 2.5) * self.px:
                p0 = own + v / L * (DOT_R_PT + 1.2) * self.px
                inv = self.ax.transData.inverted()
                (x0, y0), (x1, y1) = inv.transform(p0), inv.transform(q)
                # under the dots
                self.ax.plot([x0, x1], [y0, y1], color=B.CAP, lw=0.6, zorder=2.5, solid_capstyle="butt")
        self.boxes.append(bb)
        return True

    def keep_clear(self, x: float, y: float, r_px: float) -> None:
        px, py = self.ax.transData.transform((x, y))
        self.boxes.append(Bbox.from_extents(px - r_px, py - r_px, px + r_px, py + r_px))


# members_k25_map

MAP_LABELS = [            # (bloc or speaker id, name, arc), the crowded cluster first
    ("Right", "Right", (60, 170)),
    ("965", "Binyamin Netanyahu", (95, 200)),
    ("427", "Avigdor Lieberman", (150, 230)),
    ("30811", "Itamar Ben-Gvir", (255, 300)),
    ("Orthodox", "Orthodox", (190, 260)),
    ("Secular Centre", "Secular Centre", (95, 190)),
    ("23594", "Yair Lapid", (80, 200)),
    ("Left", "Left", (-40, 40)),
    ("Arab-Israeli", "Arab-Israeli", (190, 300)),
    ("23565", "Merav Michaeli", (260, 360)),
    ("30713", "Mansour Abbas", (80, 190)),
]


def prep_members_k25(tables: T.Tables) -> pd.DataFrame:
    """Members sufficient on both scales in the 25th Knesset."""
    m = tables.mk
    k = m[(m["period_type"] == "knesset") & (m["period"].astype(str) == "25")].copy()
    k["speaker_id"] = k["speaker_id"].astype(str)
    k = k[k["sufficient"].fillna(False).astype(bool) & np.isfinite(k["estimate"])]
    x = k[k["dimension"] == "israel_palestine"]
    y = k[k["dimension"] == "lrecon"]
    j = x.merge(y[["speaker_id", "faction_id", "estimate", "n_speeches"]], on=["speaker_id", "faction_id"],
                suffixes=("", "_y"))
    j["both_n"] = j["n_speeches"] + j["n_speeches_y"]
    j = j.sort_values("both_n", ascending=False).drop_duplicates("speaker_id")
    j = j.rename(columns={"estimate": "x", "estimate_y": "y"})
    return j[["speaker_id", "bloc", "x", "y"]].reset_index(drop=True)


def draw_members_k25_map(tables: T.Tables) -> list[Path]:
    df = prep_members_k25(tables)
    n = B.sentences_behind(["israel_palestine", "lrecon"], [(s, "25") for s in df["speaker_id"]])
    title = "In the 25th Knesset, members spread out on the conflict but bunch together on the economy"
    sub = (f"One dot per member of the 2022–26* Knesset, placed by their speeches; a ring marks each bloc's average. "
           f"{B.from_sentences(n).capitalize()}")
    page = B.BlogPage(title, sub, B.sources(B.SRC_CORPUS_SHORT, "Jev model", B.SRC_CHES, partial=True))
    side = 4.85
    left, top = page.W - page.M - side, page.top
    page.set_body(side + 0.56)
    ax = page.axes(left, top, side, side)
    lo, hi = -0.25, 10.25
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.axvline(5, color=B.GRID, lw=0.6, zorder=0)
    ax.axhline(5, color=B.GRID, lw=0.6, zorder=0)
    ax.plot([lo, hi], [lo, lo], color=B.RULE, lw=0.9, zorder=0, solid_capstyle="butt")
    ax.plot([lo, lo], [lo, hi], color=B.RULE, lw=0.9, zorder=0, solid_capstyle="butt")
    ax.set_xticks([0, 5, 10], ["0", "5", "10"], family=B.SANS, fontsize=B.T_TICK, color=B.CAP)
    ax.set_yticks([0, 5, 10], ["0", "5", "10"], family=B.SANS, fontsize=B.T_TICK, color=B.CAP)
    ax.tick_params(pad=4)
    for b in ["Secular Centre", "Arab-Israeli", "Left", "Orthodox", "Right"]:
        g = df[df["bloc"] == b]
        ax.scatter(g["x"], g["y"], s=B.DOT_S, color=B.BLOC[b], alpha=0.85, edgecolor=B.PAPER, linewidths=0.6, zorder=3)
    y = top + side + 0.42
    page.note(left, y, "← more dovish")
    page.note(left + side / 2, y, "The conflict", ha="center")
    page.note(left + side, y, "more hawkish →", ha="right")
    x, up = page.M + 0.02, dict(rotation=90, va="bottom", rotation_mode="anchor")
    page.note(x, top + side, "← more left-wing", ha="left", **up)
    page.note(x, top + side / 2, "The economy", ha="center", **up)
    page.note(x, top, "more right-wing →", ha="right", **up)

    placer = Placer(page, ax, df[["x", "y"]].to_numpy())
    ring_r = 0.5 * np.sqrt(230) / 72 * page.fig.dpi + 2        # px, with halo
    for key, name, arc in MAP_LABELS:
        if key in B.BLOC:
            g = df[df["bloc"] == key]
            cx, cy = float(g["x"].mean()), float(g["y"].mean())
            ax.scatter([cx], [cy], s=230, facecolor="none", edgecolor=B.PAPER, linewidths=4.2, zorder=5)
            ax.scatter([cx], [cy], s=230, facecolor="none", edgecolor=B.BLOC[key], linewidths=2.2, zorder=6)
            placer.keep_clear(cx, cy, ring_r)
            placer.place(cx, cy, name, arc, leader_from=12, fontsize=B.T_LABEL, weight=600, color=B.INK, family=B.SANS)
        else:
            row = df[df["speaker_id"] == key].iloc[0]
            ax.scatter([row["x"]], [row["y"]], s=B.DOT_S, facecolor="none", edgecolor=B.INK, linewidths=0.9, zorder=4)
            placer.place(float(row["x"]), float(row["y"]), name, arc, leader_from=9, fontsize=B.T_LABEL, weight=500,
                         color=B.SEC, family=B.SANS)

    sd_x, sd_y = float(df["x"].std()), float(df["y"].std())
    med = df.groupby("bloc")[["x", "y"]].median()
    in_y = df["y"].quantile([0.1, 0.9]).to_numpy()
    spread = "half" if 1.7 <= sd_x / sd_y <= 2.3 else f"{sd_y / sd_x:.1f} times"
    alt = (f"Scatter of the {len(df)} members of the 2022–26 Knesset with enough speech on both scales. Left to right, "
           f"on the conflict, they run from dovish Arab and Left members near {med.loc['Arab-Israeli', 'x']:.0f}, "
           f"through the Secular Centre around {med.loc['Secular Centre', 'x']:.0f}, to Right and Orthodox members "
           f"near {med.loc['Right', 'x']:.0f}. Up and down, on the economy, they sit much closer together: the middle "
           f"80% of members span {in_y[0]:.0f} to {in_y[1]:.0f}, a spread about {spread} as wide.")
    return page.save("members_k25_map", alt=alt)


# validation_by_scale

VAL_NAMES = {"religious_principles": "Religion in politics", "jewish_settlements": "Settlements",
             "israel_palestine": "The conflict", "democratic_v_jewish_state": "Jewish or\ndemocratic state",
             "lrgen": "Overall left-right", "galtan": "Social values",
             "civlib": "Civil liberties or\nlaw and order", "gender_equality": "Gender equality",
             "lrecon": "The economy", "spendvtax": "Public services or\nlower taxes"}
BENCH_HALF_IN, BENCH_LW = 0.10, 2.6      # bloc-alone benchmark


def prep_validation_rows(tables: T.Tables) -> pd.DataFrame:
    """Scales with a conclusive r, highest first, plus r for the Jewish parties alone."""
    v = tables.validation_summary
    ok = v[(v["tag"] == "primary") & (v["wave"] == "pooled") & (v["party_set"] == "own") & (v["kind"] == "position")
           & (v["status"] == "ok")].copy()
    jewish = T.prep_validation(tables)
    jewish = T.validation_r(jewish[jewish["bloc"] != "Arab-Israeli"]).set_index("dimension")["r"]
    ok["r_jewish"] = ok["dimension"].map(jewish)
    return ok.sort_values("pearson_r", ascending=False).reset_index(drop=True)


def draw_validation_by_scale(tables: T.Tables) -> list[Path]:
    rows = prep_validation_rows(tables)
    title = "The model agrees with the experts on religion and the conflict, less so on the economy"
    sub = (f"Correlation between the model's party placements and the Chapel Hill experts' 2021–22 scores "
           f"({int(rows['n_clusters'].min())} to {int(rows['n_clusters'].max())} parties per scale); lines are 95% "
           f"intervals")
    # this order keeps 'Goldin et al. 2025' on one footer line
    src = B.sources(B.SRC_CORPUS_SHORT, "party placements by the Jev model", B.SRC_CHES)
    key = [("all parties", B.INK, "dot"), ("Jewish parties only", B.INK, "hollow")]
    page = B.BlogPage(title, sub, src, key=key)
    # third key item, the benchmark tick
    kx = page.M + sum(0.17 + page.width_of(lab, B.T_KEY) + 0.28 for lab, *_ in key)
    kc = page.key_y - B.T_KEY * 0.33 / 72
    page.bg.plot([kx + 0.06, kx + 0.06], [kc - BENCH_HALF_IN, kc + BENCH_HALF_IN], color=B.CONTEXT, lw=BENCH_LW,
                 solid_capstyle="butt")
    page.text(kx + 0.17, page.key_y, "a party's bloc alone", size=B.T_KEY)

    rh = 0.40
    left = page.M + 1.55
    width = page.W - page.M - 0.42 - left
    top, height = page.top, rh * len(rows)
    page.set_body(height + 0.52)
    ax = page.axes(left, top, width, height)
    x0, x1 = -0.4 - 0.03, 1.0 + 0.03
    ax.set_xlim(x0, x1)
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.set_yticks([])
    for v in (0, 0.5, 1.0):
        ax.axvline(v, color=B.GRID, lw=0.6, zorder=0)
    ax.set_xticks([0, 0.5, 1.0], ["0", "0.5", "1"], family=B.SANS, fontsize=B.T_TICK, color=B.CAP)
    ax.tick_params(axis="x", pad=5)
    bench = BENCH_HALF_IN / rh                  # in row units
    for i, rw in rows.iterrows():
        ax.plot([rw.r_floor_bloc] * 2, [i - bench, i + bench], color=B.CONTEXT, lw=BENCH_LW, solid_capstyle="butt",
                zorder=1.5)
        ax.plot([rw.pearson_r_bca_lo, rw.pearson_r_bca_hi], [i, i], color=B.INK, lw=1.2, alpha=0.7,
                solid_capstyle="butt", zorder=2)
        ax.scatter([rw.r_jewish], [i], s=80, facecolor=B.PAPER, edgecolor=B.INK, linewidths=1.4, zorder=3)
        ax.scatter([rw.pearson_r], [i], s=B.DOT_S, color=B.INK, linewidths=0, zorder=4)
        yc = top + rh * (i + 0.5)
        page.text(page.M, yc, VAL_NAMES[rw.dimension], size=B.T_LABEL, weight=500, va="center", linespacing=1.0)
        page.text(page.W - page.M, yc, f"{rw.pearson_r:.2f}", size=B.T_VALUE, weight=600, ha="right", va="center")
    for v, word in ((0, "no agreement"), (1.0, "perfect")):
        page.note(left + width * (v - x0) / (x1 - x0), top + height + 0.42, word, ha="center")

    rv = rows.set_index("dimension")
    r, floor = rv["pearson_r"], rv["r_floor_bloc"]
    ident = r[["religious_principles", "jewish_settlements", "israel_palestine", "democratic_v_jewish_state", "lrgen",
               "galtan"]]
    alt = (f"Dot plot of ten scales. The model's party placements correlate with the Chapel Hill experts' at "
           f"{ident.min():.2f} to {ident.max():.2f} on religion, settlements, the conflict, Jewish or democratic "
           f"state, overall left-right and social values, well above what a party's bloc alone gives on religion "
           f"({floor['religious_principles']:.2f}) and settlements ({floor['jewish_settlements']:.2f}). On the economy "
           f"the figure is {r['lrecon']:.2f}, and on public services against taxes {r['spendvtax']:.2f}, "
           f"{'below' if r['spendvtax'] < floor['spendvtax'] else 'against'} the bloc alone "
           f"({floor['spendvtax']:.2f}); the pattern is much the same among Jewish parties alone.")
    return page.save("validation_by_scale", alt=alt)


# polarisation_by_issue

# The economy is dashed to tell it from the courts without colour. Labels use darker shades for contrast.
ISSUES = [("judicial_power", "The courts", B.COURTS, B.LW_STORY, "-", "#8c5a44"),
          ("israel_palestine", "The conflict", B.CONFLICT, B.LW_DATA, "-", B.CONFLICT),
          ("lrecon", "The economy", B.ECONOMY, B.LW_DATA, (0, (2.6, 1.4)), "#4c7359")]
THIN_ABS, THIN_REL = 50, 2 / 3           # hollow dot when few members are placed


def knesset_members(tables: T.Tables, dimensions: Sequence[str]) -> pd.DataFrame:
    """Sufficient member estimates by Knesset, one faction per member."""
    m = tables.mk
    k = m[(m["period_type"] == "knesset") & m["dimension"].isin(list(dimensions))
          & m["sufficient"].fillna(False).astype(bool)]
    k = k[np.isfinite(k["estimate"])].copy()
    k["speaker_id"] = k["speaker_id"].astype(str)
    return k.sort_values("n_speeches", ascending=False).drop_duplicates(["dimension", "period", "speaker_id"])


def prep_polarisation_ends(tables: T.Tables) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Share of members at either end of each scale, by Knesset."""
    k = knesset_members(tables, [d for d, *_ in ISSUES])
    k["end"] = (k["estimate"] <= 2) | (k["estimate"] >= 8)
    g = k.groupby(["dimension", "period"]).agg(share=("end", "mean"), n=("speaker_id", "nunique")).reset_index()
    g["share"] *= 100
    g["thin"] = (g["n"] < THIN_ABS) | (g["n"] < THIN_REL * g.groupby("dimension")["n"].transform("median"))
    return g, k


def draw_polarisation_by_issue(tables: T.Tables) -> list[Path]:
    g, k = prep_polarisation_ends(tables)
    slots = B.KNESSET_SLOTS
    S = {d: g[g["dimension"] == d].set_index("period").reindex(slots) for d, *_ in ISSUES}
    terms = k.loc[k["period"].isin(slots), ["speaker_id", "period"]].itertuples(index=False, name=None)
    n = B.sentences_behind([d for d, *_ in ISSUES], terms)
    title = "On the courts, four in five members now sit at one end or the other; in 1992–96, one in seven"
    sub = (f"Share of members at either end (0–2 or 8–10) of each 0–10 scale, Knesset by Knesset. "
           f"{B.from_sentences(n).capitalize()}")
    src = B.sources(B.SRC_CORPUS_SHORT, "Jev model", f"{B.SRC_CHES}, except the courts scale", partial=True)
    page = B.BlogPage(title, sub, src, key=[("fewer members placed than usual", B.CAP, "hollow")])
    # tight margins give the x labels room
    left = page.M + 0.42
    width = page.W - page.M - 0.64 - left
    top, height = page.top + 0.10, 3.3
    page.set_body(height + 0.10 + 0.34)
    ax = page.axes(left, top, width, height)
    xs = B.knesset_axis(ax)
    ax.set_ylim(0, 100)
    B.hgrid(ax, [0, 25, 50, 75, 100], fmt=lambda v: f"{v:.0f}%", baseline=0)
    ax.tick_params(axis="x", pad=9)             # keeps '0%' clear of '1992–96'
    ax.tick_params(axis="y", pad=6)
    xv = np.array([xs[p] for p in slots])
    for d, name, col, lw, ls, _ in ISSUES:
        s = S[d]
        ax.plot(xv, s["share"], color=col, lw=lw, ls=ls, zorder=3 if d == "judicial_power" else 2,
                solid_capstyle="round", solid_joinstyle="round", dash_capstyle="butt")
        thin = s["thin"].astype(bool).to_numpy()
        ax.scatter(xv[~thin], s["share"][~thin], s=B.DOT_S, color=col, edgecolor=B.PAPER, linewidths=0.6, zorder=4)
        ax.scatter(xv[thin], s["share"][thin], s=B.DOT_S * 0.85, facecolor=B.PAPER, edgecolor=col, linewidths=1.4,
                   zorder=5)
    x_end = left + width * (xs["25"] + 0.5) / len(slots)
    for d, name, _, _, _, tcol in ISSUES:
        v = float(S[d].loc["25", "share"])
        y = top + height * (1 - v / 100)
        page.text(x_end + 0.11, y, f"{v:.0f}%", size=B.T_VALUE, weight=600, colour=tcol, va="center", halo=True)
        dy = {"judicial_power": -0.20, "israel_palestine": 0.20, "lrecon": 0.18}[d]
        page.text(page.W - page.M, y + dy, name, size=B.T_LABEL, weight=600, colour=tcol, ha="right", va="center",
                  halo=True)

    courts, conflict, econ = (S[d]["share"] for d in ("judicial_power", "israel_palestine", "lrecon"))
    alt = (f"Line chart by Knesset, 1992–96 to 2022–26. The share of members at one end or the other of the courts "
           f"scale rose from {courts['13']:.0f}% to {courts['25']:.0f}%. On the conflict it was {conflict['13']:.0f}% "
           f"then and {conflict['25']:.0f}% now, after a peak of {conflict.max():.0f}% in "
           f"{B.KNESSET_YEARS[conflict.idxmax()]} that rests on fewer members than usual; on the economy it stayed "
           f"under {'a quarter' if econ.max() < 25 else 'a third'} throughout.")
    return page.save("polarisation_by_issue", alt=alt)


# voters_and_seats

INES_ITEM = "INES political tendency (5 categories)"


def prep_voters_seats() -> pd.DataFrame:
    """Left and Left-plus-centre shares of Jewish-list seats and of Jewish voters, by election."""
    seats = pd.read_csv(EXTERNAL / "left_seat_share.csv")
    d = pd.to_datetime(seats["election_date"])
    jewish = 120 - seats["arab_list_seats"]
    out = pd.DataFrame({"year": d.dt.year + (d.dt.dayofyear - 1) / 365.25, "election": seats["election_date"].str[:7],
                        "left_seats": seats["share_of_non_arab_seats"].astype(float),
                        "left_centre_seats": 100 * (seats["left_seats"] + seats["secular_centre_seats"]) / jewish})
    v = pd.read_csv(EXTERNAL / "voter_ideology.csv", dtype={"year_or_election": str})
    v = v[(v["source"] == INES_ITEM) & v["population"].str.startswith("Jewish")].drop_duplicates("year_or_election")
    by = {str(r.year_or_election): r for r in v.itertuples()}
    # a survey dated by year stands for that year's election, but 2019 had two
    found = [by.get(e, by.get(e[:4]) if e[:4] != "2019" else None) for e in out["election"]]
    out["left_voters"] = [float(r.left) if r is not None else np.nan for r in found]
    out["left_centre_voters"] = [float(r.left) + float(r.centre) if r is not None else np.nan for r in found]
    return out[(out["year"] >= 1992) & (out["year"] < 2023)].reset_index(drop=True)


def draw_voters_and_seats(tables: T.Tables) -> list[Path]:
    df = prep_voters_seats()
    title = "The Left's seats fell far faster than the share of Jews who call themselves left"
    sub = ("Share of the seats won by lists other than the Arab parties, against the share of Jewish adults calling "
           "themselves left (or centre) in the election study")
    foot = ("Sources: Israel National Election Studies, political tendency item, Jewish respondents (first run online "
            "in 2022); seats: Central Elections Committee via the Israel Democracy Institute")
    page = B.BlogPage(title, sub, foot)
    left = page.M + 0.40
    width = page.W - page.M - 1.62 - left          # room for the end labels
    ph, title_h, gap = 1.75, 0.40, 0.30
    tops = [page.top + title_h, page.top + title_h + ph + gap + title_h]
    page.set_body(title_h + ph + gap + title_h + ph + 0.32)
    x0, x1 = 1987.0, 2023.4                      # room for the start values
    x_in = lambda v: left + width * (v - x0) / (x1 - x0)  # noqa: E731
    panels = [("The Left", "left_seats", "left_voters", B.BLOC["Left"]),
              ("The Left and the centre together", "left_centre_seats", "left_centre_voters", B.INK)]
    for i, ((ptitle, sc, vc, col), ptop) in enumerate(zip(panels, tops)):
        ax = page.axes(left, ptop, width, ph)
        ax.set_xlim(x0, x1)
        ax.set_ylim(0, 60)
        B.hgrid(ax, [0, 20, 40, 60], fmt=lambda v: f"{v:.0f}%", baseline=0)
        ax.tick_params(axis="y", pad=5)
        if i == 1:
            ax.set_xticks([1992, 2006, 2022], ["1992", "2006", "2022"], family=B.SANS, fontsize=B.T_TICK, color=B.CAP)
            ax.tick_params(axis="x", pad=6)
        else:
            ax.set_xticks([])
        page.text(page.M, ptop - 0.19, ptitle, size=B.T_PANEL, weight=600, serif=True)
        ax.plot(df["year"], df[sc], color=col, lw=2.4, zorder=3, solid_joinstyle="round", solid_capstyle="round")
        # seat dots over the wider voter rings, so both show
        ax.scatter(df["year"], df[sc], s=B.DOT_S * 0.75, color=col, edgecolor=B.PAPER, linewidths=0.5, zorder=5)
        # not asked in September 2019, 2020 or 2021, hence the breaks
        vv = df.dropna(subset=[vc])
        ax.plot(df["year"], df[vc], color=B.VOTERS, lw=2.0, ls=(0, (3.2, 1.8)), zorder=2, dash_capstyle="butt")
        ax.scatter(vv["year"], vv[vc], s=B.DOT_S * 1.45, facecolor=B.PAPER, edgecolor=B.VOTERS, linewidths=1.4,
                   zorder=4)
        y_in = lambda v: ptop + ph * (1 - v / 60)  # noqa: E731
        ends = [(f"{df[sc].iloc[-1]:.0f}% of seats", col, y_in(df[sc].iloc[-1])),
                (f"{vv[vc].iloc[-1]:.0f}% of Jewish voters", B.INK, y_in(vv[vc].iloc[-1]))]
        for (s, c, _), y in zip(ends, B.spread([e[2] for e in ends], 0.21, ptop, ptop + ph + 0.05)):
            page.text(x_in(df["year"].iloc[-1]) + 0.12, y, s, size=B.T_LABEL, weight=600, colour=c, va="center")
        s0, v0 = float(df[sc].iloc[0]), float(vv[vc].iloc[0])
        if col == B.INK and abs(round(s0) - round(v0)) <= 1:
            # one label when both ink lines start together
            lo, hi = sorted((round(s0), round(v0)))
            starts = [(f"{lo}–{hi}%" if lo != hi else f"{lo}%", B.INK, y_in((s0 + v0) / 2))]
        else:
            starts = [(f"{s0:.0f}%", col, y_in(s0)), (f"{v0:.0f}%", B.INK, y_in(v0))]
        for (s, c, _), y in zip(starts, B.spread([e[2] for e in starts], 0.19, ptop - 0.05, ptop + ph)):
            page.text(x_in(df["year"].iloc[0]) - 0.12, y, s, size=B.T_VALUE, weight=600, colour=c, ha="right",
                      va="center", halo=True)
        if i == 0:
            zu = df[df["election"] == "2015-03"].iloc[0]
            ax.annotate("Zionist Union\n(Labour with Hatnua)", (zu.year, zu[sc]), xytext=(0, 9),
                        textcoords="offset points", ha="center", va="bottom", fontsize=B.T_NOTE, color=B.SEC,
                        family=B.SERIF, style="italic", linespacing=1.05, path_effects=B.HALO)

    first, last = df.iloc[0], df.iloc[-1]
    alt = (f"Two line charts, 1992 to 2022. Top: the share of Jews calling themselves left fell from "
           f"{first.left_voters:.0f}% to {last.left_voters:.0f}%, while the Left's share of seats won by lists other "
           f"than the Arab parties fell from {first.left_seats:.0f}% to {last.left_seats:.0f}%, breaking away after "
           f"2015. Bottom: the Left and centre together won {last.left_centre_seats:.0f}% of those seats in 2022, "
           f"against {last.left_centre_voters:.0f}% of Jewish voters who call themselves left or centre.")
    return page.save("voters_and_seats", alt=alt)


CHARTS = {
    "members_k25_map": draw_members_k25_map,
    "validation_by_scale": draw_validation_by_scale,
    "polarisation_by_issue": draw_polarisation_by_issue,
    "voters_and_seats": draw_voters_and_seats,
}
