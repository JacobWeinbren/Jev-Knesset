"""Charts on centre parties, Arab parties, Arab and Druze members of Zionist parties, and the courts by observance.

    python -m knesset_ches.blog groups
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from knesset_ches import blog_style as B
from knesset_ches import tables as T
from knesset_ches.blog_style import BlogPage

GREEN, RED, YELLOW = B.BLOC["Arab-Israeli"], B.BLOC["Left"], B.BLOC["Secular Centre"]
MINUS = "−"
END_2026 = 2026 + 209 / 365           # the record ends on 28 July 2026


def _and(items: Sequence[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def _signed(v: float) -> str:
    """One decimal, signed, with a true minus."""
    return f"{v:+.1f}".replace("-", MINUS)


CENTRE_NAMES = {"third_way": "Third Way", "center_party": "Centre Party", "shinui": "Shinui", "kadima": "Kadima",
                "hatnuah": "Hatnuah", "independence": "Independence", "kulanu": "Kulanu", "yesh_atid": "Yesh Atid",
                "blue_white": "Blue and White"}
ELECTIONS = [(1992, 6, 23), (1996, 5, 29), (1999, 5, 17), (2003, 1, 28), (2006, 3, 28), (2009, 2, 10), (2013, 1, 22),
             (2015, 3, 17), (2019, 4, 9), (2019, 9, 17), (2020, 3, 2), (2021, 3, 23), (2022, 11, 1)]


def _yf(y: int, m: int, d: int) -> float:
    """A date as a fraction of a year."""
    return y + (dt.date(y, m, d).timetuple().tm_yday - 1) / 365


def _knesset_spans() -> dict[str, tuple[float, float]]:
    """Each Knesset from its election to the next, plus 21-22-23 merged."""
    e = [_yf(*x) for x in ELECTIONS] + [END_2026]
    spans = {str(13 + i): (e[i], e[i + 1]) for i in range(len(ELECTIONS))}
    spans["21-22-23"] = (spans["21"][0], spans["23"][1])
    return spans


def _knessets_crossed(bar: Sequence[tuple[float, float]], spans: dict[str, tuple[float, float]]) -> int:
    """Knessets a bar spends more than a tenth of a year in, as a reader counts them."""
    return sum(any(min(b, k1) - max(a, k0) > 0.1 for a, b in bar)
               for k, (k0, k1) in spans.items() if "-" not in k)


def _intersect(a: Sequence[tuple[float, float]], b: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    out = sorted((max(a0, b0), min(a1, b1)) for a0, a1 in a for b0, b1 in b if min(a1, b1) - max(a0, b0) > 1e-6)
    merged: list[list[float]] = []
    for lo, hi in out:
        if merged and lo <= merged[-1][1] + 1e-6:
            merged[-1][1] = max(merged[-1][1], hi)
        else:
            merged.append([lo, hi])
    return [(lo, hi) for lo, hi in merged]


def _runs(years: Iterable[int]) -> list[tuple[int, int]]:
    """Runs of years, bridging gaps of a single missing year."""
    runs: list[list[int]] = []
    for y in sorted(years):
        if runs and y - runs[-1][1] <= 2:
            runs[-1][1] = y
        else:
            runs.append([y, y])
    return [(a, b) for a, b in runs]


def _centre_bar(ax: Any, i: int, bar: Sequence[tuple[float, float]], bridged: Sequence[int], cap: float,
                alpha: float) -> None:
    """One party's bar, with a thin line across each bridged year."""
    cuts = [(float(y), float(y + 1)) for y in bridged]
    for a, end in bar:
        pieces, lo = [], a
        for c0, c1 in cuts:
            if lo < c0 < end:
                pieces.append((lo, c0))
                ax.plot([c0, min(c1, end)], [i, i], color=YELLOW, alpha=alpha, lw=1.0, solid_capstyle="butt",
                        zorder=2)
                lo = c1
        pieces.append((lo, end))
        for p0, p1 in pieces:
            if p1 - p0 > 2 * cap:
                ax.plot([p0 + cap, p1 - cap], [i, i], color=YELLOW, alpha=alpha, lw=0.18 * 72, solid_capstyle="round",
                        zorder=3)


def draw_centre_parties(tables: T.Tables) -> list[Path]:
    g = tables.group
    g = g[(g["estimator"] == "hierarchical") & (g["dimension"] == "lrgen") & (g["actor_type"] == "lineage")
          & (g["bloc"] == "Secular Centre") & (g["n_mks"] >= 2)]
    kn, yr = g[g["period_type"] == "knesset"], g[g["period_type"] == "year"]
    mk = tables.mk
    mk = mk[(mk["dimension"] == "lrgen") & (mk["bloc"] == "Secular Centre") & (mk["period_type"] == "knesset")]
    # n_mks counts member-faction rows and would put Yiud at five members, not three
    distinct = mk.groupby(["lineage", "period"])["speaker_id"].nunique().groupby(level=0).max()
    families = sorted(distinct[distinct >= 4].index)
    lasting = [f for f in families if kn.loc[kn["actor_id"] == f, "period"].nunique() > 2]

    spans = _knesset_spans()
    runs, filled, bars = {}, {}, {}
    for f in families:
        years = set(yr.loc[yr["actor_id"] == f, "period"].astype(int))
        runs[f] = _runs(years)
        filled[f] = sorted(y for a, b in runs[f] for y in range(a, b + 1) if y not in years)
        # trimmed to the party's Knessets, or one that left at a spring election runs on to December
        in_knesset = [spans[p] for p in kn.loc[kn["actor_id"] == f, "period"].astype(str)]
        bars[f] = _intersect([(float(a), END_2026 if b >= 2026 else float(b + 1)) for a, b in runs[f]], in_knesset)
    order = sorted(families, key=lambda f: (bars[f][0][0], bars[f][-1][1]))
    crossed = {f: _knessets_crossed(bars[f], spans) for f in families}
    in_2019_22 = [e for e in ELECTIONS if (2019, 1, 1) <= e <= (2022, 12, 31)]

    title = "Nine centre parties in thirty years; only three lasted more than two Knessets"
    subtitle = ("Centre parties that had four or more members in a Knesset, and the years in which two or more of them "
                "spoke")
    page = BlogPage(title, subtitle, sources=B.sources(B.SRC_CORPUS, "Jev model", partial=True))
    pitch, n = 0.36, len(order)
    top = page.top
    height = pitch * n + 0.18
    gap_in = 0.08
    cap_in = 0.09
    count_of = {f: f"{crossed[f]} Knesset" + ("" if crossed[f] == 1 else "s") for f in order}
    name_w = {f: page.width_of(CENTRE_NAMES[f], B.T_LABEL, weight=600 if f in lasting else 400) for f in order}
    count_w = {f: page.width_of(count_of[f], B.T_COUNT) for f in order}
    # widen the x range until every name and count fits beside its bar
    per_in = page.inner / 37.0
    for _ in range(30):
        xmin = min(bars[f][0][0] - (name_w[f] + gap_in) / per_in for f in order) - 0.02 / per_in
        xmax = max(bars[f][-1][1] + (count_w[f] + gap_in) / per_in for f in order) + 0.02 / per_in
        per_in = page.inner / (xmax - xmin)
    ax = page.axes(page.M, top, page.inner, height)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(n - 0.5 + 0.18 / pitch, -0.5)
    cap = cap_in / per_in
    for e in ELECTIONS:
        ax.axvline(_yf(*e), color=B.RULE, lw=0.7, zorder=0)
    for i, f in enumerate(order):
        strong = f in lasting
        _centre_bar(ax, i, bars[f], filled[f], cap, alpha=1.0 if strong else 0.45)
        start, stop = bars[f][0][0] - gap_in / per_in, bars[f][-1][1] + gap_in / per_in
        weight = 600 if strong else 400
        B.ax_label(ax, start, i, CENTRE_NAMES[f], weight=weight, ha="right", va="center", zorder=5)
        B.ax_label(ax, stop, i, count_of[f], size=B.T_COUNT, colour=B.INK if strong else B.CAP, weight=weight,
               ha="left", va="center", zorder=5)
        if f == "blue_white":
            B.ax_label(ax, start, i + 0.52, "as National Unity since 2022", size=B.T_COUNT, colour=B.CAP, ha="right",
                   va="center", zorder=5)
    cluster = [_yf(*e) for e in in_2019_22]
    note = f"lines mark elections; {T.number_word(len(in_2019_22))} in 2019–22"
    note_w = page.width_of(note, B.T_NOTE, serif=True, italic=True)
    note_x = min((cluster[0] + cluster[-1]) / 2, xmax - (note_w / 2 + 0.02) / per_in)
    B.ax_note(ax, note_x, 0, note, ha="center", va="center", zorder=5)
    ax.set_xticks([_yf(1992, 6, 23), _yf(2006, 3, 28), _yf(2022, 11, 1), END_2026])
    ax.set_xticklabels(["1992", "2006", "2022", "2026*"], fontsize=B.T_TICK, color=B.CAP, family=B.SANS)
    ax.set_yticks([])
    page.fit(top + height + 0.30)

    short = [CENTRE_NAMES[f] for f in order if f not in lasting]
    long_ = [f"{CENTRE_NAMES[f]} ({T.number_word(crossed[f])}{' Knessets' if j == 0 else ''}, {runs[f][0][0]}"
             + (" to now)" if runs[f][-1][1] >= 2026 else f"–{str(runs[f][-1][1])[2:]})")
             for j, f in enumerate(f for f in order if f in lasting)]
    alt = (f"Timeline of {T.number_word(len(families))} centre parties, 1992 to 2026. "
           f"{T.number_word(len(short)).capitalize()} ({_and(short)}) sat for one or two Knessets. Only {_and(long_)} "
           f"lasted longer.")
    return page.save("centre_parties", alt=alt)


ARAB_ROWS = [("lrgen", "Overall left-right", "left-wing", "right-wing"),
             ("israel_palestine", "The conflict", "dovish", "hawkish"),
             ("judicial_power", "The courts", "defend courts", "curb courts"),
             ("democratic_v_jewish_state", "Jewish or democratic state", "democratic", "Jewish"),
             ("galtan", "Social values", "liberal", "traditional"),
             ("religious_principles", "Religion in politics", "secular", "religious")]
ARAB_PARTIES = [("Hadash-Ta'al", "Hadash-Ta'al", GREEN, "o"), ("Ra'am", "Ra'am", GREEN, "D"),
                ("Labor", "Labour", RED, "s")]
THIN_MKS = 5
LANE = 0.09                                    # inches between dodged marks


def _lanes(xs: dict[str, float]) -> dict[str, float]:
    """Vertical offsets in inches that keep close marks on a row from hiding each other."""
    order = [p[0] for p in ARAB_PARTIES]
    clusters: list[list[str]] = []
    for k in sorted(xs, key=xs.get):
        if clusters and xs[k] - xs[clusters[-1][-1]] < 0.3:
            clusters[-1].append(k)
        else:
            clusters.append([k])
    out = {}
    for c in clusters:
        for j, k in enumerate(sorted(c, key=order.index)):
            out[k] = (j - (len(c) - 1) / 2) * LANE
    return out


def draw_arab_parties_k25(tables: T.Tables) -> list[Path]:
    g = tables.group
    g = g[(g["estimator"] == "hierarchical") & (g["actor_type"] == "party") & (g["period_type"] == "knesset")]
    dims = [r[0] for r in ARAB_ROWS]
    names = [p[0] for p in ARAB_PARTIES]
    d = g[(g["period"] == "25") & g["actor_name"].isin(names) & g["dimension"].isin(dims)]
    val = {(r.dimension, r.actor_name): r for r in d.itertuples()}
    est = {k: r.estimate for k, r in val.items()}
    hollow = {k: r.n_mks < THIN_MKS for k, r in val.items()}
    lab_members = int(d.loc[d["actor_name"] == "Labor", "n_mks"].max())
    labour_then = (g["period"] == "13") & (g["dimension"] == "lrgen") & (g["actor_name"] == "Labor")
    x_old = g.loc[labour_then, "estimate"].iloc[0]
    x_now = est[("lrgen", "Labor")]
    max_diff = max(abs(est[(dim, "Labor")] - est[(dim, "Hadash-Ta'al")]) for dim in dims)
    raam_social = val[("galtan", "Ra'am")]

    mk = tables.mk
    mk = mk[mk["period_type"] == "knesset"]
    members = mk[(mk["period"] == "25") & mk["dimension"].isin(dims) & mk["faction_name"].isin(names)]
    lab13 = mk[(mk["period"] == "13") & (mk["dimension"] == "lrgen") & (mk["faction_name"] == "Labor")]
    n = (B.sentences_behind(dims, [(s, "25") for s in members["speaker_id"].unique()])
         + B.sentences_behind(["lrgen"], [(s, "13") for s in lab13["speaker_id"].unique()]))

    title = "Labour now sits close to the Arab parties; Ra'am parts from both on religion and social values"
    subtitle = (f"Party averages in 2022–26*; hollow marks rest on fewer than {T.number_word(THIN_MKS)} members, and "
                f"Labour has {T.number_word(lab_members)}. {B.from_sentences(n).capitalize()}")
    page = BlogPage(title, subtitle,
                    sources=B.sources("Knesset Corpus (Goldin et al. 2025) to July 2026", "Jev model", B.SRC_CHES,
                                      "courts scale: the project's own", partial=True))
    lab_w, stair_h, pitch = 1.42, 0.62, 0.56
    left = page.M + lab_w
    width = page.W - page.M - left
    top = page.top
    first = stair_h + 0.14
    height = first + pitch * (len(ARAB_ROWS) - 1) + 0.56
    ax = page.axes(left, top, width, height)
    ax.set_xlim(-0.12, 10.12)
    ax.set_ylim(height, 0)                             # y in inches from the top of the plot
    ax.set_xticks([])
    ax.set_yticks([])
    per_in = width / 10.24
    ys = [first + i * pitch for i in range(len(ARAB_ROWS))]
    for v in (0, 5, 10):
        ax.plot([v, v], [first - 0.14, ys[-1] + 0.32], color=B.GRID, lw=0.6, zorder=0)
    mark_y: dict[tuple[str, str], float] = {}
    for (dim, name, lo, hi), y in zip(ARAB_ROWS, ys):
        xs = {p[0]: est[(dim, p[0])] for p in ARAB_PARTIES}
        lanes = _lanes(xs)
        ax.plot([min(xs.values()), max(xs.values())], [y, y], color=B.CONTEXT, lw=1.1, zorder=1,
                solid_capstyle="round")
        for key, label, colour, marker in ARAB_PARTIES:
            yy = y + lanes[key]
            mark_y[(dim, key)] = yy
            thin = hollow[(dim, key)]
            ax.scatter([xs[key]], [yy], s=B.DOT_S * (0.95 if marker == "s" else 1.0), marker=marker,
                       facecolor=B.PAPER if thin else colour, edgecolor=colour if thin else B.PAPER,
                       linewidths=1.5 if thin else 0.8, zorder=4)
        B.ax_note(ax, 0, y + 0.28, lo, ha="left", zorder=5)
        B.ax_note(ax, 10, y + 0.28, hi, ha="right", zorder=5)
        wrapped = page.wrap(name, B.T_LABEL, weight=500, width=lab_w - 0.12)
        lh = B.T_LABEL * 1.2 / 72
        for j, s in enumerate(wrapped):
            page.text(page.M, top + y + (j - (len(wrapped) - 1) / 2) * lh, s, size=B.T_LABEL, weight=500, va="center")
        if dim in ("galtan", "religious_principles"):
            x = est[(dim, "Ra'am")]
            B.ax_label(ax, x + 0.13 / per_in, mark_y[(dim, "Ra'am")], f"{x:.1f}", size=B.T_VALUE, weight=600,
                   va="center", zorder=6)
    y_lab = mark_y[("lrgen", "Labor")]
    x_stop = max(est[("lrgen", p)] for p in names) + 0.12 / per_in
    ax.annotate("", xy=(x_stop, y_lab), xytext=(x_old - 0.09 / per_in, y_lab),
                arrowprops=dict(arrowstyle="-|>", color=B.OTHER, lw=1.0, mutation_scale=9, shrinkA=0, shrinkB=0),
                zorder=2)
    ax.scatter([x_old], [y_lab], s=B.DOT_S * 0.95, marker="s", facecolor=B.OTHER, edgecolor=B.PAPER, linewidths=0.8,
               zorder=4)
    B.ax_label(ax, x_old + 0.10 / per_in, y_lab, "Labour in 1992–96", size=B.T_COUNT, colour=B.CAP, va="center",
           zorder=6)
    stair = sorted(ARAB_PARTIES, key=lambda p: est[("lrgen", p[0])])
    for k, (key, label, colour, marker) in enumerate(stair):
        x = est[("lrgen", key)]
        base = 0.16 + k * 0.2
        ax.plot([x, x], [base + 0.04, mark_y[("lrgen", key)] - 0.075], color=colour, lw=0.7, zorder=3)
        B.ax_label(ax, x + 0.03 / per_in, base, label, colour=colour, weight=500, zorder=6)
    for v in (0, 5, 10):
        ax.text(v, ys[-1] + 0.52, str(v), ha="center", va="baseline", fontsize=B.T_TICK, color=B.CAP, family=B.SANS)
    page.fit(top + height + 0.04)

    ra, ht = "Ra'am", "Hadash-Ta'al"
    alt = (f"Dot plot of three parties on six scales in the 2022–26 Knesset. Labour, at {x_old:.1f} on left-right in "
           f"1992–96 and {x_now:.1f} now, sits within about {'one and a half' if max_diff < 1.65 else 'two'} points "
           f"of Hadash-Ta'al on every scale. Ra'am sits with both on the conflict and the courts but reaches "
           f"{est[('religious_principles', ra)]:.1f} on religion and {est[('galtan', ra)]:.1f} on social values (the "
           f"latter from only {T.number_word(raam_social.n_mks)} members), against "
           f"{min(est[('galtan', 'Labor')], est[('galtan', ht)]):.1f} to "
           f"{max(est[('religious_principles', 'Labor')], est[('religious_principles', ht)]):.1f} for the other two.")
    return page.save("arab_parties_k25", alt=alt)


DRUZE_ROWS = [("democratic_v_jewish_state", "Jewish or democratic state"), ("israel_palestine", "The conflict"),
              ("civlib", "Civil liberties or law and order"), ("lrgen", "Overall left-right")]
DRUZE, ARAB = B.SLATE, B.CLAY


def _copartisan_gaps(tables: T.Tables) -> pd.DataFrame:
    """Each Druze or Arab member's gap from the Jewish members of their party in that Knesset."""
    mk = tables.mk
    k = mk[(mk["period_type"] == "knesset") & mk["sufficient"].astype(bool) & (mk["bloc"] != "Arab-Israeli")].copy()
    k["speaker_id"] = k["speaker_id"].astype(str)
    k = k.sort_values("n_speeches", ascending=False).drop_duplicates(["speaker_id", "period", "dimension"])
    demo = pd.read_csv(tables.path / "members_demographics.csv", dtype={"speaker_id": str})
    k = k.merge(demo[["speaker_id", "group"]], on="speaker_id", how="left")
    jew = k[k["group"] == "Jewish"].groupby(["lineage", "period", "dimension"])["estimate"].agg(["mean", "count"])
    jew = jew[jew["count"] >= 2].reset_index()
    x = k[k["group"].isin(["Druze", "Arab"])].merge(jew, on=["lineage", "period", "dimension"])
    x["gap"] = x["estimate"] - x["mean"]
    return x


def draw_arab_druze_members(tables: T.Tables) -> list[Path]:
    x = _copartisan_gaps(tables)
    dims = [r[0] for r in DRUZE_ROWS]
    stats: dict[tuple[str, str], dict] = {}
    for dim in dims:
        for grp in ("Druze", "Arab"):
            gap = x.loc[(x["dimension"] == dim) & (x["group"] == grp), "gap"]
            mean, se = float(gap.mean()), float(gap.std(ddof=1) / np.sqrt(len(gap)))
            stats[(dim, grp)] = dict(mean=mean, lo=mean - 1.96 * se, hi=mean + 1.96 * se, n=len(gap),
                                     below=int((gap < 0).sum()))
    first = x[x["dimension"] == dims[0]]
    counts = {}
    for grp in ("Druze", "Arab"):
        y = first[first["group"] == grp]
        parties = set(y["lineage"])
        where = "all Labour or Meretz" if parties <= {"labor", "meretz"} else f"from {len(parties)} parties"
        counts[grp] = f"{grp}: {y['speaker_id'].nunique()} members, {where}"
    terms = x.loc[x["dimension"].isin(dims), ["speaker_id", "period"]].drop_duplicates()
    n = B.sentences_behind(dims, terms.itertuples(index=False, name=None))

    title = "Druze and Arab members of Zionist parties part from their colleagues on the Jewish state"
    subtitle = (f"Average gap in points from the Jewish members of the same party in the same Knesset; members of Arab "
                f"parties left out. {B.from_sentences(n).capitalize()}")
    # not "Arab members of Labour and Meretz", which reads as if Arabs sit only there
    key = [("Druze members", DRUZE, "square"), ("Arab members of Zionist parties", ARAB, "dot")]
    page = BlogPage(title, subtitle, key=key,
                    sources=B.sources(B.SRC_CORPUS, "Jev model", f"scales from the {B.SRC_CHES}"))
    top = page.top
    count_w = max(page.width_of(c, B.T_COUNT) for c in counts.values())
    xmin = -3.35
    xmax = 1.2
    for _ in range(20):
        per_in = page.inner / (xmax - xmin)
        xmax = max(1.2, (count_w + 0.10) / per_in)
    per_in = page.inner / (xmax - xmin)
    note_h, pitch = 0.30, 0.70
    height = note_h + pitch * len(DRUZE_ROWS) + 0.02
    ax = page.axes(page.M, top, page.inner, height)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(height, 0)
    label_band = []
    for i, (dim, name) in enumerate(DRUZE_ROWS):
        y0 = note_h + i * pitch
        w = page.width_of(name, B.T_LABEL, weight=500)
        label_band.append((xmin + (w + 0.06) / per_in, y0 + 0.2 - 0.15, y0 + 0.2 + 0.05))
        B.ax_label(ax, xmin, y0 + 0.2, name, weight=500, zorder=5)
    for v in (-3, -2, -1, 1):
        cuts = [(a, b) for x_end, a, b in label_band if v < x_end]
        if v > xmax - (count_w + 0.04) / per_in:
            cuts.append((note_h + 0.36 - 0.10, note_h + 0.55 + 0.10))
        lo = note_h
        for a, b in sorted(cuts):
            ax.plot([v, v], [lo, a], color=B.GRID, lw=0.6, zorder=0)
            lo = b
        ax.plot([v, v], [lo, height], color=B.GRID, lw=0.6, zorder=0)
    ax.plot([0, 0], [note_h - 0.08, height], color=B.INK, lw=0.9, zorder=1)
    B.ax_note(ax, 0, note_h - 0.14, "same as their Jewish colleagues", ha="center")
    for i, (dim, name) in enumerate(DRUZE_ROWS):
        y0 = note_h + i * pitch
        for grp, colour, marker, dy in (("Druze", DRUZE, "s", 0.36), ("Arab", ARAB, "o", 0.55)):
            s = stats[(dim, grp)]
            ax.plot([s["lo"], s["hi"]], [y0 + dy, y0 + dy], color=colour, lw=1.2, alpha=0.7, zorder=2,
                    solid_capstyle="butt")
            ax.scatter([s["mean"]], [y0 + dy], s=B.DOT_S, marker=marker, color=colour, zorder=3, linewidths=0)
            B.ax_label(ax, s["lo"] - 0.07 / per_in, y0 + dy, _signed(s["mean"]), size=B.T_VALUE, weight=600, ha="right",
                   va="center", zorder=4)
            if i == 0:
                B.ax_label(ax, xmax, y0 + dy, counts[grp], size=B.T_COUNT, colour=colour, weight=500, ha="right",
                       va="center", zorder=4)
    ax.set_xticks([-3, -2, -1, 0, 1])
    ax.set_xticklabels([f"{MINUS}3", f"{MINUS}2", f"{MINUS}1", "0", "+1"], fontsize=B.T_TICK, color=B.CAP,
                       family=B.SANS)
    ax.tick_params(axis="x", pad=4)
    ax.set_yticks([])
    page.note(page.M, top + height + 0.52, "← more democratic, dovish, liberal or left-wing")
    page.fit(top + height + 0.58)

    dz, ar = stats[(dims[0], "Druze")], stats[(dims[0], "Arab")]
    lr_d, lr_a = stats[("lrgen", "Druze")]["mean"], stats[("lrgen", "Arab")]["mean"]
    alt = (f"Dot chart of how Druze and Arab members of Zionist parties differ from Jewish members of the same party. "
           f"On Jewish or democratic state, Druze members sit {abs(dz['mean']):.1f} points further towards democratic "
           f"({dz['below']} of {dz['n']} cases) and Arab members of Zionist parties (all Labour or Meretz) "
           f"{abs(ar['mean']):.1f} points ({ar['below']} of {ar['n']}). On overall left-right the gaps are smaller, "
           f"{abs(lr_d):.1f} and {abs(lr_a):.1f}.")
    return page.save("arab_druze_members", alt=alt)


ERAS = [("1992–2009", ["13", "14", "15", "16", "17"]), ("2009–19", ["18", "19", "20"]),
        ("2019–26*", ["21-22-23", "24", "25"])]
RELIG = [("Charedi", B.HAREDI, "D"), ("Religious", B.RELIGIOUS, "s"), ("Secular", B.SECULAR, "o")]


def _share_missing(share: float) -> str:
    if 0.17 <= share < 0.23:
        return "a fifth"
    if share < 0.26:
        return "about a quarter"
    return "over a quarter" if share < 0.31 else T.share_words(share)


def draw_courts_by_religiosity(tables: T.Tables) -> list[Path]:
    mk = tables.mk
    k = mk[(mk["period_type"] == "knesset") & (mk["dimension"] == "judicial_power") & mk["sufficient"].astype(bool)
           & mk["bloc"].isin(["Right", "Orthodox"])].copy()
    k["speaker_id"] = k["speaker_id"].astype(str)
    k = k.sort_values("n_speeches", ascending=False).drop_duplicates(["speaker_id", "period"])
    demo = pd.read_csv(tables.path / "members_demographics.csv", dtype={"speaker_id": str})
    k = k.merge(demo[["speaker_id", "religiosity"]], on="speaker_id", how="left")
    k["religiosity"] = k["religiosity"].replace({"Haredi": "Charedi"})      # the post's spelling
    k["era"] = k["period"].map({p: i for i, (_, ps) in enumerate(ERAS) for p in ps})
    k = k.dropna(subset=["era"])
    missing = k[k["era"] == 0].groupby("speaker_id")["religiosity"].first().isna().mean()
    k = k.dropna(subset=["religiosity"])
    # average each member within an era, then bootstrap over members
    mem = k.groupby(["era", "religiosity", "speaker_id"])["estimate"].mean().reset_index()
    rng = np.random.default_rng(20260923)
    res: dict[tuple[str, int], dict[str, float]] = {}
    for (e, r), t in mem.groupby(["era", "religiosity"]):
        v = t["estimate"].to_numpy()
        boots = rng.choice(v, size=(2000, len(v)), replace=True).mean(axis=1)
        res[(r, int(e))] = dict(mean=float(v.mean()), lo=float(np.percentile(boots, 2.5)),
                                hi=float(np.percentile(boots, 97.5)), n=len(v))
    terms = k[["speaker_id", "period"]].drop_duplicates()
    n = B.sentences_behind(["judicial_power"], terms.itertuples(index=False, name=None))

    title = "Charedi members wanted the courts curbed first; the religious right has since gone further"
    subtitle = (f"Right and Orthodox members by recorded religious observance; {_share_missing(missing)} before 2009 "
                f"have none recorded. {B.from_sentences(n).capitalize()}")
    page = BlogPage(title, subtitle,
                    sources=B.sources(B.SRC_CORPUS, "Jev model on the project's own courts scale", partial=True))
    top = page.top
    page.text(page.M, top + 0.02, "Power of the courts (0–10 scale, cropped at 4)", size=B.T_PANEL, weight=600,
              serif=True, va="top")
    ptop = top + 0.42
    width = page.inner - 1.25
    height = 3.0
    ax = page.axes(page.M, ptop, width, height)
    ax.set_xlim(-0.3, 2.15)
    ax.set_ylim(4, 10)
    B.hgrid(ax, [4, 6, 8, 10])
    pole_x = 0.2                                        # clear of the 1992-2009 whiskers
    B.ax_note(ax, pole_x, 10, "↑ curb the courts", va="center")
    B.ax_note(ax, pole_x, 4 + 0.07 * 6.0 / height, "↓ defend the courts", va="bottom", zorder=5)
    dodge = {"Charedi": -0.07, "Religious": 0.0, "Secular": 0.07}
    for r, colour, marker in RELIG:
        xs = [e + dodge[r] for e in range(3)]
        for e, xv in enumerate(xs):
            ax.plot([xv, xv], [res[(r, e)]["lo"], res[(r, e)]["hi"]], color=colour, lw=1.2, alpha=0.7, zorder=2)
        ax.plot(xs, [res[(r, e)]["mean"] for e in range(3)], color=colour, lw=2.4, zorder=3, marker=marker,
                markersize=6.5, markerfacecolor=colour, markeredgecolor=B.PAPER, markeredgewidth=0.6)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels([e[0] for e in ERAS], fontsize=B.T_TICK, color=B.CAP, family=B.SANS)
    per_unit = height / 6.0
    ys = [ptop + (10 - res[(r, 2)]["mean"]) * per_unit for r, _, _ in RELIG]
    spread = B.spread([-v for v in ys], 0.42, -(ptop + height), -ptop)
    xl = page.M + width + 0.08
    for (r, colour, _), yv in zip(RELIG, spread):
        page.text(xl, -yv - 0.03, f"{r} {res[(r, 2)]['mean']:.1f}", size=B.T_LABEL, weight=600, colour=colour,
                  va="baseline", halo=True)
        page.text(xl, -yv + 0.16, f"{res[(r, 2)]['n']} members", size=B.T_COUNT, colour=B.CAP, va="baseline",
                  halo=True)
    page.fit(ptop + height + 0.32)

    mean = {key: v["mean"] for key, v in res.items()}
    alt = (f"Slope chart of Right and Orthodox members on the power of the courts in three eras. Charedi members "
           f"averaged {mean[('Charedi', 0)]:.1f} in 1992–2009 and {mean[('Charedi', 2)]:.1f} in 2019–26. Religious "
           f"members rose from {mean[('Religious', 0)]:.1f} to {mean[('Religious', 2)]:.1f}, passing them, and secular "
           f"members rose from {mean[('Secular', 0)]:.1f} to {mean[('Secular', 2)]:.1f}.")
    return page.save("courts_by_religiosity", alt=alt)


CHARTS = {
    "centre_parties": draw_centre_parties,
    "arab_parties_k25": draw_arab_parties_k25,
    "arab_druze_members": draw_arab_druze_members,
    "courts_by_religiosity": draw_courts_by_religiosity,
}
