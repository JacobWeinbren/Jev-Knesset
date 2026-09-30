"""Charts on women in the plenum, heckles at women and men, a card of sexist heckles, and Netanyahu by party.

    python -m knesset_ches.blog women
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from knesset_ches import blog_style as B
from knesset_ches import tables as T
from knesset_ches.blog_style import BlogPage

NAMES = {"Blue & White": "Blue and White", "State Camp": "National Unity"}     # as named across the post


def _inches(v: float, lim: tuple[float, float], start: float, length: float) -> float:
    """Page inches of v on an axis. Page y runs down, so give y limits top first."""
    return start + (v - lim[0]) / (lim[1] - lim[0]) * length


def _over(page: BlogPage, ax: Any, x: float, y: float, s: str, **kw: Any):
    """A label at page inches, drawn in `ax` to sit above its lines."""
    return B.ax_label(ax, x, y, s, transform=page.bg.transData, zorder=8, clip_on=False, **kw)


def _plenary_sentences() -> int:
    """The sentences behind women_share.csv."""
    u = pd.read_parquet(T.UNITS, columns=["is_chairman", "n_sentences"])
    return int(u.loc[~u["is_chairman"].fillna(False).astype(bool), "n_sentences"].sum())


def draw_women_on_the_floor(tables: T.Tables) -> list[Path]:
    d = pd.read_csv(tables.path / "women_share.csv", dtype={"knesset": str}).set_index("knesset")
    ks = B.KNESSET_SLOTS
    sp = d.loc[ks, "share_of_speakers"].to_numpy(float)
    se = d.loc[ks, "share_of_sentences"].to_numpy(float)

    title = "Women are nearly a quarter of the members who speak, and now say a third of what is said"
    sub = (f"Women's share of the members who spoke in plenary debates, and of the sentences spoken, Knesset by "
           f"Knesset. {B.from_sentences(_plenary_sentences()).capitalize()}")
    page = BlogPage(title, sub, B.sources(B.SRC_CORPUS, "members' gender from the corpus's member records",
                                          partial=True))
    left, width, height = page.M + 0.44, page.inner - 0.44, 3.0
    top = page.top + 0.08
    ax = page.axes(left, top, width, height)
    xs = B.knesset_axis(ax)
    ax.set_ylim(0.0, 0.40)
    B.hgrid(ax, [0.0, 0.1, 0.2, 0.3, 0.4], fmt=T.pct, baseline=0.0)
    ax.tick_params(axis="x", pad=6)
    ax.tick_params(axis="y", pad=6)
    x = np.array([xs[k] for k in ks])
    ax.plot(x, sp, color=B.WOMEN, lw=B.LW_DATA, ls=(0, (3.2, 2.0)), zorder=3)
    ax.scatter(x, sp, s=B.DOT_S, marker="o", facecolor=B.PAPER, edgecolor=B.WOMEN, linewidths=1.4, zorder=4)
    ax.plot(x, se, color=B.WOMEN, lw=2.4, zorder=5, solid_capstyle="round")
    ax.scatter(x, se, s=B.DOT_S, marker="o", facecolor=B.WOMEN, edgecolor=B.PAPER, linewidths=0.8, zorder=6)

    def glyph(xr: float, y: float, dashed: bool) -> None:
        cy = y - B.T_LABEL * 0.34 / 72
        ax.add_line(Line2D([xr - 0.30, xr - 0.04], [cy, cy], color=B.WOMEN, lw=B.LW_DATA, transform=page.bg.transData,
                           clip_on=False, ls=(0, (2.4, 1.5)) if dashed else "-", solid_capstyle="round", zorder=8))
        ax.scatter([xr - 0.17], [cy], s=B.DOT_S * 0.8, marker="o", facecolor=B.PAPER if dashed else B.WOMEN,
                   transform=page.bg.transData, edgecolor=B.WOMEN, linewidths=1.4, zorder=9, clip_on=False)

    right = page.W - page.M
    y1 = _inches(se[-1], (0.40, 0.0), top, height) - 0.20
    t1 = _over(page, ax, right, y1, f"{T.share_words(se[-1])} of what was said, {T.pct(se[-1])}", weight=500,
               ha="right")
    glyph(right - page.width_of(t1), y1, dashed=False)
    y2 = _inches(0.125, (0.40, 0.0), top, height)
    t2 = _over(page, ax, right, y2, "nearly a quarter of the", weight=500, ha="right")
    _over(page, ax, right, y2 + B.T_LABEL * 1.25 / 72, f"members who spoke, {T.pct(sp[-1])}", weight=500, ha="right")
    glyph(right - page.width_of(t2), y2, dashed=True)
    page.fit(top + height + 0.36)

    alt = (f"Line chart by Knesset, 1992–96 to 2022–26. Women rose from {T.pct(sp[0])} to {T.pct(sp[-1])} of the "
           f"members who spoke in plenary debates ({T.pct(sp[ks.index('24')])} in 2021–22). Their share of the "
           f"sentences spoken rose from {T.pct(se[0])} to {T.pct(se[-1])}.")
    return page.save("women_on_the_floor", alt=alt)


HECKLE_ROWS = [("heckle_hostile", "Hostile"), ("heckle_personal", "A personal attack"),
               ("heckle_silencing", "Telling the speaker to stop")]
BROAD = ["explicit", "insult", "trope", "family"]       # hand labels that count as about looks, age or sex


def _confirmed_draws(row: pd.Series, rng: np.random.Generator, n: int = 400_000) -> np.ndarray:
    """Draws of the confirmed rate, the flagged rate times the share of flags that held up."""
    ns, nf = float(row["n_scored"]), float(row["n_flagged"])
    k = round(float(row["held_up_broad"]) * nf)
    f = float(row["flagged"]) * rng.beta(nf + 0.5, ns - nf + 0.5, n) / (nf / ns)
    h = rng.beta(k + 0.5, nf - k + 0.5, n)
    return f * h


def _in_words(rate: float) -> str:
    inv = 1 / rate
    step = 500 if inv >= 1000 else 100
    return f"1 in {int(round(inv / step) * step):,}"


def _shares_panel(page: BlogPage, h: pd.DataFrame, left: float, width: float, top: float) -> float:
    """Paired dots for heckles at women and at men. Returns the panel's bottom."""
    rowh, dodge = 0.55, 0.10
    page.text(page.M, top + 0.02, "Share of heckles", size=B.T_PANEL, weight=600, serif=True, va="top")
    top += 0.42
    xlim = (0.0, 0.20)
    ax = page.axes(left, top, width, rowh * len(HECKLE_ROWS))
    ax.set_xlim(*xlim)
    ax.set_ylim(len(HECKLE_ROWS) - 0.5, -0.5)
    ax.set_yticks([])
    for v in (0.0, 0.05, 0.10, 0.15, 0.20):
        ax.axvline(v, color=B.GRID, lw=0.6, zorder=0)
    ax.set_xticks([0.0, 0.05, 0.10, 0.15, 0.20])
    ax.set_xticklabels(["0%", "5%", "10%", "15%", "20%"], fontsize=B.T_TICK, color=B.CAP, family=B.SANS)
    ax.tick_params(axis="x", pad=5)
    off = dodge / rowh
    for i, (c, lab) in enumerate(HECKLE_ROWS):
        yc = top + (i + 0.5) * rowh
        page.text(page.M, yc + B.T_LABEL * 0.34 / 72, lab, size=B.T_LABEL, weight=500)
        w, m = float(h.loc["Women", c]), float(h.loc["Men", c])
        ax.plot([m, w], [i + off, i - off], color=B.RULE, lw=1.0, zorder=1)
        ax.scatter([w], [i - off], s=B.DOT_S, marker="o", facecolor=B.WOMEN, edgecolor=B.WOMEN, linewidths=1.4,
                   zorder=4)
        ax.scatter([m], [i + off], s=B.DOT_S, marker="o", facecolor=B.PAPER, edgecolor=B.MEN, linewidths=1.4, zorder=5)
        if round(w * 100) == round(m * 100):
            _over(page, ax, _inches(max(w, m), xlim, left, width) + 0.12, yc, f"{w * 100:.0f}% for both",
                  size=B.T_VALUE, weight=600, va="center")
        else:
            _over(page, ax, _inches(w, xlim, left, width) + 0.12, yc - dodge, f"women {T.pct(w)}", size=B.T_VALUE,
                  weight=600, va="center")
            _over(page, ax, _inches(m, xlim, left, width) + 0.12, yc + dodge, f"men {T.pct(m)}", size=B.T_VALUE,
                  weight=600, va="center")
        if i == 0:
            xl = _inches(min(w, m), xlim, left, width) - 0.10
            _over(page, ax, xl, yc - dodge, "women", size=B.T_COUNT, colour=B.WOMEN, ha="right", va="center")
            _over(page, ax, xl, yc + dodge, "men", size=B.T_COUNT, colour=B.INK, ha="right", va="center")
    return top + rowh * len(HECKLE_ROWS)


def _looks_panel(page: BlogPage, rates: list[tuple], left: float, width: float, y: float) -> float:
    """Bars for heckles about looks, age or sex. Returns the panel's bottom."""
    page.text(page.M, y, "About looks, age or sex, per 10,000 heckles", size=B.T_PANEL, weight=600, serif=True,
              va="top")
    y += 0.42
    barh_row = 0.42
    xlim = (0.0, 30.0)
    ax = page.axes(left, y, width, barh_row * 2)
    ax.set_xlim(*xlim)
    ax.set_ylim(1.5, -0.5)
    ax.set_yticks([])
    for v in (0, 10, 20, 30):
        ax.axvline(v, color=B.GRID, lw=0.6, zorder=0)
    ax.set_xticks([0, 10, 20, 30])
    ax.set_xticklabels(["0", "10", "20", "30"], fontsize=B.T_TICK, color=B.CAP, family=B.SANS)
    ax.axvline(0, color=B.INK, lw=0.9, zorder=4)
    ax.tick_params(axis="x", pad=5)
    for i, (who, rate, ci) in enumerate(rates):
        if who == "At men":                                         # men hollow, as their dots are
            ax.barh(i, rate * 1e4, height=0.5, facecolor=B.PAPER, edgecolor=B.MEN, linewidth=1.4, zorder=2)
        else:
            ax.barh(i, rate * 1e4, height=0.5, color=B.WOMEN, zorder=2)
        ax.plot([ci[0] * 1e4, ci[1] * 1e4], [i, i], color=B.INK, lw=1.2, alpha=0.7, zorder=3, solid_capstyle="butt")
        for e in ci:
            ax.plot([e * 1e4] * 2, [i - 0.12, i + 0.12], color=B.INK, lw=1.2, alpha=0.7, zorder=3)
        yc = y + (i + 0.5) * barh_row
        page.text(page.M, yc + B.T_LABEL * 0.34 / 72, who, size=B.T_LABEL, weight=500)
        _over(page, ax, _inches(ci[1] * 1e4, xlim, left, width) + 0.10, yc, _in_words(rate), size=B.T_VALUE,
              weight=600, va="center")
        if i == 0:
            _over(page, ax, _inches((ci[0] + ci[1]) / 2 * 1e4, xlim, left, width), yc - 0.5 * 0.5 * barh_row - 0.05,
                  "95% range", size=B.T_COUNT, colour=B.CAP, ha="center")
    return y + barh_row * 2


def draw_heckling_women_and_men(tables: T.Tables) -> list[Path]:
    h = pd.read_csv(tables.path / "heckles.csv")
    h = h[h["facet"] == "standardised"].set_index("target_gender")
    g = pd.read_csv(tables.path / "heckles_gendered.csv").set_index("target_gender")
    hc = pd.read_csv(T.HANDCHECK)
    rw, rm = float(g.loc["Women", "confirmed_broad"]), float(g.loc["Men", "confirmed_broad"])
    rng = np.random.default_rng(20260923)
    ci_w = np.percentile(_confirmed_draws(g.loc["Women"], rng), [2.5, 97.5])
    ci_m = np.percentile(_confirmed_draws(g.loc["Men"], rng), [2.5, 97.5])
    n_read = int(hc["target_gender"].isin(["Women", "Men"]).sum())
    held = hc[hc["hand_label"].isin(BROAD)]
    n_held = {t: int((held["target_gender"] == t).sum()) for t in ("Women", "Men")}

    title = "A heckle at a woman is four times as likely as one at a man to be about her looks or sex"
    # heckles at men were sampled, so the counts are not totals
    sub = (f"Heckles by members in plenary speeches, compared Knesset by Knesset, 1992–2026*. "
           f"From all {T.round_count(h.loc['Women', 'n_scored'])} at women and a sample of "
           f"{T.round_count(h.loc['Men', 'n_scored'])} at men")
    # our own questions, so no study is credited
    foot = B.sources(B.SRC_CORPUS, "Jev model, the project's own questions, flags read by hand", partial=True)
    page = BlogPage(title, sub, foot)
    name_w = max(page.width_of(lab, B.T_LABEL, weight=500) for _, lab in HECKLE_ROWS) + 0.22
    left = page.M + name_w
    width = page.inner - name_w - 1.05
    y = _shares_panel(page, h, left, width, page.top)
    y = _looks_panel(page, [("At women", rw, ci_w), ("At men", rm, ci_m)], left, width, y + 0.30 + 0.28)
    y += 0.36 + 0.14
    note = (f"Of {T.round_count(g['n_scored'].sum())} heckles, {n_read} were flagged and read by hand; "
            f"{n_held['Women']} at women and {n_held['Men']} at men held up")
    page.note(page.M, y, note)
    page.fit(y + 0.06)

    alt = (f"Chart of heckles in the plenum, 1992 to 2026. A heckle at a woman is about {rw / rm:.0f} times as likely "
           f"as one at a man to be about her looks, age or sex: about {rw * 1e4:.0f} in every 10,000 heckles at women "
           f"against {rm * 1e4:.0f} at men. All {n_read} flagged cases were read by hand, and {n_held['Women']} at "
           f"women and {n_held['Men']} at men held up. The top panel shows the shares of hostile heckles, personal "
           f"attacks and calls to stop for each.")
    return page.save("heckling_women_and_men", alt=alt)


# (handcheck row, our translation), from explicit or insult rows naming no person, title or office
HECKLES = [
    (108, "You look younger."),          # not row 70, which could read as a jab at favouritism
    (46, "… whoever teaches a woman Torah teaches her folly … you're a classic example."),
    (42, "But put your glasses on, they suit you."),
    (24, "Never mind, never mind. Men find it hard to concentrate."),
    (75, "Cluck, cluck, cluck, hen. Rightly called a hen, clucking all day."),
    (184, "Be pretty and shut up."),
]
ONE = {"Women": "a woman", "Men": "a man"}


def draw_sexist_heckles_card(tables: T.Tables) -> list[Path]:
    hc = pd.read_csv(T.HANDCHECK)
    g = pd.read_csv(tables.path / "heckles_gendered.csv").set_index("target_gender")
    rw, rm = float(g.loc["Women", "confirmed_broad"]), float(g.loc["Men", "confirmed_broad"])
    picks = sorted(({"idx": i, "en": en, "he": str(hc.loc[i, "heckle"]), "year": int(hc.loc[i, "year"]),
                     "who": f"{ONE[hc.loc[i, 'heckler_gender']]}, heckling {ONE[hc.loc[i, 'target_gender']]}",
                     "at_women": hc.loc[i, "target_gender"] == "Women"} for i, en in HECKLES),
                   key=lambda p: p["year"])

    title = "Some of the heckles about looks or sex, each confirmed by reading"
    sub = (f"Flagged by the model and confirmed by reading: about {_in_words(rw)} heckles at women is of this kind, "
           f"and {_in_words(rm)} at men. Translations are ours")
    page = BlogPage(title, sub, B.sources(B.SRC_CORPUS, "Jev flags, each read by hand"))
    x_text = page.M + 0.22
    col_w = page.W - page.M - x_text
    lh_en, lh_he, lh_cap = 13.5 * 1.22 / 72, 11 * 1.32 / 72, 10 * 1.25 / 72
    y = page.top + 0.02
    for k, p in enumerate(picks):
        y0 = y
        for line in page.wrap(p["en"], 13.5, serif=True, width=col_w, italic=True):
            page.text(x_text, y, line, size=13.5, serif=True, italic=True, va="top")
            y += lh_en
        y += 0.04
        # Hebrew in logical order, as matplotlib runs the bidi algorithm itself
        page.bg.text(page.W - page.M, y, p["he"], fontsize=11, color=B.SEC, ha="right", va="top", family=B.SANS)
        y += lh_he + 0.03
        page.text(x_text, y, f"{p['year']} · {p['who']}", size=10, colour=B.CAP, va="top")
        y += lh_cap
        page.bg.add_line(Line2D([page.M + 0.04, page.M + 0.04], [y0 + 0.03, y - 0.02], color=B.WOMEN, lw=1.0,
                                solid_capstyle="butt"))
        if k < len(picks) - 1:
            y += 0.15
    page.fit(y)

    years = [p["year"] for p in picks]
    pretty = next(p["year"] for p in picks if p["idx"] == 184)
    alt = (f"{T.number_word(len(picks)).capitalize()} heckles shouted in the Knesset plenum between {min(years)} and "
           f"{max(years)}, {T.number_word(sum(p['at_women'] for p in picks))} at women members and one by a woman at a "
           f"man, among them “Be pretty and shut up” ({pretty}). Each is in English with the original Hebrew beneath; "
           f"about {_in_words(rw)} heckles at women is of this kind, against {_in_words(rm)} at men.")
    return page.save("sexist_heckles_card", alt=alt)


def draw_netanyahu_by_party(tables: T.Tables) -> list[Path]:
    d = pd.read_csv(tables.path / "netanyahu_by_party.csv")
    d["name"] = d["party"].replace(NAMES)
    # outside his camp by attacks, then his camp by defences
    d["order"] = np.where(d["in_camp"], 1 + d["share_defend"], -d["share_attack"])
    d = d.sort_values("order").reset_index(drop=True)

    title = "His opponents attack Netanyahu far more than they defend him; his allies lean the other way"
    # personal attacks only, not policy disagreement (config/dimensions_bibi.yaml)
    sub = (f"Share of each party's plenary speeches naming Netanyahu that attack him personally or defend him, "
           f"2019–2026*. From {T.round_count(d['n_speeches'].sum())} speeches")
    page = BlogPage(title, sub, B.sources(B.SRC_CORPUS, "Jev classifications of plenary speeches naming Netanyahu",
                                          partial=True))
    rowh = 0.28
    name_x = page.M + 0.17
    name_w = max(page.width_of(s, B.T_LABEL, weight=500) for s in d["name"]) + 0.17 + 0.12
    val_w = page.width_of("74%", B.T_VALUE, weight=600) + 0.08
    area_l, area_r = page.M + name_w, page.W - page.M
    amax, dmax = float(d["share_attack"].max()), float(d["share_defend"].max())
    s = (area_r - area_l - 2 * val_w) / (amax + dmax)            # inches per unit share
    cx = area_l + val_w + amax * s
    top = page.top + 0.02
    page.text(cx - 0.08, top + 0.12, "Attack him personally", size=B.T_LABEL, weight=500, colour=B.OPP, ha="right",
              va="center")
    page.text(cx + 0.08, top + 0.12, "Defend or praise him", size=B.T_LABEL, weight=500, colour=B.GOV, ha="left",
              va="center")
    first_in_camp = d.index[d["in_camp"]][0]
    y = first_row = top + 0.34
    for i, r in d.iterrows():
        if i == first_in_camp:
            y += 0.06
            page.bg.plot([page.M, page.W - page.M], [y, y], color=B.RULE, lw=0.9, solid_capstyle="butt")
            page.note(page.M, y + 0.06, "his camp", va="top")
            y += 0.34 - 0.06
        yc = y + rowh / 2
        page.bg.scatter([page.M + 0.06], [yc], s=B.DOT_S * 0.75, marker="o", color=B.BLOC.get(r["bloc"], B.OTHER),
                        zorder=3)
        page.text(name_x, yc, r["name"], size=B.T_LABEL, weight=500, va="center")
        bh = rowh * 0.62
        a, df = float(r["share_attack"]), float(r["share_defend"])
        page.bg.add_patch(Rectangle((cx - a * s, yc - bh / 2), a * s, bh, facecolor=B.OPP, edgecolor="none", zorder=2))
        page.bg.add_patch(Rectangle((cx, yc - bh / 2), df * s, bh, facecolor=B.GOV, edgecolor="none", zorder=2))
        page.text(cx - a * s - 0.05, yc, T.pct(a), size=B.T_VALUE, weight=600, ha="right", va="center", colour=B.INK)
        page.text(cx + df * s + 0.05, yc, T.pct(df), size=B.T_VALUE, weight=600, ha="left", va="center", colour=B.INK)
        y += rowh
    page.bg.plot([cx, cx], [first_row - 0.02, y + 0.02], color=B.INK, lw=0.9, zorder=4, solid_capstyle="butt")
    page.fit(y + 0.02)

    lab = d.set_index("name")
    others = d[d["in_camp"] & (d["name"] != "Likud")]
    alt = (f"Diverging bars for {len(d)} parties' plenary speeches naming Netanyahu, 2019 to 2026. Parties outside his "
           f"camp attack him personally far more often than they defend him (Labour in "
           f"{T.pct(lab.loc['Labour', 'share_attack'])} of such speeches against "
           f"{T.pct(lab.loc['Labour', 'share_defend'])}). Inside it, Likud defends him in "
           f"{T.pct(lab.loc['Likud', 'share_defend'])} and attacks him in {T.pct(lab.loc['Likud', 'share_attack'])}, "
           f"while {', '.join(others['name'].iloc[:-1])} and {others['name'].iloc[-1]} defend him in "
           f"{T.pct(others['share_defend'].min())} to {T.pct(others['share_defend'].max())} and leave most mentions "
           f"neutral.")
    return page.save("netanyahu_by_party", alt=alt)


CHARTS = {
    "women_on_the_floor": draw_women_on_the_floor,
    "heckling_women_and_men": draw_heckling_women_and_men,
    "sexist_heckles_card": draw_sexist_heckles_card,
    "netanyahu_by_party": draw_netanyahu_by_party,
}
