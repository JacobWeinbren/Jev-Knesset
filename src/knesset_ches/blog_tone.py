"""Charts on tone, attacks, the agenda, committee rooms and corruption.

    python -m knesset_ches.blog tone
"""
from __future__ import annotations

import datetime as dt
import functools
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

from knesset_ches import aggregate as agg
from knesset_ches import blog_style as B
from knesset_ches import tables as T

THIN_SPEECHES = 2000
THIN_MEMBERS = 5
LW = 2.4
DASH = (0, (3.4, 1.7))         # government side and committee
SAND = "#f1e7d8"


def darker(colour: str, k: float = 0.35) -> str:
    """A light colour mixed towards ink, for text."""
    a = np.array(mcolors.to_rgb(colour))
    b = np.array(mcolors.to_rgb(B.INK))
    return mcolors.to_hex(a * (1 - k) + b * k)


def date_x(year: int, month: int, day: int) -> float:
    """A date as x on a year axis with points at mid-year."""
    d = dt.date(year, month, day)
    start = dt.date(year, 1, 1)
    days = (dt.date(year + 1, 1, 1) - start).days
    return year + (d - start).days / days - 0.5


def speech_table(tables: T.Tables) -> pd.DataFrame:
    df = pd.read_csv(tables.path / "speech_by_year.csv")
    df["year"] = df["period"].astype(int)
    return df


def by_side(sp: pd.DataFrame, item: str) -> pd.DataFrame:
    """Plenary years by coalition and opposition for one speech item."""
    s = sp[(sp["protocol_type"] == "plenary") & (sp["actor_type"] == "coalition_status") & (sp["item"] == item)]
    return s.pivot_table(index="year", columns="actor_id", values="value").sort_index()


def knesset_wide(sp: pd.DataFrame, item: str, protocol_type: str = "plenary") -> pd.Series:
    s = sp[(sp["protocol_type"] == protocol_type) & (sp["actor_type"] == "knesset_wide") & (sp["item"] == item)]
    return s.set_index("year")["value"].sort_index()


def thin_years(tables: T.Tables, protocol_type: str) -> set[int]:
    cov = pd.read_csv(tables.path / "speech_coverage.csv")
    cov = cov[(cov["protocol_type"] == protocol_type) & (cov["n_speeches"] < THIN_SPEECHES)]
    return set(cov["year"].astype(int))


def n_speech_sentences(tables: T.Tables, protocol_types: Sequence[str], first: int, last: int) -> int:
    cov = pd.read_csv(tables.path / "speech_coverage.csv")
    cov = cov[cov["protocol_type"].isin(protocol_types) & cov["year"].between(first, last)]
    return int(cov["n_sentences"].sum())


def side_sentences() -> int:
    u = pd.read_parquet(T.UNITS, columns=["protocol_type", "is_chairman", "coalition", "year", "n_sentences"])
    u = u[(u["protocol_type"] == "plenary") & ~u["is_chairman"].fillna(False).astype(bool)
          & u["coalition"].astype(str).isin(["coalition", "opposition"]) & u["year"].between(1992, 2026)]
    return int(u["n_sentences"].sum())


def runs(years: Iterable[int]) -> list[list[int]]:
    out: list[list[int]] = []
    for y in years:
        if out and y == out[-1][-1] + 1:
            out[-1].append(y)
        else:
            out.append([y])
    return out


def series(ax: Any, s: pd.Series, colour: str, ls: Any = "-", marker: str = "o", thin: Iterable[int] = (),
           lw: float = LW, zorder: float = 4) -> None:
    """One yearly line. Thin years get hollow marks and faint segments."""
    s = s.dropna().sort_index()
    thin = {int(y) for y in thin if y in s.index}
    for r in runs(int(y) for y in s.index):
        strong: list[list[int]] = [[]]
        for a, b in zip(r[:-1], r[1:]):
            if a in thin or b in thin:
                ax.plot([a, b], [s.loc[a], s.loc[b]], color=colour, lw=B.LW_CONTEXT, ls=ls, solid_capstyle="round",
                        dash_capstyle="butt", zorder=zorder - 0.1)
                strong.append([])
            else:
                strong[-1] += [a, b] if not strong[-1] else [b]
        for run in strong:
            if len(run) > 1:
                ax.plot(run, s.loc[run].to_numpy(), color=colour, lw=lw, ls=ls, solid_capstyle="round",
                        dash_capstyle="butt", solid_joinstyle="round", zorder=zorder)
    if thin:
        ty = sorted(thin)
        ax.scatter(ty, s.loc[ty].to_numpy(), s=B.DOT_S, marker=marker, facecolor=B.PAPER, edgecolor=colour,
                   linewidths=1.4, zorder=zorder + 1, clip_on=False)
    for y in (s.index.min(), s.index.max()):
        if y not in thin:
            ax.scatter([y], [s.loc[y]], s=B.DOT_S, marker=marker, color=colour, lw=0, zorder=zorder + 1, clip_on=False)


def event(ax: Any, x: float) -> None:
    ax.plot([x, x], list(ax.get_ylim()), color=B.RULE, lw=0.9, zorder=1, solid_capstyle="butt")


def band(ax: Any, start: tuple[int, int, int], end: tuple[int, int, int], top: float) -> tuple[float, float]:
    """Shade a period from the baseline to `top` and return its x span."""
    x0, x1 = date_x(*start), date_x(*end)
    ax.add_patch(Rectangle((x0, 0), x1 - x0, top, fc=B.SHADE, ec="none", zorder=0.2))
    return x0, x1


note = functools.partial(B.ax_note, ha="left", va="top", zorder=8, clip_on=False)
label = functools.partial(B.ax_label, ha="left", va="center", weight=500, zorder=9, clip_on=False)
value = functools.partial(label, size=B.T_VALUE, weight=600)


def spread_in(ax: Any, height_in: float, ys: Sequence[float], gap_in: float) -> list[float]:
    """B.spread with the gap in inches of an axes `height_in` tall."""
    y0, y1 = ax.get_ylim()
    per_in = (y1 - y0) / height_in
    return B.spread(list(ys), gap_in * per_in, y0, y1)


def ypad(ax: Any) -> None:
    ax.tick_params(axis="y", pad=7)
    ax.tick_params(axis="x", pad=5)


def panel_title(page: B.BlogPage, y: float, s: str) -> None:
    page.text(page.M, y, s, size=B.T_PANEL, weight=600, serif=True, va="baseline")


def stacked(page: B.BlogPage, titles: Sequence[str], heights: Sequence[float]) -> list[Any]:
    left = page.M + 0.42
    width = page.inner - 0.42 - 0.52
    y = page.top
    axes = []
    for t, h in zip(titles, heights):
        panel_title(page, y + 0.16, t)
        top = y + 0.30
        axes.append(page.axes(left, top, width, h))
        ypad(axes[-1])
        y = top + h + 0.30
    page.fit(y)
    return axes


def draw_tone_by_side(tables: T.Tables) -> list[Path]:
    sp = speech_table(tables)
    host = by_side(sp, "hostility").loc[1992:2026]
    deleg = by_side(sp, "delegitimisation").loc[1992:2026]
    gov, opp = host["coalition"], host["opposition"]
    dg, do = deleg["coalition"], deleg["opposition"]
    same = pd.read_csv(tables.path / "same_members.csv").set_index("item").loc["hostility"]
    thin = thin_years(tables, "plenary")

    title = "Thirty steady years, then harsher debate from 2021: first in opposition, then in government"
    sub = (f"Plenary speeches, yearly; the same {int(same['n_members'])} members who spoke in 2015–20 and since 2021 "
           f"went from {same['before']:.1f} to {same['after']:.1f} on hostility. "
           f"{B.from_sentences(side_sentences()).capitalize()}")
    # the short corpus name keeps the footer to two lines
    src = B.sources(B.SRC_CORPUS_SHORT, f"Jev model, after {B.SRC_TONE} and {B.SRC_DELEG}", partial=True)
    page = B.BlogPage(title, sub, src)
    heights = (1.75, 1.6)
    ax1, ax2 = stacked(page, ["Hostility toward opponents, on a 0–10 scale",
                              "Speeches casting opponents as illegitimate"], heights)
    ax1.set_ylim(0, 8)
    B.hgrid(ax1, [0, 2, 4, 6, 8], baseline=0)
    ax2.set_ylim(0, 0.30)
    B.hgrid(ax2, [0, 0.1, 0.2, 0.3], fmt=T.pct, baseline=0)
    B.year_axis(ax1)
    B.year_axis(ax2)
    ax1.set_xticklabels([])

    x_oct = date_x(2023, 10, 7)
    for ax in (ax1, ax2):
        x_lapid, _ = band(ax, (2021, 6, 13), (2022, 12, 29), ax.get_ylim()[1])
        event(ax, x_oct)
    note(ax1, x_lapid - 0.35, 7.85, "Bennett–Lapid government, 2021–22", ha="right")
    note(ax1, x_oct + 0.35, 7.85, "7 October", ha="left")
    for ax, df in ((ax1, host), (ax2, deleg)):
        series(ax, df["opposition"], B.OPP, "-", "s", thin)
        series(ax, df["coalition"], B.GOV, DASH, "o", thin)
    note(ax1, 2019.4, min(gov.loc[2012:2019].min(), opp.loc[2012:2019].min()) - 0.45,
         "hollow: 2019, two\nelections, few sittings", ha="right", va="top", linespacing=1.05)

    label(ax1, 1992.3, opp.loc[1992:1999].max() + 0.45, "Opposition", colour=B.OPP, va="bottom")
    label(ax1, 1992.3, gov.loc[1992:1999].min() - 0.35, "Government side", colour=B.GOV, va="top")
    label(ax2, 2020.3, do.loc[2021] - 0.012, "Opposition", colour=B.OPP, ha="right")
    label(ax2, 2020.6, dg.loc[2020:2022].min() - 0.02, "Government side", colour=B.GOV, ha="right", va="top")
    xe = 2026 + 0.75
    ys = spread_in(ax1, heights[0], [opp.loc[2026], gov.loc[2026]], 0.2)
    value(ax1, xe, ys[0], f"{opp.loc[2026]:.1f}", colour=B.OPP)
    value(ax1, xe, ys[1], f"{gov.loc[2026]:.1f}", colour=B.GOV)
    value(ax2, xe, (dg.loc[2026] + do.loc[2026]) / 2, f"both {T.pct(do.loc[2026])}")

    g_old, g_new = gov.loc[1992:2022], gov.loc[2023:2026]
    o_old, o_new = opp.loc[1992:2020], opp.loc[2021:2026]
    alt = (f"Two line charts, 1992 to 2026. Opposition hostility toward opponents ran between {o_old.min():.1f} and "
           f"{o_old.max():.1f} until 2020 and {o_new.min():.1f} to {o_new.max():.1f} since 2021; on the government "
           f"side it ran between {g_old.min():.1f} and {g_old.max():.1f} until 2022 and {g_new.min():.1f} to "
           f"{g_new.max():.1f} since 2023. The share of speeches casting opponents as illegitimate rose from about one "
           f"in ten to about one in five on both sides.")
    return page.save("tone_by_side", alt=alt)


def draw_attacks_new_targets(tables: T.Tables) -> list[Path]:
    sp = speech_table(tables)
    courts = by_side(sp, "attacks_courts").loc[1992:2026]
    relig = by_side(sp, "attacks_religious").loc[1992:2026]
    thin = thin_years(tables, "plenary")

    title = "Record attacks since 2023: the government on the courts, the opposition on the Charedim"
    sub = f"Share of each side's plenary speeches, yearly. {B.from_sentences(side_sentences()).capitalize()}"
    page = B.BlogPage(title, sub, B.sources(B.SRC_CORPUS, "Jev model, the project's own attack items", partial=True))
    heights = (1.6, 1.6)
    ax1, ax2 = stacked(page, ["Attacking the courts or legal advisers",
                              "Attacking religious or Charedi Jews or their parties as a group"], heights)
    for ax in (ax1, ax2):
        ax.set_ylim(0, 0.25)
        B.hgrid(ax, [0, 0.05, 0.10, 0.15, 0.20, 0.25], fmt=T.pct, baseline=0)
        B.year_axis(ax)
    ax1.set_xticklabels([])
    x_overhaul, x_draft = date_x(2023, 1, 4), date_x(2024, 6, 25)
    event(ax1, x_overhaul)
    note(ax1, x_overhaul - 0.35, 0.245, "judicial overhaul, January 2023", ha="right")
    event(ax2, x_draft)
    note(ax2, x_draft - 0.35, 0.245, "High Court orders Charedim drafted, June 2024", ha="right")
    for ax, df in ((ax1, courts), (ax2, relig)):
        series(ax, df["opposition"], B.OPP, "-", "s", thin)
        series(ax, df["coalition"], B.GOV, DASH, "o", thin)

    # each panel names only the side setting the record
    label(ax1, 2022.2, courts["coalition"].loc[2023] + 0.03, "Government side", colour=B.GOV, ha="right")
    label(ax2, 2022.2, relig["opposition"].loc[2023] + 0.035, "Opposition", colour=B.OPP, ha="right")
    xe = 2026 + 0.75
    for ax, df, h in ((ax1, courts, heights[0]), (ax2, relig, heights[1])):
        o, g = df["opposition"].loc[2026], df["coalition"].loc[2026]
        ys = spread_in(ax, h, [o, g], 0.2)
        value(ax, xe, ys[0], T.pct(o), colour=B.OPP)
        value(ax, xe, ys[1], T.pct(g), colour=B.GOV)

    gc_old, gc_new = courts["coalition"].loc[:2022], courts["coalition"].loc[2023:]
    or_old, or_new = relig["opposition"].loc[:2022], relig["opposition"].loc[2023:]
    alt = (f"Two line charts, 1992 to 2026. Government-side speeches attacking the courts or legal advisers ran at "
           f"{gc_old.min() * 100:.0f}–{gc_old.max() * 100:.0f}% until 2022 and "
           f"{gc_new.min() * 100:.0f}–{gc_new.max() * 100:.0f}% since 2023. Opposition speeches attacking religious or "
           f"Charedi Jews or their parties as a group ran at {or_old.min() * 100:.0f}–{or_old.max() * 100:.0f}% until "
           f"2022 and rose to {or_new.loc[2026] * 100:.0f}% in January to July 2026.")
    return page.save("attacks_new_targets", alt=alt)


RELIGION = B.HAREDI
RELIGION_DASH = (0, (4.6, 1.8))
USUAL_DASH = (0, (1.2, 1.9))
INTIFADA = ((2000, 9, 28), (2005, 2, 8))   # from the Temple Mount riots to the Sharm el-Sheikh summit


def draw_security_after_october_7(tables: T.Tables) -> list[Path]:
    sp = speech_table(tables)
    sec = knesset_wide(sp, "topic_security").loc[1992:2026]
    eco = knesset_wide(sp, "topic_economy").loc[1992:2026]
    rel = knesset_wide(sp, "topic_religion").loc[1992:2026]
    thin = thin_years(tables, "plenary")
    usual = float(sec.loc[1992:2022].mean())
    v24, r26, r_prev = float(sec.loc[2024]), float(rel.loc[2026]), rel.loc[:2025]
    n = n_speech_sentences(tables, ["plenary"], 1992, 2026)

    title = "After 7 October, security took three plenary speeches in five, twice its usual share"
    sub = ("Share of plenary speeches substantially about each subject, yearly; a speech can touch several. "
           f"{B.from_sentences(n).capitalize()}")
    src = B.sources(B.SRC_CORPUS, "Jev model", "topics after the Comparative Agendas Project", partial=True)
    page = B.BlogPage(title, sub, src)
    left_pad, h = 0.42, 3.0
    ax = page.axes(page.M + left_pad, page.top, page.inner - left_pad - 1.05, h)
    ypad(ax)
    page.fit(page.top + h + 0.30)
    top = 0.70
    ax.set_ylim(0, top)
    B.hgrid(ax, [0, 0.2, 0.4, 0.6], fmt=T.pct, baseline=0)
    B.year_axis(ax)
    xi0, xi1 = band(ax, *INTIFADA, top)
    note(ax, (xi0 + xi1) / 2, top - 0.01, "second intifada", ha="center")
    x_oct = date_x(2023, 10, 7)
    event(ax, x_oct)
    note(ax, x_oct - 0.35, top - 0.01, "7 October", ha="right")

    ax.plot([1991.4, 2026.6], [usual, usual], color=B.INK, lw=1.1, ls=USUAL_DASH, dash_capstyle="round", zorder=3)
    # the economy kept light, as it often crosses security before 2020
    series(ax, eco, B.ECONOMY, "-", "o", thin, lw=1.3, zorder=4)
    series(ax, rel, RELIGION, RELIGION_DASH, "D", thin, lw=1.8, zorder=4.5)
    series(ax, sec, B.CONFLICT, "-", "o", thin, lw=B.LW_STORY, zorder=5)
    note(ax, 2019.4, float(rel.loc[2012:2019].min()) - 0.022, "hollow: 2019, two\nelections, few sittings",
         ha="right", va="top", linespacing=1.05)
    value(ax, 2024.3, v24 + 0.012, T.pct(v24), colour=B.CONFLICT, ha="left", va="bottom")
    label(ax, 2022.4, 0.52, "The conflict, security and defence", colour=B.CONFLICT, ha="right")
    label(ax, 1992.3, rel.loc[1992:2004].min() - 0.018, "Religion and state", colour=darker(RELIGION, 0.3), va="top")
    xe = 2026 + 0.75
    ends = [(float(sec.loc[2026]), T.pct(sec.loc[2026]), B.CONFLICT, value),
            (usual, f"usual share, {T.pct(usual)}", B.INK, None),
            (float(eco.loc[2026]), "The economy", darker(B.ECONOMY), label),
            (r26, T.pct(r26), darker(RELIGION, 0.3), value)]
    for (_, s, colour, draw), y in zip(ends, spread_in(ax, h, [e[0] for e in ends], 0.2)):
        if draw is None:
            note(ax, xe, y, s, va="center", colour=colour)
        else:
            draw(ax, xe, y, s, colour=colour)

    alt = (f"Line chart, 1992 to 2026. The share of plenary speeches about the conflict, security or defence averaged "
           f"{T.pct(usual)} from 1992 to 2022 and never passed {T.pct(sec.loc[:2022].max())}, a high reached in "
           f"{sec.loc[:2022].idxmax()} during the second intifada. It reached {T.pct(v24)} in 2024, "
           f"{T.pct(sec.loc[2025])} in 2025 and {T.pct(sec.loc[2026])} in January to July 2026, while the economy's "
           f"share fell to about a quarter. Religion and state reached a record {T.pct(r26)} in January to July 2026, "
           f"above its previous high of {T.pct(r_prev.max())} in {r_prev.idxmax()}.")
    return page.save("security_after_october_7", alt=alt)


def draw_committee_calmer(tables: T.Tables) -> list[Path]:
    sp = speech_table(tables)
    first, last = 1999, 2023
    plen = knesset_wide(sp, "hostility", "plenary").loc[first:last]
    comm = knesset_wide(sp, "hostility", "committee").loc[first:last]
    n = n_speech_sentences(tables, ["plenary", "committee"], first, last)

    # the title says remarks, not members, as committee remarks are shorter and length may explain some of the gap
    title = "Committee remarks are about half as hostile as plenary speeches, and the gap has widened"
    sub = (f"Plenary speeches and the shorter committee remarks, yearly; the committee record ends in March 2024. "
           f"{B.from_sentences(n).capitalize()}")
    page = B.BlogPage(title, sub, B.sources(B.SRC_CORPUS, "Jev model", f"hostility after {B.SRC_TONE}"))
    left_pad, h = 0.40, 2.5
    ax = page.axes(page.M + left_pad, page.top + 0.36, page.inner - left_pad - 1.35, h)
    ypad(ax)
    panel_title(page, page.top + 0.16, "Hostility toward opponents, on a 0–10 scale")
    page.fit(page.top + 0.36 + h + 0.30)
    ax.set_ylim(0, 6)                   # the data never pass 5
    B.hgrid(ax, [0, 2, 4, 6], baseline=0)
    ax.set_xlim(first - 0.6, last + 0.6)
    ticks = [1999, 2005, 2011, 2017, 2023]
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks], fontsize=B.T_TICK, color=B.CAP, family=B.SANS)

    yrs = list(plen.index)
    ax.fill_between(yrs, comm.loc[yrs].to_numpy(), plen.loc[yrs].to_numpy(), color=SAND, lw=0, zorder=1)
    series(ax, plen, B.PLENUM, "-", "o", thin_years(tables, "plenary"))
    series(ax, comm, B.COMMITTEE, DASH, "o", thin_years(tables, "committee"))
    xe = last + 0.75
    ys = spread_in(ax, h, [plen.loc[last], comm.loc[last]], 0.2)
    label(ax, xe, ys[0], f"In the plenum {plen.loc[last]:.1f}", colour=B.PLENUM)
    label(ax, xe, ys[1], f"In committee {comm.loc[last]:.1f}", colour=darker(B.COMMITTEE))
    note(ax, 2019, comm.loc[2015:2023].min() - 0.22, "2019: two elections,\nfew sittings", ha="center",
         multialignment="center", linespacing=1.1)

    alt = (f"Line chart, {first} to {last}. Hostility in committee stayed between {comm.min():.1f} and "
           f"{comm.max():.1f}, while in the plenum it rose from {plen.loc[first]:.1f} to {plen.loc[last]:.1f}, most of "
           f"it after 2020. Committee remarks, which are shorter, are about half as hostile as plenary speeches on "
           f"average.")
    return page.save("committee_calmer", alt=alt)


# keyed to Likud, as the Left still led on corruption while Labour sat in Likud-led governments
NO_LIKUD = [((1992, 7, 13), (1996, 6, 18), "Rabin\nand Peres"), ((1999, 7, 6), (2001, 3, 7), "Barak"),
            ((2006, 1, 4), (2009, 3, 31), "Olmert"), ((2021, 6, 13), (2022, 12, 29), "Bennett\nand Lapid")]


def corruption_sentences() -> int:
    """Plenary sentences, 1992 to 2026, in Left and Right packs with a corruption score."""
    packs = pd.read_parquet(T.ROOT / "data" / "corpus" / "processed" / "packs.parquet",
                            columns=["unit_id", "faction_id", "knesset", "year", "n_sentences"])
    packs = packs[packs["unit_id"].isin(B.usable_units(["corrupt_salience"]))]
    bloc = agg.project_blocs(packs, pd.Series("Other", index=packs.index))
    return int(packs.loc[bloc.isin(["Left", "Right"]) & packs["year"].between(1992, 2026), "n_sentences"].sum())


def draw_corruption_follows_opposition(tables: T.Tables) -> list[Path]:
    sal = pd.read_csv(tables.path / "salience.csv", dtype={"period": str, "actor_id": str}, low_memory=False)
    sal = sal[(sal["dimension"] == "corrupt_salience") & (sal["period_type"] == "year") & (sal["actor_type"] == "bloc")
              & sal["sufficient"].astype(bool)]
    sal = sal.assign(year=sal["period"].astype(int))
    blocs = sal.pivot_table(index="year", columns="actor_id", values="salience")
    n_mks = sal.pivot_table(index="year", columns="actor_id", values="n_mks")
    left, right = blocs["Left"].loc[1992:2026], blocs["Right"].loc[1992:2026]
    thin_left = [y for y in left.index if n_mks.loc[y, "Left"] < THIN_MEMBERS]

    title = "The Right talks more of corruption when Likud is out of power, the Left when Likud governs"
    sub = ("Left and Right blocs' attention to corruption (0 to 10) in plenary speeches, yearly; shaded: governments "
           f"without Likud. {B.from_sentences(corruption_sentences()).capitalize()}")
    src = B.sources(B.SRC_CORPUS, "Jev model, scoring how much speeches dwell on corruption", partial=True)
    page = B.BlogPage(title, sub, src)
    left_pad, h = 0.40, 2.8
    top = page.top + 0.30
    ax = page.axes(page.M + left_pad, top, page.inner - left_pad - 0.85, h)
    ypad(ax)
    page.fit(top + h + 0.30)
    ax.set_ylim(0, 5.12)
    B.hgrid(ax, [0, 1, 2, 3, 4, 5], baseline=0)
    B.year_axis(ax)
    page.note(page.M, top - 0.12, "more attention ↑")
    for start, end, name in NO_LIKUD:
        x0, x1 = band(ax, start, end, 5.12)
        note(ax, (x0 + x1) / 2, 4.97, name, ha="center", multialignment="center", linespacing=1.1, halo=False)
    series(ax, right, B.BLOC["Right"], "-", "o", ())
    series(ax, left, B.BLOC["Left"], "-", "o", thin_left)
    xe = 2026 + 0.75
    ys = spread_in(ax, h, [left.loc[2026], right.loc[2026]], 0.2)
    label(ax, xe, ys[0], f"Left {left.loc[2026]:.1f}", colour=B.BLOC["Left"])
    label(ax, xe, ys[1], f"Right {right.loc[2026]:.1f}", colour=B.BLOC["Right"])
    note(ax, xe, ys[0] - 0.22, "four Labour\nmembers since\n2023", va="top", linespacing=1.1)

    alt = (f"Line chart of the Left and Right blocs' attention to corruption, 1992 to 2026. The Right talks about it "
           f"more in the four periods when Likud was out of government (1992–96, 1999–2001, 2006–09, 2021–22), and "
           f"the Left the rest of the time. Since 2023 the Left is Labour's four members alone, at about "
           f"{left.loc[2025]:.0f} in 2025–26 against the Right's {right.loc[2025]:.0f}.")
    return page.save("corruption_follows_opposition", alt=alt)


CHARTS = {
    "tone_by_side": draw_tone_by_side,
    "attacks_new_targets": draw_attacks_new_targets,
    "security_after_october_7": draw_security_after_october_7,
    "committee_calmer": draw_committee_calmer,
    "corruption_follows_opposition": draw_corruption_follows_opposition,
}
