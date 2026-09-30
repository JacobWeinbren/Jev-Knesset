"""What each bloc talks about, attacks on Arab citizens, and dealignment.

    python -m knesset_ches.blog extra

Dealignment follows Shamir, Ventura, Arian and Kedar (2008).
"""
from __future__ import annotations

import functools
from pathlib import Path

import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

from knesset_ches import aggregate as agg
from knesset_ches import blog_style as B
from knesset_ches import tables as T

_label = functools.partial(B.ax_label, weight=500, zorder=9, clip_on=False)


# issues_by_bloc

TOPIC_LABEL = {          # line breaks set by hand
    "topic_security": "Security, defence\nand the conflict",
    "topic_law": "Law, courts and\ninstitutions",
    "topic_economy": "The economy and\nthe budget",
    "topic_minorities": "Arab citizens and\ncivil rights",
    "topic_welfare": "Welfare, health\nand housing",
    "topic_religion": "Religion and state",
    "topic_education": "Education\nand culture",
    "topic_environment": "Environment and\ntransport",
    "topic_immigration": "Immigration",
}
COLUMNS = {"Right": "Right", "Orthodox": "Orthodox", "Secular Centre": "Secular\nCentre", "Left": "Left",
           "Arab-Israeli": "Arab-\nIsraeli", "all": "All\nmembers"}
JEWISH_BLOCS = ["Right", "Orthodox", "Secular Centre", "Left"]
T_ROW = 11.0
HEAT = mcolors.LinearSegmentedColormap.from_list("seq", ["#eaf2f2", "#cfe3e8", "#8fbcc8", "#3f8fa3", "#164a56"])
HEAT_MAX = 0.70


def heat_fill(v: float) -> str:
    """Cell fill, darkened where neither ink nor white text would reach 4.5:1."""
    t = min(1.0, v / HEAT_MAX)
    face = mcolors.to_hex(HEAT(t))
    while max(B.contrast(face, B.INK), B.contrast(face, "#ffffff")) < 4.5 and t < 1.0:
        t = min(1.0, t + 0.005)
        face = mcolors.to_hex(HEAT(t))
    return face


def draw_issues_by_bloc(tables: T.Tables) -> list[Path]:
    sk = pd.read_csv(tables.path / "speech_by_knesset.csv", dtype={"period": str})
    x = sk[(sk["protocol_type"] == "plenary") & (sk["period"] == "25") & sk["actor_type"].isin(["bloc", "knesset_wide"])
           & sk["item"].str.startswith("topic_")]
    cols = list(COLUMNS)
    tab = x.pivot_table(index="item", columns="actor_id", values="value")[cols].sort_values("all", ascending=False)
    units = pd.read_parquet(T.UNITS, columns=["knesset", "is_chairman", "n_sentences"])
    n = int(units.loc[(units["knesset"] == 25) & ~units["is_chairman"].fillna(False).astype(bool), "n_sentences"].sum())

    title = "Every Jewish bloc talks most about security and the law, Arab members about Arab citizens"
    sub = (f"Share of each bloc's plenary speeches that discuss each subject, 2022–26*; a speech can touch several. "
           f"{B.from_sentences(n).capitalize()}")
    src = B.sources(B.SRC_CORPUS_SHORT, "Jev model classifications; subjects after the Comparative Agendas Project "
                    "codebook", partial=True)
    page = B.BlogPage(title, sub, src)
    label_w = max(page.width_of(line, T_ROW, weight=500) for lab in TOPIC_LABEL.values() for line in lab.split("\n"))
    label_w += 0.12
    gap, rh, pad = 0.14, 0.47, 0.022          # room for two-line labels
    cw = (page.inner - label_w - gap) / len(cols)
    y_table = page.top + 0.66
    page.fit(y_table + len(tab) * rh)
    xs = [page.M + label_w + (j + 0.5) * cw + (gap if c == "all" else 0) for j, c in enumerate(cols)]

    # one text per header line, so a two-line name doesn't crowd its neighbours
    y_dot = y_table - 0.10
    for c, xc in zip(cols, xs):
        if c != "all":
            page.bg.scatter([xc], [y_dot], s=B.DOT_S, color=B.BLOC[c], lw=0, zorder=4)
        for k, line in enumerate(reversed(COLUMNS[c].split("\n"))):
            page.text(xc, y_dot - 0.13 - k * B.T_KEY * 1.2 / 72, line, size=B.T_KEY, weight=500, ha="center")
    x_rule = page.M + label_w + 5 * cw + gap / 2
    page.bg.plot([x_rule, x_rule], [page.top - 0.02, y_table + len(tab) * rh], color=B.RULE, lw=0.9, zorder=1,
                 solid_capstyle="butt")
    for i, (item, row) in enumerate(tab.iterrows()):
        y = y_table + i * rh
        page.text(page.M, y + rh / 2, TOPIC_LABEL[item], size=T_ROW, weight=500, va="center", linespacing=1.1)
        for c, xc in zip(cols, xs):
            v = float(row[c])
            face = heat_fill(v)
            page.bg.add_patch(Rectangle((xc - cw / 2 + pad, y + pad), cw - 2 * pad, rh - 2 * pad, fc=face, ec="none",
                                        zorder=2))
            ink = B.INK if B.contrast(face, B.INK) >= B.contrast(face, "#ffffff") else "#ffffff"
            page.text(xc, y + rh / 2, f"{v * 100:.0f}%", size=B.T_VALUE, weight=600, colour=ink, ha="center",
                      va="center_baseline")

    law, security = tab.loc["topic_law", JEWISH_BLOCS] * 100, tab.loc["topic_security", JEWISH_BLOCS] * 100
    alt = (f"Table of the nine subjects of plenary speech by bloc, 2022 to 2026. Every Jewish bloc talked most about "
           f"law, the courts and institutions ({law.min():.0f}–{law.max():.0f}%) and security "
           f"({security.min():.0f}–{security.max():.0f}%). Arab members spoke most about Arab citizens, minorities and "
           f"civil rights, in {tab.loc['topic_minorities', 'Arab-Israeli'] * 100:.0f}% of their speeches against "
           f"{tab.loc['topic_minorities', 'all'] * 100:.0f}% for the whole chamber.")
    return page.save("issues_by_bloc", alt=alt)


# anti_arab_attacks

# yearly bars, as quarterly ones are too busy
CAMP = "Right and Orthodox"
RAAM_YEARS = (2021, 2022)       # Ra'am in the coalition, June 2021 to June 2022
# (label lines, label centre x, label bottom y, years)
WAVES = [
    (["Riots and the second", "intifada, 2000–01"], 2000.5, 0.0340, (2000, 2001)),
    (["Arab members visit", "Gaddafi, 2010"], 2009.7, 0.0295, (2010,)),
    (["Knife attacks and Temple", "Mount shooting, 2015–17"], 2014.8, 0.0378, (2015, 2017)),
    (["Ra'am in government,", "2021–22"], 2021.5, 0.0458, RAAM_YEARS),
]


def camp_sentences(quarters: set[str]) -> int:
    """Sentences in Right and Orthodox plenary speeches in `quarters`, chair lines left out."""
    u = pd.read_parquet(T.UNITS, columns=["date", "knesset", "faction_id", "is_chairman", "n_sentences"])
    u = u[~u["is_chairman"].fillna(False).astype(bool)].copy()
    u["bloc"] = agg.project_blocs(u, pd.Series("?", index=u.index))
    u["q"] = pd.to_datetime(u["date"]).dt.to_period("Q").astype(str)
    u = u[u["q"].isin(quarters) & u["bloc"].isin(["Right", "Orthodox"])]
    return int(u["n_sentences"].sum())


def draw_anti_arab_attacks(tables: T.Tables) -> list[Path]:
    t = pd.read_csv(tables.path / "attacks_arab_by_quarter.csv")
    t["year"] = t["quarter"].str[:4].astype(int)
    t = t[t["year"].between(1992, 2026)]
    yearly = t.groupby(["group", "year"])[["n_attacks", "n_speeches"]].sum()
    yearly["share"] = yearly["n_attacks"] / yearly["n_speeches"]
    camp = yearly.loc[CAMP]["share"]
    rest = yearly.loc["Everyone else"]
    n = camp_sentences(set(t["quarter"]))

    # Bare 'attacks' would read as violence. Ra'am held no ministry, so it was in government but not in power.
    title = "Attacks on Arab citizens spike after violence and peaked with an Arab party in government"
    sub = (f"Share of Right and Orthodox members' speeches that attack Arab citizens as a group; other members "
           f"almost never do. {B.from_sentences(n).capitalize()}")
    src = B.sources(B.SRC_CORPUS_SHORT, "Jev classifications; the waves' triggers read in the flagged speeches",
                    partial=True)
    page = B.BlogPage(title, sub, src)
    ax = page.axes(page.M, page.top + 0.05, page.inner - 0.45, 2.75)
    ax.set_ylim(0, 0.055)
    B.year_axis(ax)
    B.hgrid(ax, [0.01, 0.02, 0.03, 0.04, 0.05], fmt=T.pct, label_side="right")
    ax.axhline(0, color=B.INK, lw=0.9, zorder=3)
    for y, v in camp.items():
        ax.bar(y, v, width=0.72, color=B.INK if y in RAAM_YEARS else B.SLATE, zorder=2, lw=0)
    for lines, x, y0, years in WAVES:
        story = years == RAAM_YEARS
        peak = max(years, key=lambda y: camp[y])
        ax.plot([peak, x], [camp[peak] + 0.0008, y0 - 0.0005], color=B.INK if story else B.CAP, lw=0.8, zorder=1)
        ax.text(x, y0, "\n".join(lines), ha="center", va="bottom", fontsize=B.T_NOTE, family=B.SERIF, style="italic",
                linespacing=1.05, color=B.INK if story else B.SEC, path_effects=B.HALO, zorder=5)
    page.fit(page.top + 0.05 + 2.75 + 0.36)

    rest_share = float(rest["n_attacks"].sum() / rest["n_speeches"].sum())
    alt = (f"Bar chart, 1992 to 2026. Speeches attacking Arab citizens, Arab members or Arab parties as a group come "
           f"almost only from Right and Orthodox members: about {camp.median() * 100:.0f}% of their speeches in a "
           f"typical year. It rises in waves: {camp[2000] * 100:.0f}% in 2000 (the October riots and the second "
           f"intifada), {camp[2010] * 100:.0f}% in 2010 (Arab members' visit to Gaddafi), {camp[2017] * 100:.0f}% in "
           f"2017 (knife attacks and the Temple Mount shooting), and a peak of {camp[2022] * 100:.1f}% in 2022, when "
           f"Ra'am sat in the government. Other members' speeches: {rest_share * 100:.1f}%.")
    return page.save("anti_arab_attacks", alt=alt)


# dealignment_shamir

CAMP_OF = {"Right": "Right and Orthodox", "Orthodox": "Right and Orthodox", "Other lists": "Other lists"}
ELECTION_TICK = {"1996-05-29": "1996", "2006-03-28": "2006", "2013-01-22": "2013", "2019-04-09": "April 2019",
                 "2022-11-01": "2022"}
NEWCOMERS = {"2006-03-28": "Kadima\nand Gil", "2013-01-22": "Yesh Atid\nand Hatnuah", "2019-04-09": "Blue and\nWhite"}
PARTY_DEF = "A party and its successors, or a joint list and its parts, count as one party"


def election_results() -> pd.DataFrame:
    e = pd.read_csv(T.ROOT / "data" / "external" / "knesset_election_results.csv", dtype={"components": str})
    e["components"] = e["components"].fillna("")
    e["camp"] = [CAMP_OF.get(b, "Everyone else") for b in e["bloc"]]
    return e


def _families(df: pd.DataFrame) -> list[set[str]]:
    """Each list's lead family and the families it joined."""
    return [{lin} | {c for c in comp.split(";") if c} for lin, comp in zip(df["lineage"], df["components"])]


def pedersen(a: pd.DataFrame, b: pd.DataFrame, key: str) -> float:
    """Pedersen volatility over `key`. Mergers and splits don't count as votes moving (Bartolini and Mair)."""
    fa, fb = a.groupby(key)["vote_share"].sum(), b.groupby(key)["vote_share"].sum()
    parent = {u: u for u in sorted(set(fa.index) | set(fb.index))}

    def find(x: str) -> str:
        while parent[x] != x:
            x = parent[x]
        return x
    if key == "lineage":
        for fam in _families(a) + _families(b):
            fam = [f for f in fam if f in parent]
            for f in fam[1:]:
                parent[find(f)] = find(fam[0])
    ua = fa.groupby(fa.index.map(find)).sum()
    ub = fb.groupby(fb.index.map(find)).sum()
    idx = sorted(set(ua.index) | set(ub.index))
    return float(0.5 * (ub.reindex(idx, fill_value=0) - ua.reindex(idx, fill_value=0)).abs().sum())


def volatility_table(e: pd.DataFrame) -> pd.DataFrame:
    """Per election, volatility between parties and between camps, and seats won by new parties."""
    dates = sorted(e["election_date"].unique())
    rows = []
    for d0, d1 in zip(dates[:-1], dates[1:]):
        a, b = e[e["election_date"] == d0], e[e["election_date"] == d1]
        before: set[str] = set().union(*_families(a[a["seats"] > 0]))
        new = b[(b["seats"] > 0) & ~b["lineage"].isin(before)]
        rows.append(dict(election=d1, family=pedersen(a, b, "lineage"), camps=pedersen(a, b, "camp"),
                         new_seats=int(new["seats"].sum())))
    return pd.DataFrame(rows).set_index("election")


def draw_dealignment_shamir(tables: T.Tables) -> list[Path]:
    v = volatility_table(election_results())
    title = "The vote swings between parties, often to new ones, far more than between the two camps"
    sub = ("Each point is one Knesset election against the one before, 1996–2022; the two camps are Right and Orthodox "
           "parties and everyone else")
    src = B.sources("Central Elections Committee via Wikipedia", "dealignment after Shamir, Ventura, Arian and Kedar "
                    "(2008). " + PARTY_DEF)
    page = B.BlogPage(title, sub, src)
    dates = list(v.index)
    x = np.arange(len(dates))
    left_pad = 0.46
    width = page.inner - left_pad
    panel_gap, h1, h2 = 0.55, 2.45, 1.45

    y1 = page.top + 0.30
    page.text(page.M, y1 - 0.12, "Share of the vote that moved", size=B.T_PANEL, weight=600, serif=True)
    ax = page.axes(page.M + left_pad, y1, width, h1)
    ax.set_ylim(0, 50)
    B.hgrid(ax, [0, 10, 20, 30, 40], fmt=lambda t: f"{t:.0f}%", baseline=0)
    ax.set_xlim(-0.5, len(dates) - 0.5)
    ax.set_xticks([])
    ax.plot(x, v["family"], color=B.SLATE, lw=B.LW_DATA, ls="-", marker="o", ms=6.5, markeredgewidth=0, zorder=4,
            clip_on=False, solid_joinstyle="round")
    ax.plot(x, v["camps"], color=B.CLAY, lw=B.LW_DATA, ls=(0, (3.4, 1.7)), marker="s", ms=6.0, markeredgewidth=0,
            zorder=4, clip_on=False)
    k06 = dates.index("2006-03-28")
    _label(ax, k06, v["family"].iloc[k06] + 0.20 * (50 / h1), f"{v['family'].iloc[k06]:.0f}%", colour=B.SLATE,
           size=B.T_VALUE, weight=600, ha="center", va="bottom")
    _label(ax, x[-1] + 0.1, 28.0, "Between\nparties", colour=B.SLATE, ha="right", va="top", linespacing=1.05)
    _label(ax, 6.3, 5.4, "Between the\ntwo camps", colour=B.CLAY, ha="center", va="bottom", linespacing=1.05)

    y2 = y1 + h1 + panel_gap
    page.text(page.M, y2 - 0.12, "Seats won by lists led by a new party", size=B.T_PANEL, weight=600, serif=True)
    bx = page.axes(page.M + left_pad, y2, width, h2)
    bx.set_ylim(0, 52)
    B.hgrid(bx, [0, 20, 40], fmt=lambda t: f"{t:.0f}", baseline=0)
    bx.set_xlim(-0.5, len(dates) - 0.5)
    ticks = [i for i, d in enumerate(dates) if d in ELECTION_TICK]
    bx.set_xticks(ticks)
    bx.set_xticklabels([ELECTION_TICK[dates[i]] for i in ticks], fontsize=B.T_TICK, color=B.CAP, family=B.SANS)
    bx.bar(x, v["new_seats"], width=0.56, color="#d9cfc3", lw=0, zorder=3)
    for d, lab in NEWCOMERS.items():
        i = dates.index(d)
        _label(bx, i, v["new_seats"].iloc[i] + 0.06 * (52 / h2), lab, colour=B.INK, ha="center", va="bottom",
               linespacing=1.0)
    for a in (ax, bx):
        a.tick_params(axis="y", pad=7)
        a.tick_params(axis="x", pad=5)
    page.fit(y2 + h2 + 0.30)

    alt = (f"Line and bar chart of the thirteen Knesset elections from 1992 to 2022. At each election a fifth of the "
           f"vote moved between parties on average ({v['family'].mean():.0f}%, counting a party with its successors "
           f"and a joint list with its parts), and {v.loc['2006-03-28', 'family']:.0f}% in 2006, when Kadima and Gil, "
           f"both new, won {int(v.loc['2006-03-28', 'new_seats'])} seats. Between the two camps, Right and Orthodox "
           f"against everyone else, only {v['camps'].mean():.0f}% moved on average. New lists also won "
           f"{int(v.loc['2013-01-22', 'new_seats'])} seats in 2013 (Yesh Atid and Hatnuah) and "
           f"{int(v.loc['2019-04-09', 'new_seats'])} in April 2019 (Blue and White).")
    return page.save("dealignment_shamir", alt=alt)


CHARTS = {
    "issues_by_bloc": draw_issues_by_bloc,
    "anti_arab_attacks": draw_anti_arab_attacks,
    "dealignment_shamir": draw_dealignment_shamir,
}
