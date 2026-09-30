"""Loading and formatting for the blog chart tables."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from knesset_ches.questions import ROOT

TABLES = ROOT / "outputs" / "tables"
HANDCHECK = ROOT / "data" / "heckle_gendered_v2_handcheck.csv"
UNITS = ROOT / "data" / "corpus" / "processed" / "units.parquet"
NOTABLE_MKS = ROOT / "config" / "notable_mks.csv"
RIGHT_CAMP = ("Right", "Orthodox")
OTHER_CAMP = ("Left", "Secular Centre", "Arab-Israeli")

# crosswalk short names to the names printed
DISPLAY_NAMES = {
    "Labor": "Labour", "Labor-Meretz": "Labour-Meretz", "Center Party": "Centre Party",
    "Y. Beiteinu": "Yisrael Beiteinu", "YB": "Yisrael Beiteinu", "Y. BaAliyah": "Yisrael BaAliyah",
    "B&W": "Blue & White", "StateCamp": "State Camp", "RZ": "Religious Zionism", "NU-YB": "National Union-Beiteinu",
    "Agudat Yisr.": "Agudat Yisrael", "Degel HaTor.": "Degel HaTorah", "Otzma Yehud.": "Otzma Yehudit",
    "Dem. Choice": "Democratic Choice", "Secular Fac.": "Secular Faction", "UTJ": "United Torah Judaism",
    "Arab Natl P.": "Arab National Party", "Ind. O. Levy": "Orly Levy (independent)",
}


@dataclass
class Tables:
    group: pd.DataFrame              # party, lineage and bloc estimates
    mk: pd.DataFrame
    validation_points: pd.DataFrame
    validation_summary: pd.DataFrame
    path: Path


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"period": str, "actor_id": str}, low_memory=False)


def load_tables(tables_dir: Path | str) -> Tables:
    d = Path(tables_dir)
    return Tables(group=_read(d / "group_scores.csv"), mk=_read(d / "mk_scores.csv"),
                  validation_points=_read(d / "validation_points.csv"),
                  validation_summary=_read(d / "validation_summary.csv"), path=d)


def display_name(name: str, year: float | None = None) -> str:
    """Printed party name. The corpus uses one id for the National Union and Religious Zionism."""
    if name == "NU / RZ":
        return "Religious Zionism" if year is None or year >= 2021 else "National Union"
    return DISPLAY_NAMES.get(name, name)


def english_names() -> dict[str, str]:
    """English names of the members we label."""
    table = pd.read_csv(NOTABLE_MKS, dtype=str)
    return dict(zip(table["speaker_id"], table["name_en"]))


def round_count(n: float) -> str:
    """Round a count for print, such as 194,000 for 194,388."""
    n = float(n)
    if n >= 1e6:
        return f"{n / 1e6:.1f}m"
    if n >= 1e4:
        return f"{int(round(n, -3)):,}"
    if n >= 1e3:
        return f"{int(round(n, -2)):,}"
    return f"{int(round(n, -1))}" if n >= 100 else f"{int(n)}"


NUMBER_WORDS = "zero one two three four five six seven eight nine ten eleven twelve".split()


def number_word(k: int) -> str:
    return NUMBER_WORDS[int(k)]


def pct(v: float) -> str:
    return f"{v * 100:.0f}%"


SHARE_WORDS = [(0.04, "almost none"), (0.08, "one in fifteen"), (0.12, "one in ten"), (0.17, "one in seven"),
               (0.22, "one in five"), (0.29, "a quarter"), (0.37, "a third"), (0.45, "nearly half"), (0.55, "half"),
               (0.62, "well over half"), (0.7, "two-thirds"), (0.79, "three-quarters"), (0.9, "most"),
               (1.01, "nearly all")]


def share_words(x: float) -> str:
    """A share in words, such as 'a quarter' for 0.25."""
    return next(words for limit, words in SHARE_WORDS if x < limit)


def prep_chamber(tables: Tables, dimension: str, period_type: str, periods: list[str]) -> pd.DataFrame:
    """One row per member, party and period. `thin` means too little speech."""
    m = tables.mk
    df = m[(m["dimension"] == dimension) & (m["period_type"] == period_type) & m["period"].isin(periods)].copy()
    df["value"] = df["estimate"]
    df["thin"] = ~df["sufficient"].fillna(False).astype(bool)
    cols = ["period", "period_start", "speaker_id", "speaker_name", "faction_id", "faction_name", "bloc", "value",
            "n_speeches", "thin"]
    return df[cols].sort_values(["period_start", "speaker_id"]).reset_index(drop=True)


def chamber_medians(df: pd.DataFrame) -> pd.DataFrame:
    """Each camp's median member by period, and the gap."""
    rows = []
    for period, g in df.groupby("period", sort=False):
        a = g.loc[g["bloc"].isin(RIGHT_CAMP), "value"].median()
        b = g.loc[g["bloc"].isin(OTHER_CAMP), "value"].median()
        rows.append({"period": period, "median_a": a, "median_b": b, "gap": a - b})
    return pd.DataFrame(rows, columns=["period", "median_a", "median_b", "gap"])


def prep_validation(tables: Tables) -> pd.DataFrame:
    """Party points on scales with at least eight parties."""
    p = tables.validation_points
    p = p[p.groupby("dimension")["dimension"].transform("size") >= 8]
    return p[["dimension", "bloc", "jev", "ches"]].reset_index(drop=True)


def validation_r(points: pd.DataFrame) -> pd.DataFrame:
    """Pearson r between model and experts by scale, NaN under three parties."""
    rows = []
    for d, g in points.groupby("dimension", sort=False):
        r = np.nan
        if len(g) >= 3 and g["jev"].std() > 0 and g["ches"].std() > 0:
            r = float(np.corrcoef(g["jev"], g["ches"])[0, 1])
        rows.append({"dimension": d, "n": len(g), "r": r})
    return pd.DataFrame(rows, columns=["dimension", "n", "r"])
