"""Left, centre and right self-placement of Jewish Israelis in INES, 1988 to 2025, for voter_ideology.csv.

    .venv/bin/python scripts/ines_left_right.py <folder> > ines_rows.csv

The folder holds the Stata files in STUDIES, from https://socsci4.tau.ac.il/mu2/ines/data/our-data/.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


def equals(var: str, value: int = 1):
    return lambda d: d[var] == value


def not_arab_2006(d: pd.DataFrame) -> pd.Series:
    """2006 has no sector variable: Arabs are those interviewed in Arabic or Muslim, Christian or Druze."""
    return ~((d["c93"] == 2) | d["c87"].isin([2, 3, 4]))


# label, file, Jewish filter (None if all are Jews), 7-point item, tendency item, weight, fieldwork.
# Studies before 2019 have no weight for Jews and are left unweighted, as the INES appendices advise.
STUDIES = [
    ("1988", "1988_N=873.dta", None, "f125", "g72", None, "Oct 1988 wave 2; Jewish sample"),
    ("1992", "1992.dta", None, "a110", "i56", None, "pre-election, spring 1992; Jewish sample"),
    ("1996", "1996j.dta", None, "c69", "ccc46", None, "pre-election, spring 1996; Jewish file"),
    ("1999", "1999-1.dta", lambda d: d["c40"] < 80, "i19", "c19", None,
     "pre-election, Apr-May 1999; Jews = locality code < 80"),
    ("2001", "2001.dta", lambda d: d["b28"].isna(), "a66", None, None, "pre-election (PM race), Jan-Feb 2001"),
    ("2003", "2003.dta", equals("jew"), "a49", "b73", None, "pre-election, Jan 2003"),
    ("2006", "2006.dta", not_arab_2006, "b62", "b61", None, "pre-election, Feb-Mar 2006; items asked of a subsample"),
    ("2009", "2009.dta", equals("v195"), "v135", "v134", None, "pre-election, Jan-Feb 2009; split sample"),
    ("2013", "2013.dta", equals("v153"), "v88", "v103", None, "pre-election, Dec 2012-Jan 2013"),
    ("2015", "2015.dta", equals("v148"), "v103", "v71", None, "pre-election, Feb-Mar 2015"),
    ("2019a", "Apr-Sep_2019_update_STATA.dta", equals("v149"), "v111", "v53", "weights_panel_1",
     "pre-election, Feb-Apr 2019"),
    ("2019b", "Apr-Sep_2019_update_STATA.dta", equals("c149"), "c111", None, "weights_panel_3",
     "pre-election, Sep 2019 (panel)"),
    ("2020", "March_2020_data.dta", equals("sector"), "v111", None, "weights_panel_1", "pre-election, Feb 2020"),
    ("2021", "March_2021_data_website_STATA.dta", equals("sector"), "v111", None, "weights_panel_1",
     "pre-election, Feb-Mar 2021; phone + internet"),
    ("2022", "2022_STATA.dta", equals("sector"), "v111", "v53", "w_jews_panel1", "pre-election, Oct 2022"),
    ("2025", "2025_STATA.dta", equals("sector"), "v111", None, "w_jews", "March 2025 (no election)"),
]
# scale7 runs from 1 right to 7 left, cut as the Israel Democracy Institute does
ITEMS = {
    "scale7": {"right": [1, 2, 3], "centre": [4], "left": [5, 6, 7]},
    "tendency": {"left": [1, 2], "centre": [3], "right": [4, 5]},
}
# Until 2003 everyone got both items, so a blank is a don't know. Later a blank can mean not asked.
EVERYONE_ASKED = {"1988", "1992", "1996", "1999", "2001", "2003"}


def shares(x: pd.Series, w: pd.Series, groups: dict[str, list[int]]) -> dict[str, float]:
    out = {k: float(100 * w[x.isin(codes)].sum() / w.sum()) for k, codes in groups.items()}
    out["dont_know"] = 100 - sum(out.values())
    out["n"] = int(len(x))
    return out


def main(folder: str) -> None:
    rows = []
    for label, file, jewish, scale7, tendency, weight, note in STUDIES:
        df = pd.read_stata(Path(folder) / file, convert_categoricals=False)
        j = df[jewish(df)] if jewish else df
        w = j[weight].astype(float).fillna(0) if weight else pd.Series(1.0, index=j.index)
        for item, var in (("scale7", scale7), ("tendency", tendency)):
            if var is None:
                continue
            x = j[var]
            asked = pd.Series(True, index=x.index) if label in EVERYONE_ASKED else x.notna()
            s = shares(x[asked], w[asked], ITEMS[item])
            rows.append({"study": label, "item": item, "var": var, "weight": weight or "none", "note": note,
                         **{k: round(v, 1) if isinstance(v, float) else v for k, v in s.items()}})
    columns = ["study", "item", "var", "weight", "n", "left", "centre", "right", "dont_know", "note"]
    pd.DataFrame(rows)[columns].to_csv(sys.stdout, index=False)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
