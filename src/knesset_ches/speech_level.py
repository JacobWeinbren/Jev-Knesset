"""Summarise Jev's answers about single speeches into the tables the blog reads. Run after aggregate.

    python -m knesset_ches.speech_level [--out DIR]
"""
from __future__ import annotations

import argparse
import functools
import json
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from knesset_ches import aggregate as agg
from knesset_ches.questions import ROOT, build_questions, load_dimensions, load_settings
from knesset_ches.tables import display_name, english_names

SPEECH_YAML = ROOT / "config" / "dimensions_speech.yaml"
UNITS = {"plenary": ROOT / "data" / "corpus" / "processed" / "units.parquet",
         "committee": ROOT / "data" / "corpus" / "processed_committee" / "units.parquet"}
NETANYAHU_UNITS = ROOT / "data" / "corpus" / "processed" / "units_bibi.parquet"
TABLES = agg.TABLES_DIR
META = ["unit_id", "speaker_id", "faction_id", "faction_name", "knesset", "date", "year", "coalition", "is_chairman",
        "n_words", "n_sentences"]
# five-level items with a neutral middle also get the shares at levels 0-1 and 3-4
SPLIT_SCORES = {"majority_rule": ("checks", "majority")}
ATTACK_GROUPS = {"Right": "Right and Orthodox", "Orthodox": "Right and Orthodox",
                 "Left": "Everyone else", "Secular Centre": "Everyone else", "Arab-Israeli": "Everyone else"}
NETANYAHU_FLAGS = ("attack", "defend", "must_go")
NETANYAHU_CAMP = {"Likud", "Shas", "United Torah Judaism", "Religious Zionism", "Otzma Yehudit", "Noam", "Jewish Power"}


def query(sql: str, params: Sequence[Any] = ()) -> pd.DataFrame:
    with closing(sqlite3.connect(f"{agg.DB_PATH.as_uri()}?mode=ro", uri=True)) as con:
        return pd.read_sql(sql, con, params=params)


def flags(stage: str) -> pd.DataFrame:
    """Yes-probabilities of a stage's yes/no questions, one column per question."""
    a = query("select unit_id, qid, noul from answers where stage = ?", (stage,))
    wide = a.pivot_table(index="unit_id", columns="qid", values="noul")
    wide.columns = [c.replace("__flag", "") for c in wide.columns]
    return wide


def periods(knesset: pd.Series, settings: dict[str, Any]) -> pd.Series:
    """Knesset numbers as period labels, with the 2019-21 Knessets merged."""
    return knesset.astype(int).map(agg.knesset_labels(knesset.unique(), settings))


def write_tables(tables: dict[str, pd.DataFrame], out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, table in tables.items():
        paths.append(out_dir / f"{name}.csv")
        table.to_csv(paths[-1], index=False)
    return paths


def items() -> list[tuple[str, str, int | None]]:
    """(item, qid, levels) for each speech-level question. Yes/no questions have no levels."""
    return [(q.dimension, q.qid, q.n_levels) for q in build_questions(load_dimensions(SPEECH_YAML))]


def _item_answers(item: str, qid: str, n_levels: int | None) -> pd.DataFrame:
    value = "noul" if n_levels is None else "score"
    extra = ", probabilities" if item in SPLIT_SCORES else ""
    a = query(f"select unit_id, {value} as {item}{extra} from answers where stage = 'speech' and qid = ?", (qid,))
    if n_levels:
        a[item] = a[item] * 10 / (n_levels - 1)
    if extra:
        probs = a.pop("probabilities").map(json.loads)
        low, high = SPLIT_SCORES[item]
        a[f"{item}_{low}"] = probs.map(lambda d: sum(float(d.get(str(k), 0)) for k in (0, 1)))
        a[f"{item}_{high}"] = probs.map(lambda d: sum(float(d.get(str(k), 0)) for k in (3, 4)))
    return a


def answers() -> pd.DataFrame:
    parts = [_item_answers(*question) for question in items()]
    return functools.reduce(lambda a, b: a.merge(b, on="unit_id", how="outer"), parts)


def speech_answers(wide: pd.DataFrame, protocol_type: str, settings: dict[str, Any]) -> pd.DataFrame:
    """Answers joined to speech metadata, bloc and period, without chair turns."""
    df = wide.merge(pd.read_parquet(UNITS[protocol_type], columns=META), on="unit_id", how="inner")
    if settings["filters"]["exclude_chair_units"]:
        df = df[~df["is_chairman"].fillna(False).astype(bool)]
    df["bloc"] = agg.project_blocs(df, pd.Series("Other", index=df.index))
    df["protocol_type"] = protocol_type
    df["period"] = periods(df["knesset"], settings)
    return df.reset_index(drop=True)


def _weighted(df: pd.DataFrame, cols: Sequence[str], by: list[str]) -> pd.DataFrame:
    w = np.sqrt(df["n_words"].clip(lower=1).astype(float))
    out = []
    for keys, g in df.groupby(by, sort=True, observed=True):
        gw = w.loc[g.index]
        row = dict(zip(by, keys))
        for c in cols:
            ok = g[c].notna()
            row[c] = float(np.average(g.loc[ok, c], weights=gw[ok])) if ok.any() else np.nan
        row["n_speeches"], row["n_words"] = len(g), int(g["n_words"].sum())
        out.append(row)
    return pd.DataFrame(out)


def summarise(df: pd.DataFrame, period_col: str) -> pd.DataFrame:
    """Weighted means by protocol type, period and actor, in long form."""
    cols = [i for i, _, _ in items()] + [f"{i}_{side}" for i, sides in SPLIT_SCORES.items() for side in sides]
    frames = []
    for actor_type, key in (("knesset_wide", None), ("bloc", "bloc"), ("coalition_status", "coalition")):
        part = df if key is None else df[df[key].notna() & (df[key].astype(str) != "Other")]
        by = ["protocol_type", period_col] + ([key] if key else [])
        wide = _weighted(part, cols, by)
        wide["actor_type"] = actor_type
        wide["actor_id"] = wide[key] if key else "all"
        long = wide.melt(id_vars=["protocol_type", period_col, "actor_type", "actor_id", "n_speeches", "n_words"],
                         value_vars=cols, var_name="item", value_name="value")
        frames.append(long.rename(columns={period_col: "period"}))
    out = pd.concat(frames, ignore_index=True)
    return out[["protocol_type", "period", "actor_type", "actor_id", "item", "value", "n_speeches", "n_words"]]


def same_members(df: pd.DataFrame) -> pd.DataFrame:
    """Tone in 2015-20 and from 2021 among members who spoke in both, to separate a change of cast from one of tone."""
    d = df[df["protocol_type"] == "plenary"].copy()
    d["w"] = np.sqrt(d["n_words"].clip(lower=1).astype(float))
    d["gov"] = (d["coalition"].astype(str) == "coalition").astype(float)
    rows = []
    for item in ("hostility", "emotional_intensity", "delegitimisation"):
        x = d[d[item].notna()]
        my = x.groupby(["speaker_id", "year"]).apply(
            lambda g: pd.Series({"v": np.average(g[item], weights=g["w"]), "gov": g["gov"].mean(), "n": len(g)}),
            include_groups=False).reset_index()
        my = my[my["n"] >= 10]
        before, after = my[my["year"].between(2015, 2020)], my[my["year"] >= 2021]
        pre = before.groupby("speaker_id")[["v", "gov"]].mean()
        post = after.groupby("speaker_id")[["v", "gov"]].mean()
        both = pre.index.intersection(post.index)
        same_side = both[(pre.loc[both, "gov"].round() == post.loc[both, "gov"].round()).to_numpy()]
        rows.append({"item": item, "n_members": len(both),
                     "before": float(pre.loc[both, "v"].mean()), "after": float(post.loc[both, "v"].mean()),
                     "n_same_side": len(same_side), "before_same_side": float(pre.loc[same_side, "v"].mean()),
                     "after_same_side": float(post.loc[same_side, "v"].mean()),
                     "chamber_before": float(before["v"].mean()), "chamber_after": float(after["v"].mean())})
    return pd.DataFrame(rows)


def attacks_on_arab_citizens(df: pd.DataFrame) -> pd.DataFrame:
    """Share of plenary speeches attacking Arab citizens as a group, by quarter and group of blocs."""
    d = df[(df["protocol_type"] == "plenary") & df["attacks_arab_citizens"].notna()].copy()
    d["flag"] = d["attacks_arab_citizens"] >= 0.5
    d["quarter"] = pd.to_datetime(d["date"]).dt.to_period("Q").astype(str)
    d["group"] = d["bloc"].map(ATTACK_GROUPS)
    d = pd.concat([d.dropna(subset=["group"]), d.assign(group="All members")])
    return d.groupby(["quarter", "group"])["flag"].agg(n_speeches="size", n_attacks="sum", share="mean").reset_index()


def netanyahu_speeches(settings: dict[str, Any]) -> pd.DataFrame:
    """Plenary speeches naming Netanyahu, not his own, with party and bloc from group_scores.csv."""
    d = pd.read_parquet(NETANYAHU_UNITS, columns=["unit_id", "speaker_id", "faction_id", "knesset", "year", "n_words"])
    d = d.merge(flags("bibi").reset_index(), on="unit_id")
    d["period"] = periods(d["knesset"], settings)
    g = pd.read_csv(TABLES / "group_scores.csv", dtype=str,
                    usecols=["actor_type", "period_type", "actor_id", "period", "actor_name", "bloc"])
    parties = g[(g["actor_type"] == "party") & (g["period_type"] == "knesset")].drop_duplicates(["actor_id", "period"])
    d = d.merge(parties[["actor_id", "period", "actor_name", "bloc"]], how="left",
                left_on=["faction_id", "period"], right_on=["actor_id", "period"])
    d["party"] = [display_name(n, y) for n, y in zip(d["actor_name"].fillna(""), d["year"])]
    d["bloc"] = d["bloc"].fillna("")
    netanyahu = {sid for sid, name in english_names().items() if "Netanyahu" in name}
    return d[~d["speaker_id"].astype(str).isin(netanyahu)]


def _netanyahu_shares(d: pd.DataFrame, by: str) -> pd.DataFrame:
    rows = []
    for key, x in d.groupby(by):
        row: dict[str, Any] = {by: key}
        for flag in NETANYAHU_FLAGS:
            row[f"share_{flag}"] = np.average(x[f"bibi_{flag}"] >= 0.5, weights=x["w"])
        row["n_speeches"] = len(x)
        rows.append(row)
    return pd.DataFrame(rows)


def netanyahu_tables(d: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = d.assign(w=np.sqrt(d["n_words"].clip(lower=1).astype(float)))
    late = d[d["knesset"] >= 21]
    by_party = _netanyahu_shares(late, "party")
    by_party["bloc"] = by_party["party"].map(late.groupby("party")["bloc"].agg(lambda b: b.mode().iloc[0]))
    by_party = by_party[(by_party["party"] != "") & (by_party["n_speeches"] >= 40)].copy()
    by_party["in_camp"] = by_party["party"].isin(NETANYAHU_CAMP)
    by_party["period"] = "2019–2026*"
    return by_party.sort_values("share_attack", ascending=False), _netanyahu_shares(d, "year")


def build_tables(out_dir: Path) -> list[Path]:
    settings = load_settings()
    wide = answers()
    df = pd.concat([speech_answers(wide, p, settings) for p in UNITS], ignore_index=True)
    coverage = df.groupby(["protocol_type", "year"]).agg(n_speeches=("unit_id", "size"),
                                                         n_sentences=("n_sentences", "sum")).reset_index()
    by_party, by_year = netanyahu_tables(netanyahu_speeches(settings))
    return write_tables({"speech_by_year": summarise(df, "year"), "speech_by_knesset": summarise(df, "period"),
                         "speech_coverage": coverage, "same_members": same_members(df),
                         "attacks_arab_by_quarter": attacks_on_arab_citizens(df),
                         "netanyahu_by_party": by_party, "netanyahu_by_year": by_year}, out_dir)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m knesset_ches.speech_level", description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, default=TABLES)
    args = ap.parse_args(argv)
    for path in build_tables(args.out):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
