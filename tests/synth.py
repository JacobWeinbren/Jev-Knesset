"""A synthetic Knesset with known positions, for the aggregate and validate tests.

    world = generate(out_dir, n_sittings=400, speeches_per_sitting=30, seed=7)

Its faction ids clash with real ones, so tests pass world.blocs_path rather than the real bloc file.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from knesset_ches.questions import build_questions, load_dimensions, load_settings

RESULTS_SCHEMA = """CREATE TABLE answers (
  unit_id TEXT NOT NULL, qid TEXT NOT NULL, prompt_hash TEXT NOT NULL, model TEXT NOT NULL, stage TEXT NOT NULL,
  score REAL, noul REAL, PRIMARY KEY (unit_id, qid, prompt_hash, model, stage))"""

KNESSET_TERMS = {
    13: ("1992-07-13", "1996-06-16"), 14: ("1996-06-17", "1999-06-06"), 15: ("1999-06-07", "2003-02-16"),
    16: ("2003-02-17", "2006-04-16"), 17: ("2006-04-17", "2009-02-23"), 18: ("2009-02-24", "2013-02-04"),
    19: ("2013-02-05", "2015-03-30"), 20: ("2015-03-31", "2019-04-29"), 21: ("2019-04-30", "2019-10-02"),
    22: ("2019-10-03", "2020-03-15"), 23: ("2020-03-16", "2021-04-05"), 24: ("2021-04-06", "2022-11-14"),
    25: ("2022-11-15", "2024-04-03"),
}

PROJECT_BLOCS = {"left": "Left", "arab": "Arab-Israeli", "center": "Secular Centre", "haredi": "Orthodox",
                 "right": "Right", "religious_right": "Orthodox"}
PROJECT_BLOC_OF_FACTION = {"111": "Sectoral", "115": "Sectoral"}
BLOC_AXIS = {"left": 2.0, "arab": 1.2, "center": 4.8, "haredi": 6.6, "right": 7.4, "religious_right": 8.6}
PACK_WORDS, PACK_SD, MK_SD, SITTING_SD = 900, 0.6, 0.7, 0.5
STALE_HASH, OTHER_MODEL = "0000000000000000", "typesafe/jev-0.0"

# id, Hebrew and English names, short label, lineage, bloc, first and last Knesset, seats
FACTIONS = [
    ("101", "מפלגת העמל", "Workers Party", "Workers", "workers", "left", 13, 25, 22),
    ("102", "שמאל חדש", "New Left", "NewLeft", "new_left", "left", 13, 24, 8),
    ("103", "הרשימה הערבית", "Arab List", "ArabList", "arab_list", "arab", 13, 25, 7),
    ("104", "התנועה האסלאמית", "Islamic Movement", "Islamic", "islamic", "arab", 15, 25, 4),
    ("105", "מפלגת המרכז", "Centre Party", "Centre", "centre", "center", 15, 20, 14),
    ("106", "יש תקווה", "There Is Hope", "Hope", "centre", "center", 21, 25, 18),
    ("107", "כולנו יחד", "All Together", "Together", "together", "center", 19, 25, 9),
    ("108", "שומרי תורה", "Torah Guardians", "Guardians", "guardians", "haredi", 13, 25, 10),
    ("109", "יהדות מאוחדת", "United Judaism", "UJ", "united_judaism", "haredi", 13, 25, 6),
    ("110", "האיחוד הלאומי", "National Union", "NatUnion", "national", "right", 13, 25, 28),
    ("111", "ביתנו", "Our Home", "OurHome", "our_home", "right", 15, 25, 8),
    ("112", "המפלגה הדתית", "Religious Party", "Religious", "religious_zionist", "religious_right", 13, 18, 6),
    ("113", "הבית הדתי", "Religious Home", "RelHome", "religious_zionist", "religious_right", 19, 25, 9),
    ("114", "עוצמה", "Strength", "Strength", "strength", "religious_right", 20, 25, 5),
    ("115", "קול יחיד", "Single Voice", "Single", "single_voice", "center", 16, 18, 1),
]
COALITIONS = {
    13: {"101", "102", "108"}, 14: {"110", "108", "109", "112"}, 15: {"101", "102", "105", "108"},
    16: {"110", "105", "111", "112"}, 17: {"105", "101", "108", "115"}, 18: {"110", "111", "108", "109", "112"},
    19: {"110", "105", "113", "107"}, 20: {"110", "108", "109", "113", "107"}, 21: {"110", "108", "109", "113"},
    22: {"110", "108", "109", "113"}, 23: {"110", "106", "108", "109"}, 24: {"106", "107", "101", "102", "111", "104"},
    25: {"110", "108", "109", "113", "114"},
}


@dataclass
class SynthWorld:
    out_dir: Path
    units_path: Path
    db_path: Path
    factions_path: Path
    blocs_path: Path
    positions: pd.DataFrame
    talk_rates: pd.DataFrame
    n_current_answers: int
    n_stale_answers: int


def project_bloc(faction_id: pd.Series, bloc: pd.Series) -> pd.Series:
    return faction_id.map(PROJECT_BLOC_OF_FACTION).fillna(bloc.map(PROJECT_BLOCS))


def generate(out_dir: Path | str, n_sittings: int = 400, speeches_per_sitting: int = 30, seed: int = 7,
             dimensions: list[str] | None = None) -> SynthWorld:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    all_dimensions = load_dimensions()
    kept = [d for d in all_dimensions if dimensions is None or d.id in dimensions]
    questions = [q for q in build_questions(all_dimensions) if q.dimension in {d.id for d in kept}]
    factions = pd.DataFrame(FACTIONS, columns=["faction_id", "name_he", "name_en", "short_en", "lineage", "bloc",
                                               "first_knesset", "last_knesset", "seats"])
    positions, base, drift = _true_positions(factions, [d.id for d in all_dimensions if d.kind == "position"], rng)
    talk = _talk_rates(factions, [d.id for d in all_dimensions], rng)
    packs = _packs(_speeches(_sittings(n_sittings, rng), _members(factions, rng), speeches_per_sitting, rng), rng)
    units = _units(packs, factions, rng)

    units_path = out_dir / "packs.parquet"
    units.drop(columns=["mk_offset"]).to_parquet(units_path, index=False)
    factions_path = out_dir / "factions_en.csv"
    knessets = [f"{a}-{b}" for a, b in zip(factions["first_knesset"], factions["last_knesset"])]
    factions.assign(knessets=knessets, note="synthetic")[
        ["faction_id", "name_he", "name_en", "short_en", "lineage", "bloc", "knessets", "note"]
    ].to_csv(factions_path, index=False)
    blocs_path = out_dir / "faction_blocs.csv"
    pd.DataFrame({"faction_id": factions["faction_id"], "name_en": factions["name_en"], "knesset": "*",
                  "bloc": project_bloc(factions["faction_id"], factions["bloc"]), "basis": "synthetic"}
                 ).to_csv(blocs_path, index=False)
    db_path = out_dir / "results.mock.sqlite"
    db_path.unlink(missing_ok=True)
    n_current, n_stale = _write_answers(db_path, units, questions, kept, base, drift, talk,
                                        load_settings()["jev"]["model"], rng)
    return SynthWorld(out_dir, units_path, db_path, factions_path, blocs_path, positions, talk, n_current, n_stale)


def _true_positions(factions: pd.DataFrame, dimension_ids: list[str], rng: np.random.Generator):
    """base is the 2008 position and drift the change over 16 years."""
    axis = factions["bloc"].map(BLOC_AXIS).to_numpy()
    loading = rng.choice([1.0, 1.0, 0.8, -0.8], size=len(dimension_ids))   # some dimensions run the other way
    base = 5 + loading[None, :] * (axis[:, None] - 5) + rng.normal(0, 0.7, (len(factions), len(dimension_ids)))
    drift = rng.normal(0, 1.2, base.shape)
    base = np.clip(base, 1.6, 8.4)
    drift = np.clip(drift, 1.0 - base + 0.6, 9.0 - base - 0.6)
    rows = []
    for year in range(1992, 2025):
        now = base + drift * (year - 2008) / 16
        for j, dim in enumerate(dimension_ids):
            rows.append(pd.DataFrame({"faction_id": factions["faction_id"], "lineage": factions["lineage"],
                                      "bloc": project_bloc(factions["faction_id"], factions["bloc"]), "year": year,
                                      "dimension": dim, "position": now[:, j]}))
    base = pd.DataFrame(base, index=factions["faction_id"], columns=dimension_ids)
    drift = pd.DataFrame(drift, index=factions["faction_id"], columns=dimension_ids)
    return pd.concat(rows, ignore_index=True), base, drift


def _talk_rates(factions: pd.DataFrame, dimension_ids: list[str], rng: np.random.Generator) -> pd.DataFrame:
    general = rng.uniform(0.2, 0.5, len(dimension_ids))
    rate = np.clip(general[None, :] * rng.lognormal(0, 0.35, (len(factions), len(dimension_ids))), 0.03, 0.8)
    table = pd.DataFrame(rate, index=factions["faction_id"], columns=dimension_ids)
    return table.rename_axis(columns="dimension").stack().rename("rate").reset_index()


def _sittings(n_sittings: int, rng: np.random.Generator) -> pd.DataFrame:
    spans = {k: (pd.Timestamp(a), pd.Timestamp(b)) for k, (a, b) in KNESSET_TERMS.items()}
    total = sum((b - a).days for a, b in spans.values())
    rows = []
    for k, (start, end) in spans.items():
        days = (end - start).days + 1
        offsets = np.sort(rng.choice(days, size=min(max(6, round(n_sittings * days / total)), days), replace=False))
        for i, off in enumerate(offsets):
            date = start + pd.Timedelta(days=int(off))
            rows.append({"protocol_name": f"{k}_ptm_{i:05d}", "knesset": k, "date": date, "year": date.year})
    return pd.DataFrame(rows)


def _members(factions: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    rows, next_id = [], 1000
    for f in factions.itertuples():
        previous: list[tuple[str, float, float]] = []
        for k in range(f.first_knesset, f.last_knesset + 1):
            returning = [m for m in previous if rng.random() < 0.7][: f.seats]
            fresh = []
            for _ in range(f.seats - len(returning)):
                next_id += 1
                fresh.append((str(next_id), float(rng.normal(0, MK_SD)), float(rng.lognormal(0, 0.9))))
            previous = returning + fresh
            rows += [{"speaker_id": s, "faction_id": f.faction_id, "knesset": k, "offset": off, "talkativeness": t}
                     for s, off, t in previous]
    return pd.DataFrame(rows)


def _speeches(sittings: pd.DataFrame, mks: pd.DataFrame, per_sitting: int, rng: np.random.Generator) -> pd.DataFrame:
    parts = []
    for sitting in sittings.itertuples():
        present = mks[mks["knesset"] == sitting.knesset]
        p = (present["talkativeness"] / present["talkativeness"].sum()).to_numpy()
        chosen = present.iloc[rng.choice(len(present), size=int(rng.poisson(per_sitting)) + 2, p=p)]
        parts.append(pd.DataFrame({
            "knesset": sitting.knesset, "date": sitting.date, "year": sitting.year,
            "speaker_id": chosen["speaker_id"].to_numpy(), "faction_id": chosen["faction_id"].to_numpy(),
            "mk_offset": chosen["offset"].to_numpy(),
        }))
    speeches = pd.concat(parts, ignore_index=True)
    speeches["n_words"] = np.clip(rng.lognormal(5.9, 0.9, len(speeches)), 60, 9000).astype(int)
    return speeches[rng.random(len(speeches)) >= 0.04].reset_index(drop=True)     # the chair's turns are left out


def _packs(speeches: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    keys = ["speaker_id", "faction_id", "knesset", "year"]
    total = speeches.groupby(keys)["n_words"].transform("sum").to_numpy()
    n_packs = np.maximum(1, np.ceil(total / PACK_WORDS)).astype(int)
    speeches = speeches.assign(k=(rng.random(len(speeches)) * n_packs).astype(int))
    packs = speeches.groupby([*keys, "k"], as_index=False).agg(
        date=("date", "median"), mk_offset=("mk_offset", "first"), n_words=("n_words", "sum"))
    packs["date"] = packs["date"].dt.normalize()
    packs["unit_id"] = ("pack|plenary|" + packs["speaker_id"] + "|" + packs["faction_id"] + "|"
                        + packs["knesset"].astype(str) + "|" + packs["year"].astype(str) + "|"
                        + packs["k"].map("{:02d}".format))
    return packs.drop(columns="k")


def _units(packs: pd.DataFrame, factions: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Each pack stands as its own speech and protocol."""
    units = packs.assign(speech_id=packs["unit_id"], protocol_name=packs["unit_id"], protocol_type="plenary",
                         is_chairman=False, speaker_name="חבר כנסת " + packs["speaker_id"])
    units["faction_name"] = units["faction_id"].map(factions.set_index("faction_id")["name_he"])
    in_government = [f in COALITIONS[k] for f, k in zip(units["faction_id"], units["knesset"])]
    units["coalition"] = np.where(in_government, "coalition", "opposition")
    units.loc[(units["knesset"] == 13) & (rng.random(len(units)) < 0.02), "coalition"] = None   # the corpus has gaps
    units["date"] = units["date"].dt.date
    return units


def _write_answers(db_path: Path, units: pd.DataFrame, questions, dimensions, base: pd.DataFrame,
                   drift: pd.DataFrame, talk: pd.DataFrame, model: str, rng: np.random.Generator) -> tuple[int, int]:
    """Scores are in level units, not 0 to 10."""
    by_dimension = {d.id: d for d in dimensions}
    talk_wide = talk.pivot(index="faction_id", columns="dimension", values="rate")
    faction, year = units["faction_id"].to_numpy(), units["year"].to_numpy()
    n = len(units)
    conn = sqlite3.connect(db_path)
    conn.execute(RESULTS_SCHEMA)
    insert = "INSERT INTO answers (unit_id, qid, prompt_hash, model, stage, score, noul) VALUES (?,?,?,?,?,?,?)"
    n_current = n_stale = 0
    on_topic: dict[str, np.ndarray] = {}
    for q in questions:
        dim = by_dimension[q.dimension]
        asked = np.ones(n, dtype=bool) if not dim.valid_from else (
            pd.to_datetime(units["date"]) >= pd.Timestamp(dim.valid_from)).to_numpy()
        topic = dim.gate_share_of or q.dimension
        rate = talk_wide[topic].reindex(faction).to_numpy()
        if topic not in on_topic:      # a pack is about five speeches, on topic if any of them is
            on_topic[topic] = rng.random(n) < np.clip((1 - (1 - rate) ** 5) * rng.lognormal(0, 0.5, n), 0, 0.95)
        relevant = on_topic[topic]
        ids = units["unit_id"].to_numpy()[asked]
        if q.role == "gate":
            noul = np.where(relevant, rng.beta(9, 1.3, n), rng.beta(1.2, 14, n))
            noul = np.where(rng.random(n) < 0.05, rng.uniform(0.3, 0.7, n), noul)[asked]
            rows = [(u, q.qid, q.prompt_hash, model, "unit", None, float(x)) for u, x in zip(ids, noul)]
        else:
            top = q.n_levels - 1
            if q.role == "score":
                truth = (base.loc[faction, q.dimension].to_numpy()
                         + drift.loc[faction, q.dimension].to_numpy() * (year - 2008) / 16)
                latent = truth + units["mk_offset"].to_numpy() + rng.normal(0, np.hypot(PACK_SD, SITTING_SD), n)
                score = np.where(relevant, latent / 10 * top, rng.normal(top / 2, 0.6, n))
            else:       # salience, around the faction's talk rate
                score = rate * 4 + rng.normal(0, 0.45, n)
            score = np.clip(score, 0, top)[asked]
            rows = [(u, q.qid, q.prompt_hash, model, "unit", float(s), None) for u, s in zip(ids, score)]
        conn.executemany(insert, rows)
        n_current += len(rows)
        # stale rows that aggregate must ignore
        for stale in ([(r[0], r[1], STALE_HASH, *r[3:]) for r in rows[:50]],
                      [(*r[:3], OTHER_MODEL, *r[4:]) for r in rows[:50]]):
            conn.executemany(insert, stale)
            n_stale += len(stale)
        conn.executemany(insert, [(*r[:4], "speech", *r[5:]) for r in rows[:50]])
    conn.commit()
    conn.close()
    return n_current, n_stale
