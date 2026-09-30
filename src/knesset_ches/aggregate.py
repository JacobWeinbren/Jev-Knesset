"""Average Jev's answers from packs to members, parties, party families and blocs.

    python -m knesset_ches.aggregate [--out DIR]
"""
from __future__ import annotations

import argparse
import sqlite3
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy import sparse

from knesset_ches.questions import ROOT, Dimension, Question, build_questions, load_dimensions, load_settings

UNITS_PATH = ROOT / "data" / "corpus" / "processed" / "packs.parquet"
DB_PATH = ROOT / "data" / "jev" / "results.sqlite"
TABLES_DIR = ROOT / "outputs" / "tables"
FACTIONS_PATH = ROOT / "config" / "factions_en.csv"
BLOCS_PATH = ROOT / "config" / "faction_blocs.csv"

UNIT_COLUMNS = [
    "unit_id", "speech_id", "protocol_name", "protocol_type", "knesset", "date", "year", "speaker_id", "speaker_name",
    "faction_id", "faction_name", "coalition", "is_chairman", "n_words",
]
CATEGORY_COLUMNS = ["speech_id", "protocol_name", "speaker_id", "faction_id", "lineage", "bloc", "coalition"]
SPEECH_COLUMNS = ["speech_id", "protocol_name", "date", "year", "knesset", "speaker_id", "faction_id", "lineage",
                  "bloc", "coalition", "score10", "weight"]
ACTOR_TYPES = {"faction_id": "party", "lineage": "lineage", "bloc": "bloc"}
ACTOR_COLUMNS = ["actor_type", "actor_id", "actor_name", "lineage", "bloc"]
TABLE_COLUMNS = {
    "mk_scores": ["dimension", "period_type", "period", "period_start", "period_end", "speaker_id", "speaker_name",
                  "faction_id", "faction_name", "lineage", "bloc", "estimate", "n_speeches", "sufficient"],
    "group_scores": ["dimension", "period_type", "period", "actor_type", "actor_id", "actor_name", "bloc", "estimator",
                     "estimate", "n_mks", "sufficient", "coalition_share"],
    "salience": ["dimension", "period_type", "period", "actor_type", "actor_id", "actor_name", "bloc", "salience",
                 "n_mks", "sufficient"],
}

COMMITTEE_MIN_PAIRS = 30
COMMITTEE_MIN_PACKS = 3
BOOT_CHUNK_ELEMENTS = 4_000_000  # cells x replicates in memory at once
ANSWER_CHUNK_ROWS = 400_000


def seeded_rng(seed: int, *labels: str) -> np.random.Generator:
    """Same draws whatever order the work runs in."""
    return np.random.default_rng([int(seed), *(zlib.crc32(label.encode("utf-8")) for label in labels)])


def project_blocs(units: pd.DataFrame, fallback: pd.Series, path: Path | str = BLOCS_PATH) -> pd.Series:
    """Each row's bloc from config/faction_blocs.csv, where knesset "*" means any other Knesset."""
    table = pd.read_csv(path, dtype=str)
    exact = {(r.faction_id, r.knesset): r.bloc for r in table.itertuples() if r.knesset != "*"}
    default = {r.faction_id: r.bloc for r in table.itertuples() if r.knesset == "*"}
    keys = zip(units["faction_id"].astype(str), units["knesset"].astype(int).astype(str))
    mapped = pd.Series([exact.get(key, default.get(key[0])) for key in keys], index=units.index)
    return mapped.where(mapped.notna(), fallback)


def load_factions(path: Path | str = FACTIONS_PATH) -> pd.DataFrame:
    columns = ["faction_id", "name_en", "short_en", "lineage", "bloc"]
    return pd.read_csv(path, dtype=str, keep_default_na=False)[columns].drop_duplicates("faction_id")


def load_units(path: Path | str, factions: pd.DataFrame, blocs_path: Path | str = BLOCS_PATH) -> pd.DataFrame:
    units = pd.read_parquet(path, columns=UNIT_COLUMNS)
    units["date"] = pd.to_datetime(units["date"])
    units["faction_id"] = units["faction_id"].astype(str)
    units["speaker_id"] = units["speaker_id"].astype(str)
    units["is_chairman"] = units["is_chairman"].fillna(False).astype(bool)
    lookup = factions.set_index("faction_id")
    for col, fallback in (("lineage", units["faction_id"]), ("bloc", "other")):
        mapped = units["faction_id"].map(lookup[col])
        units[col] = mapped.where(mapped.notna() & (mapped != ""), fallback)
    units["bloc"] = project_blocs(units, units["bloc"], blocs_path)
    label = units["faction_id"].map(lookup["short_en"].where(lookup["short_en"] != "", lookup["name_en"]))
    units["faction_label"] = label.where(label.notna() & (label != ""), units["faction_name"])
    for col in CATEGORY_COLUMNS:
        units[col] = units[col].astype("category")
    return units.reset_index(drop=True)


def load_all_units(settings: dict[str, Any], units_path: Path | str, factions: pd.DataFrame,
                   blocs_path: Path | str) -> pd.DataFrame:
    units = load_units(units_path, factions, blocs_path)
    if settings["aggregation"]["committee"] != "calibrated":
        return units
    committee = load_units(ROOT / settings["corpus"]["committee_packs"], factions, blocs_path)
    # A few committee protocols are in both files. The answers go with the plenary copy.
    units = pd.concat([units, committee], ignore_index=True).drop_duplicates("unit_id", keep="first")
    units = units.reset_index(drop=True)
    for col in CATEGORY_COLUMNS:
        units[col] = units[col].astype(str).astype("category")
    return units


def committee_calibration(unit_df: pd.DataFrame, gate_threshold: float) -> dict[str, float] | None:
    """committee = a + b * plenary, from members scored in both settings in a Knesset."""
    ok = unit_df[(unit_df["gate"] >= gate_threshold) & ~unit_df["is_chairman"].to_numpy(dtype=bool)]
    keys = ["speaker_id", "knesset", "protocol_type"]
    means = ok.groupby(keys, observed=True)["score10"].agg(["mean", "size"]).reset_index()
    means = means[means["size"] >= COMMITTEE_MIN_PACKS]
    wide = means.pivot_table(index=["speaker_id", "knesset"], columns="protocol_type", values="mean", observed=True)
    wide = wide.dropna()
    if len(wide) < COMMITTEE_MIN_PAIRS or "plenary" not in wide or "committee" not in wide:
        return None
    # match SDs rather than regress, as noise in the plenary means would flatten the slope
    b = float(wide["committee"].std(ddof=1)) / float(wide["plenary"].std(ddof=1))
    a = float(wide["committee"].mean()) - b * float(wide["plenary"].mean())
    return {"a": a, "b": b, "n_pairs": int(len(wide))}


def apply_calibration(unit_df: pd.DataFrame, calib: dict[str, float] | None) -> pd.DataFrame:
    out = unit_df.copy()
    out["calib_b"] = 1.0
    is_committee = (out["protocol_type"] == "committee").to_numpy()
    if calib is None:
        return out[~is_committee].reset_index(drop=True)
    a, b = calib["a"], calib["b"]
    out.loc[is_committee, "score10"] = ((out.loc[is_committee, "score10"] - a) / b).clip(0, 10)
    out.loc[is_committee, "calib_b"] = b   # chunk_rule weights by b^2, the change in precision
    return out


Answers = dict[str, dict[str, np.ndarray]]     # by dimension, gate and score arrays in units order


def load_answers(db_path: Path | str, unit_ids: pd.Series, questions: list[Question],
                 model: str) -> tuple[Answers, int, int]:
    """This model's answers to the current prompts, with counts of answers used and stale."""
    current = {q.qid: q for q in questions}
    wanted = {qid: q.prompt_hash for qid, q in current.items()}
    index = pd.Index(unit_ids)
    fields: Answers = {}
    n_used = n_stale = 0
    sql = "SELECT unit_id, qid, prompt_hash, model, score, noul FROM answers WHERE stage = 'unit'"
    conn = sqlite3.connect(f"file:{Path(db_path)}?mode=ro", uri=True)
    try:
        for chunk in pd.read_sql_query(sql, conn, chunksize=ANSWER_CHUNK_ROWS):
            keep = ((chunk["prompt_hash"] == chunk["qid"].map(wanted)) & (chunk["model"] == model)).to_numpy()
            n_stale += int((~keep).sum())
            rows = index.get_indexer(chunk["unit_id"])
            keep = keep & (rows >= 0)
            chunk, rows = chunk[keep], rows[keep]
            n_used += len(chunk)
            for qid, part in chunk.groupby("qid", sort=False):
                q = current[qid]
                name, column = ("gate", "noul") if q.role == "gate" else ("score", "score")
                target = fields.setdefault(q.dimension, {})
                target.setdefault(name, np.full(len(index), np.nan, dtype=np.float32))
                target[name][rows[chunk.index.get_indexer(part.index)]] = part[column].to_numpy(dtype=np.float32)
    finally:
        conn.close()
    return fields, n_used, n_stale


def unit_scores(units: pd.DataFrame, answers: Answers, dimension: Dimension, questions: list[Question]) -> pd.DataFrame:
    n_levels = next(q.n_levels for q in questions if q.dimension == dimension.id and q.role != "gate")
    got = answers.get(dimension.id, {})
    nan = np.full(len(units), np.nan, dtype=np.float32)
    keep = ~np.isnan(got.get("score", nan))
    if dimension.kind == "position":
        keep &= ~np.isnan(got.get("gate", nan))
    if dimension.valid_from:
        keep &= (units["date"] >= pd.Timestamp(dimension.valid_from)).to_numpy()
    out = units.loc[keep].copy()
    out["gate"] = got.get("gate", nan)[keep].astype(float)
    out["score10"] = 10.0 * got.get("score", nan)[keep].astype(float) / (n_levels - 1)
    return out


def chunk_rule(unit_df: pd.DataFrame, kind: str, settings: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Per unit, whether it is usable and a multiplier on its words."""
    filters = settings["filters"]
    usable = unit_df["score10"].notna().to_numpy().copy()
    if filters["exclude_chair_units"]:
        usable &= ~unit_df["is_chairman"].to_numpy(dtype=bool)
    multiplier = np.ones(len(unit_df))
    if kind == "position":
        usable &= unit_df["gate"].to_numpy(dtype=float) >= filters["gate_threshold"]
        if "calib_b" in unit_df:
            multiplier = np.square(unit_df["calib_b"].to_numpy(dtype=float))
    return usable, multiplier


def build_speeches(unit_df: pd.DataFrame, kind: str, settings: dict[str, Any]) -> pd.DataFrame:
    """Usable speeches on one dimension. A pack counts as its own speech."""
    usable, multiplier = chunk_rule(unit_df, kind, settings)
    codes, _ = pd.factorize(unit_df["speech_id"], sort=False)
    n_speeches = int(codes.max()) + 1 if len(codes) else 0
    usable_words = unit_df["n_words"].to_numpy(dtype=float) * usable
    effective = usable_words * multiplier

    def per_speech(values: np.ndarray) -> np.ndarray:
        return np.bincount(codes, weights=values, minlength=n_speeches)

    total_usable, total_effective = per_speech(usable_words), per_speech(effective)
    _, first_unit = np.unique(codes, return_index=True)
    out = unit_df.iloc[first_unit][[c for c in SPEECH_COLUMNS if c in unit_df.columns]].reset_index(drop=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        score = np.nan_to_num(unit_df["score10"].to_numpy(dtype=float))
        out["score10"] = per_speech(effective * score) / total_effective
        if kind == "position":
            out["weight"] = np.sqrt(total_usable) * total_effective / total_usable
        else:
            out["weight"] = total_effective
    return out[total_effective > 0].reset_index(drop=True)


def knesset_labels(knessets: Sequence[int], settings: dict[str, Any]) -> dict[int, str]:
    labels = {int(k): str(int(k)) for k in knessets}
    for group in settings["periods"]["merge_knessets"]:
        merged = "-".join(str(int(k)) for k in group)
        labels.update({int(k): merged for k in group})
    return labels


def add_periods(frame: pd.DataFrame, period_type: str, settings: dict[str, Any],
                year_range: tuple[int, int] | None = None) -> pd.DataFrame:
    """Add a period label. rolling2 puts each row in two windows, its year's and the next."""
    year = frame["year"].to_numpy(dtype=int)
    if period_type == "year":
        return frame.assign(period=year.astype(str))
    if period_type == "knesset":
        labels = knesset_labels(np.unique(frame["knesset"]), settings)
        return frame.assign(period=frame["knesset"].astype(int).map(labels).to_numpy())
    first, last = year_range or ((int(year.min()), int(year.max())) if len(year) else (0, 0))
    windows = [frame.assign(period=(year + shift).astype(str))[(year + shift > first) & (year + shift <= last)]
               for shift in (0, 1)]
    return pd.concat(windows, ignore_index=True)


def period_table(units: pd.DataFrame, period_type: str, settings: dict[str, Any]) -> pd.DataFrame:
    sittings = units[["protocol_name", "date", "year", "knesset"]].drop_duplicates("protocol_name")
    year_range = (int(units["year"].min()), int(units["year"].max()))
    rows = []
    for period, part in add_periods(sittings, period_type, settings, year_range).groupby("period", sort=True):
        if period_type == "knesset":
            start, end = part["date"].min().date(), part["date"].max().date()
        else:
            years = [int(y) for y in period.split("-")]
            start, end = f"{years[0] - (period_type == 'rolling2')}-01-01", f"{years[-1]}-12-31"
        rows.append({"period_type": period_type, "period": period, "period_start": str(start),
                     "period_end": str(end)})
    return pd.DataFrame(rows)


def _group_codes(frame: pd.DataFrame, cols: Sequence[str]) -> tuple[np.ndarray, pd.DataFrame]:
    """Sorted dense codes for the distinct rows of frame[cols], and those rows."""
    key = np.zeros(len(frame), dtype=np.int64)
    for col in cols:
        codes, _ = pd.factorize(frame[col], sort=True)
        key = key * (int(codes.max(initial=-1)) + 2) + (codes + 1)
        _, key = np.unique(key, return_inverse=True)
    _, first = np.unique(key, return_index=True)
    return key, frame.iloc[first][list(cols)].reset_index(drop=True)


@dataclass
class _Design:
    """Speeches reduced to (member, protocol) cells, sorted for np.add.reduceat by member and by slice."""

    slices: pd.DataFrame
    mks: pd.DataFrame
    mk_slice: np.ndarray
    slice_start: np.ndarray
    cell_mk: np.ndarray
    cell_protocol: np.ndarray
    n_protocol_codes: int
    cell_n: np.ndarray
    cell_stats: np.ndarray      # sums of w, wy, wy^2, w^2
    mk_start: np.ndarray


def _design(speeches: pd.DataFrame, by: Sequence[str], group_cols: Sequence[str]) -> _Design:
    mk_code, mks = _group_codes(speeches, list(dict.fromkeys([*by, "speaker_id", "faction_id", *group_cols])))
    if by:
        mk_slice, slices = _group_codes(mks, list(by))
    else:
        mk_slice, slices = np.zeros(len(mks), dtype=np.int64), pd.DataFrame(index=[0])
    protocol, protocol_labels = pd.factorize(speeches["protocol_name"], sort=False)
    n_protocol_codes = max(1, len(protocol_labels))
    _, cell_first, cell = np.unique(mk_code * n_protocol_codes + protocol, return_index=True, return_inverse=True)
    n_cells = len(cell_first)
    w = speeches["weight"].to_numpy(dtype=float)
    y = speeches["score10"].to_numpy(dtype=float)
    cell_mk = mk_code[cell_first]
    return _Design(
        slices=slices, mks=mks, mk_slice=mk_slice, slice_start=np.searchsorted(mk_slice, np.arange(len(slices))),
        cell_mk=cell_mk, cell_protocol=protocol[cell_first], n_protocol_codes=n_protocol_codes,
        cell_n=np.bincount(cell, minlength=n_cells),
        cell_stats=np.column_stack([np.bincount(cell, weights=col, minlength=n_cells)
                                    for col in (w, w * y, w * y * y, w * w)]),
        mk_start=np.searchsorted(cell_mk, np.arange(len(mks))),
    )


@dataclass
class _Grouping:
    """Each member's group, with groups sorted by slice."""

    column: str
    groups: pd.DataFrame
    of_mk: np.ndarray               # group of each member
    slice_start: np.ndarray
    membership: sparse.csr_matrix   # groups x members


def _grouping(design: _Design, by: Sequence[str], column: str) -> _Grouping:
    of_mk, groups = _group_codes(design.mks, [*by, column])
    group_slice = np.zeros(len(groups), dtype=np.int64)
    group_slice[of_mk] = design.mk_slice
    n_mks = len(design.mks)
    membership = sparse.csr_matrix((np.ones(n_mks), (of_mk, np.arange(n_mks))), shape=(len(groups), n_mks))
    return _Grouping(column, groups, of_mk, np.searchsorted(group_slice, np.arange(len(design.slices))), membership)


def _mk_level(sums: np.ndarray, design: _Design, deff: np.ndarray) -> dict[str, np.ndarray]:
    """Member means and variances from sums shaped (n_mks, n_replicates, 4)."""
    s0, s1, s2, sw2 = (sums[..., j] for j in range(4))
    active = s0 > 0
    with np.errstate(invalid="ignore", divide="ignore"):
        ybar = np.where(active, s1 / s0, 0.0)
        within_ss = np.clip(s2 - s1 * ybar, 0.0, None)
        within_df = np.where(active, s0 - sw2 / s0, 0.0)
        tot = np.add.reduceat(sums, design.slice_start, axis=0)
        pooled_df = np.add.reduceat(within_df, design.slice_start, axis=0)
        sigma2 = np.add.reduceat(within_ss, design.slice_start, axis=0) / pooled_df
        total_var = (tot[..., 2] - tot[..., 1] ** 2 / tot[..., 0]) / (tot[..., 0] - tot[..., 3] / tot[..., 0])
        sigma2 = np.where(pooled_df > 1e-9 * tot[..., 0], sigma2, total_var)
        sigma2 = np.where(np.isfinite(sigma2) & (sigma2 > 0), sigma2, 0.0)
        v = np.where(active, deff[design.mk_slice, None] * sigma2[design.mk_slice] / (s0**2 / sw2), np.inf)
    return {"active": active, "ybar": ybar, "v": v}


def _group_level(sums: np.ndarray, mk: dict[str, np.ndarray], design: _Design, grouping: _Grouping,
                 kind: str) -> tuple[np.ndarray, np.ndarray | None]:
    """Group estimates and, for positions, their analytic SE. Salience pools the packs."""
    member, of_mk = grouping.membership, grouping.of_mk
    with np.errstate(invalid="ignore", divide="ignore"):
        if kind != "position":
            return (member @ sums[..., 1]) / (member @ sums[..., 0]), None
        active, ybar, v = mk["active"], mk["ybar"], mk["v"]
        k = member @ active.astype(float)
        group_mean = (member @ (ybar * active)) / k
        deviations = np.where(active, (ybar - group_mean[of_mk]) ** 2, 0.0)
        correction = np.where(active, np.where(np.isfinite(v), v, 0.0) * (1 - 1 / k[of_mk]), 0.0)
        ssb = np.add.reduceat(deviations - correction, design.slice_start, axis=0)
        df = np.add.reduceat(np.clip(k - 1, 0, None), grouping.slice_start, axis=0)
        tau2 = np.where(df > 0, np.clip(ssb / df, 0.0, None), 0.0)
        weight = np.where(active, 1.0 / np.maximum(tau2[design.mk_slice] + v, 1e-12), 0.0)
        sum_weight = member @ weight
        se = np.sqrt(member @ np.where(active, weight**2 * v, 0.0)) / sum_weight
        return (member @ (weight * ybar)) / sum_weight, se


def _mk_table(design: _Design, settings: dict[str, Any]) -> tuple[pd.DataFrame, np.ndarray]:
    """Member estimates and the design effect of each slice."""
    rule = settings["filters"]["mk"]
    cell_stats, start = design.cell_stats, design.mk_start
    s0, s1, s2, sw2 = np.add.reduceat(cell_stats, start, axis=0).T
    ybar = s1 / s0
    n_protocols = np.diff(np.append(start, len(design.cell_mk)))
    n_speeches = np.add.reduceat(design.cell_n, start)
    residual = cell_stats[:, 1] - ybar[design.cell_mk] * cell_stats[:, 0]
    with np.errstate(invalid="ignore", divide="ignore"):
        cr_var = (n_protocols / (n_protocols - 1)) * np.add.reduceat(residual**2, start) / s0**2
        cr_var = np.where(n_protocols >= 2, cr_var, np.nan)
        within_df = s0 - sw2 / s0
        s2_mk = np.where(within_df > 1e-9 * s0, np.clip(s2 - s1 * ybar, 0, None) / within_df, np.nan)
        ratio = cr_var / (s2_mk / (s0**2 / sw2))
    eligible = (n_protocols >= 3) & np.isfinite(ratio) & (s2_mk > 0)
    deff = np.ones(len(design.slices))
    if eligible.any():
        medians = pd.Series(ratio[eligible]).groupby(design.mk_slice[eligible]).median()
        deff[medians.index.to_numpy()] = np.maximum(1.0, medians.to_numpy())
    sufficient = (n_speeches >= rule["min_speeches"]) & (n_protocols >= rule["min_protocols"])
    if rule["max_se"] is not None:
        sufficient &= np.nan_to_num(np.sqrt(cr_var), nan=np.inf) <= rule["max_se"]
    return design.mks.assign(estimate=ybar, n_speeches=n_speeches, sufficient=sufficient), deff


def _bootstrap(design: _Design, groupings: list[_Grouping], deff: np.ndarray, kind: str, n_boot: int,
               rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Resample each member's protocols and recompute."""
    n_cells = len(design.cell_mk)
    per_mk = np.diff(np.append(design.mk_start, n_cells))
    full = np.add.reduceat(design.cell_stats, design.mk_start, axis=0)
    full_v = _mk_level(full[:, None, :], design, deff)["v"][:, 0]
    with np.errstate(invalid="ignore", divide="ignore"):
        full_mean = np.where(full[:, 0] > 0, full[:, 1] / full[:, 0], 0.0)[:, None]
        # n resampled clusters give (n - 1) / n of the variance, and one pack gives none
        inflate = np.sqrt(per_mk / np.maximum(per_mk - 1, 1))[:, None]
        single_sd = np.where((per_mk == 1) & np.isfinite(full_v), np.sqrt(full_v), 0.0)[:, None]
    cell_first = design.mk_start[design.cell_mk][:, None]
    cell_count = per_mk[design.cell_mk][:, None]
    out = {g.column: np.empty((len(g.groups), n_boot), dtype=np.float32) for g in groupings}
    chunk = int(max(1, min(n_boot, BOOT_CHUNK_ELEMENTS // max(1, n_cells))))
    for lo in range(0, n_boot, chunk):
        reps = min(chunk, n_boot - lo)
        drawn = cell_first + (rng.random((n_cells, reps)) * cell_count).astype(np.int64)
        sums = np.add.reduceat(design.cell_stats[drawn], design.mk_start, axis=0)
        s0 = sums[..., 0]
        with np.errstate(invalid="ignore", divide="ignore"):
            within_ss = sums[..., 2] - np.where(s0 > 0, sums[..., 1] ** 2 / s0, 0.0)
            mean = full_mean + inflate * (np.where(s0 > 0, sums[..., 1] / s0, 0.0) - full_mean)
            sums[..., 1] = np.clip(mean + single_sd * rng.standard_normal(mean.shape), 0.0, 10.0) * s0
            sums[..., 2] = within_ss + np.where(s0 > 0, sums[..., 1] ** 2 / s0, 0.0)
        mk = _mk_level(sums, design, deff)
        for g in groupings:
            out[g.column][:, lo:lo + reps] = _group_level(sums, mk, design, g, kind)[0]
    return out


def _group_rows(design: _Design, g: _Grouping, estimate: np.ndarray, se: np.ndarray | None,
                boot: np.ndarray | None, rule: dict[str, float]) -> pd.DataFrame:
    key = g.of_mk[design.cell_mk] * design.n_protocol_codes + design.cell_protocol
    of_cell, inverse = np.unique(key, return_inverse=True)
    of_cell //= design.n_protocol_codes
    n_groups = len(g.groups)
    cell_s0 = np.bincount(inverse, weights=design.cell_stats[:, 0])
    s0 = np.bincount(of_cell, weights=cell_s0, minlength=n_groups)
    n_protocols = np.bincount(of_cell, minlength=n_groups)
    with np.errstate(invalid="ignore", divide="ignore"):
        effective = 1.0 / np.bincount(of_cell, weights=(cell_s0 / s0[of_cell]) ** 2, minlength=n_groups)
        if se is None:
            residual = np.bincount(inverse, weights=design.cell_stats[:, 1]) - estimate[of_cell] * cell_s0
            se = np.sqrt(np.where(n_protocols >= 2, n_protocols / (n_protocols - 1), np.nan)
                         * np.bincount(of_cell, weights=residual**2, minlength=n_groups) / s0**2)
    table = g.groups.rename(columns={g.column: "actor_id"}).assign(
        estimate=estimate, n_mks=np.bincount(g.of_mk, minlength=n_groups), n_protocols=n_protocols,
        n_speeches=np.bincount(of_cell, weights=np.bincount(inverse, weights=design.cell_n), minlength=n_groups),
        effective_sittings=effective, se_analytic=se)
    with np.errstate(invalid="ignore"):
        table["se"] = se if boot is None else np.where(np.isnan(estimate), np.nan, boot.std(axis=1, ddof=1))
    # the bootstrap misses the error of members who spoke in only one or two sittings
    larger_se = table[["se", "se_analytic"]].max(axis=1)
    table["sufficient"] = (
        table["estimate"].notna() & (table["n_speeches"] >= rule["min_speeches"])
        & (table["n_protocols"] >= rule["min_protocols"]) & (larger_se.fillna(np.inf) <= rule["max_se"])
        & (table["effective_sittings"] >= rule["min_effective_sittings"]))
    return table


def _estimate(speeches: pd.DataFrame, by: Sequence[str], group_cols: Sequence[str], settings: dict[str, Any],
              kind: str, n_boot: int, rng: np.random.Generator) -> tuple[pd.DataFrame, dict, dict]:
    by = list(by)
    design = _design(speeches, by, group_cols)
    mk_table, deff = _mk_table(design, settings)
    sums = np.add.reduceat(design.cell_stats, design.mk_start, axis=0)[:, None, :]
    mk = _mk_level(sums, design, deff)
    groupings = [_grouping(design, by, col) for col in group_cols]
    boot = _bootstrap(design, groupings, deff, kind, n_boot, rng) if n_boot > 0 else {}
    groups = {}
    for g in groupings:
        estimate, se = _group_level(sums, mk, design, g, kind)
        groups[g.column] = _group_rows(design, g, estimate[:, 0], None if se is None else se[:, 0],
                                       boot.get(g.column), settings["filters"]["group"])
    return mk_table, groups, boot


def estimate_groups(speeches: pd.DataFrame, group_col: str, kind: str, settings: dict[str, Any], n_boot: int,
                    seed: int, by: Sequence[str] = ()) -> tuple[pd.DataFrame, np.ndarray | None]:
    _, groups, boot = _estimate(speeches, by, [group_col], settings, kind, n_boot, seeded_rng(seed, group_col, kind))
    return groups[group_col], boot.get(group_col)


@dataclass
class _Context:
    settings: dict[str, Any]
    period_types: list[str]
    year_range: tuple[int, int]
    n_boot: int
    seed: int
    periods: pd.DataFrame
    actors: pd.DataFrame
    speakers: pd.Series
    party_blocs: pd.DataFrame


def _actor_table(units: pd.DataFrame) -> pd.DataFrame:
    def most_common(x: pd.Series) -> Any:
        return x.mode().iloc[0]

    parties = units.groupby("faction_id", observed=True)[["faction_label", "lineage", "bloc"]].agg(most_common)
    parties = parties.reset_index().rename(columns={"faction_id": "actor_id", "faction_label": "actor_name"})
    lineages = units.groupby("lineage", observed=True)["bloc"].agg(most_common).reset_index()
    lineages = lineages.assign(actor_type="lineage", actor_id=lineages["lineage"], actor_name=lineages["lineage"])
    blocs = pd.DataFrame({"bloc": sorted(units["bloc"].astype(str).unique())})
    blocs = blocs.assign(actor_type="bloc", actor_id=blocs["bloc"], actor_name=blocs["bloc"])
    table = pd.concat([parties.assign(actor_type="party"), lineages, blocs], ignore_index=True)
    return table[ACTOR_COLUMNS].astype({c: str for c in ("actor_id", "actor_name")})


def _context(units: pd.DataFrame, settings: dict[str, Any]) -> _Context:
    period_types = list(settings["periods"]["types"])
    year_range = (int(units["year"].min()), int(units["year"].max()))
    party_blocs = []
    for t in period_types:
        # a party gets the bloc it had most often in the period
        spoken = add_periods(units[["year", "knesset", "faction_id", "bloc"]], t, settings, year_range)
        counts = (spoken.astype({"faction_id": str, "bloc": str}).groupby(["period", "faction_id", "bloc"]).size()
                  .rename("n").reset_index())
        top = counts.sort_values(["period", "faction_id", "n", "bloc"], ascending=[True, True, False, True])
        top = top.drop_duplicates(["period", "faction_id"])
        top = top.rename(columns={"faction_id": "actor_id", "bloc": "period_bloc"})
        party_blocs.append(top[["period", "actor_id", "period_bloc"]].assign(period_type=t))
    aggregation = settings["aggregation"]
    return _Context(
        settings=settings, period_types=period_types, year_range=year_range,
        n_boot=int(aggregation["bootstrap_reps"]), seed=int(aggregation["seed"]),
        periods=pd.concat([period_table(units, t, settings) for t in period_types], ignore_index=True),
        actors=_actor_table(units),
        speakers=units.drop_duplicates("speaker_id").set_index("speaker_id")["speaker_name"],
        party_blocs=pd.concat(party_blocs, ignore_index=True),
    )


def _period_tables(dim: Dimension, speeches: pd.DataFrame, period_type: str, ctx: _Context) -> dict[str, pd.DataFrame]:
    """group_scores and mk_scores for a position dimension, or salience."""
    if speeches.empty:
        return {}
    frame = add_periods(speeches, period_type, ctx.settings, ctx.year_range)
    mks, groups, _ = _estimate(frame, ["period"], list(ACTOR_TYPES), ctx.settings, dim.kind, ctx.n_boot,
                               seeded_rng(ctx.seed, dim.id, period_type, "main"))
    groups = pd.concat([groups[col].assign(actor_type=t) for col, t in ACTOR_TYPES.items()], ignore_index=True)
    groups = groups.astype({"actor_id": str}).assign(dimension=dim.id, period_type=period_type)
    if dim.kind != "position":
        return {"salience": groups.rename(columns={"estimate": "salience"})}
    share = frame.assign(
        in_gov=(frame["coalition"] == "coalition") * frame["weight"],
        known=frame["coalition"].notna() * frame["weight"],
    ).groupby(["period", "faction_id"], observed=True)[["in_gov", "known"]].sum()
    share = (share["in_gov"] / share["known"]).rename("coalition_share").reset_index()
    share = share.rename(columns={"faction_id": "actor_id"}).astype({"actor_id": str}).assign(actor_type="party")
    groups = groups.merge(share, on=["period", "actor_id", "actor_type"], how="left").assign(estimator="hierarchical")
    return {"group_scores": groups, "mk_scores": mks.assign(dimension=dim.id, period_type=period_type)}


def _with_period_bloc(table: pd.DataFrame, ctx: _Context, key: str, on: list[str]) -> pd.DataFrame:
    """Use the faction's bloc in that period, since a corpus id can change bloc."""
    blocs = ctx.party_blocs.rename(columns={"actor_id": key})
    if key == "actor_id":
        blocs = blocs.assign(actor_type="party")
    else:
        blocs = blocs.assign(**{key: blocs[key].astype(str)})
    table = table.merge(blocs, on=on, how="left")
    table["bloc"] = table.pop("period_bloc").where(lambda x: x.notna(), table["bloc"])
    return table


def _finish(name: str, parts: list[pd.DataFrame], ctx: _Context) -> pd.DataFrame:
    table = pd.concat(parts, ignore_index=True)
    if name != "mk_scores":
        table = table.merge(ctx.actors, on=["actor_type", "actor_id"], how="left")
        table = _with_period_bloc(table, ctx, "actor_id", ["period_type", "period", "actor_type", "actor_id"])
        return table.reindex(columns=TABLE_COLUMNS[name])
    table = table.merge(ctx.periods, on=["period_type", "period"], how="left")
    party = ctx.actors[ctx.actors["actor_type"] == "party"].set_index("actor_id")
    table["speaker_name"] = table["speaker_id"].astype(str).map(ctx.speakers)
    for col, source in (("faction_name", "actor_name"), ("lineage", "lineage"), ("bloc", "bloc")):
        table[col] = table["faction_id"].astype(str).map(party[source])
    table = _with_period_bloc(table, ctx, "faction_id", ["period_type", "period", "faction_id"])
    return table.reindex(columns=TABLE_COLUMNS[name])


def _calibrate(unit_df: pd.DataFrame, dim: Dimension, gate_threshold: float) -> tuple[pd.DataFrame, str]:
    """Put committee packs on the plenary scale. Salience stays plenary, as committee attention is another thing."""
    calib = committee_calibration(unit_df, gate_threshold) if dim.kind == "position" else None
    note = f"; committee = {calib['a']:.2f} + {calib['b']:.2f} x plenary ({calib['n_pairs']} members)" if calib else ""
    return apply_calibration(unit_df, calib), note


def run(out_dir: Path | str = TABLES_DIR, settings: dict[str, Any] | None = None, units_path: Path | str = UNITS_PATH,
        db_path: Path | str = DB_PATH, factions_path: Path | str = FACTIONS_PATH, blocs_path: Path | str = BLOCS_PATH,
        log=print) -> None:
    settings = settings or load_settings()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    questions = build_questions()
    units = load_all_units(settings, units_path, load_factions(factions_path), blocs_path)
    with_committee = bool((units["protocol_type"] == "committee").any())
    answers, n_used, n_stale = load_answers(db_path, units["unit_id"], questions, settings["jev"]["model"])
    log(f"{len(units):,} packs; {n_used:,} current answers used, {n_stale:,} stale answers ignored")
    ctx = _context(units, settings)
    parts: dict[str, list[pd.DataFrame]] = {name: [] for name in TABLE_COLUMNS}
    used = []
    for dim in load_dimensions():
        unit_df = unit_scores(units, answers, dim, questions)
        if unit_df.empty:
            log(f"  {dim.id}: no current answers")
            continue
        note = ""
        if with_committee:
            unit_df, note = _calibrate(unit_df, dim, float(settings["filters"]["gate_threshold"]))
        speeches = build_speeches(unit_df, dim.kind, settings)
        used.append(pd.DataFrame({"unit_id": unit_df["unit_id"].to_numpy(), "dimension": dim.id,
                                  "usable": chunk_rule(unit_df, dim.kind, settings)[0]}))
        for period_type in ctx.period_types:
            for name, table in _period_tables(dim, speeches, period_type, ctx).items():
                parts[name].append(table)
        log(f"  {dim.id}: {len(unit_df):,} answered packs, {len(speeches):,} usable{note}")
    pd.concat(used, ignore_index=True).to_parquet(out_dir / "unit_scores.parquet", index=False)
    for name, frames in parts.items():
        _finish(name, frames, ctx).to_csv(out_dir / f"{name}.csv", index=False)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="knesset-ches aggregate", description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=TABLES_DIR, help="output directory (default: outputs/tables)")
    args = parser.parse_args(argv)
    run(args.out)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
