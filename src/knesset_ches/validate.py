"""Compare Jev's party positions with the Chapel Hill Expert Survey (CHES-Israel, waves 2021 and 2022).

    python -m knesset_ches.validate [--out DIR]
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from knesset_ches import aggregate as agg
from knesset_ches.questions import ROOT, build_questions, load_dimensions, load_settings

CHES_EXPERTS_PATH = ROOT / "data" / "ches" / "CHES_ISRAEL_expert_level_2021_2022.csv"
EXPERT_COLUMN_ALIASES = {"democratic_v_jewish_state": "dem_vs_jewish_state"}   # name in the expert file
N_BOOT = 2000
INCONCLUSIVE = "inconclusive (coverage)"
POINT_COLUMNS = ["dimension", "party_wave", "ches_party_name", "cluster_id", "bloc", "jev", "jev_se", "ches",
                 "n_experts"]
# blog_opening filters on tag, kind, wave and party_set
SUMMARY_COLUMNS = ["tag", "dimension", "kind", "wave", "party_set", "status", "n_clusters", "pearson_r",
                   "pearson_r_bca_lo", "pearson_r_bca_hi", "r_floor_bloc"]


def load_experts(path: Path | str, variables: Sequence[str]) -> pd.DataFrame:
    experts = pd.read_csv(path).rename(columns={v: k for k, v in EXPERT_COLUMN_ALIASES.items()})
    long = experts.melt(id_vars=["year", "party_id"], value_vars=[v for v in variables if v in experts.columns],
                        var_name="ches_variable", value_name="value").dropna(subset=["value"])
    long = long.rename(columns={"year": "ches_year", "party_id": "ches_party_id"})
    return long.astype({"ches_year": int, "ches_party_id": int, "value": float}).astype({"ches_party_id": str})


def load_crosswalk(path: Path | str) -> pd.DataFrame:
    table = pd.read_csv(path, dtype=str, keep_default_na=False)
    table = table.assign(party_wave=table["ches_year"] + ":" + table["ches_party_id"],
                         date_from=pd.to_datetime(table["date_from"]), date_to=pd.to_datetime(table["date_to"]))
    return table.astype({"ches_year": int, "knesset": int})


def row_mask(frame: pd.DataFrame, row: Any) -> np.ndarray:
    date = frame["date"].to_numpy()
    mask = (frame["knesset"].to_numpy() == row.knesset) & (frame["faction_id"].astype(str).to_numpy() == row.faction_id)
    mask &= (date >= np.datetime64(row.date_from)) & (date <= np.datetime64(row.date_to))
    speakers = [s.strip() for s in str(row.speaker_ids).split(";") if s.strip()]
    if speakers:
        mask &= np.isin(frame["speaker_id"].astype(str).to_numpy(), speakers)
    return mask


def party_wave_frame(frame: pd.DataFrame, crosswalk: pd.DataFrame) -> pd.DataFrame:
    """`frame` cut to the crosswalk's speech, tagged with party_wave and ches_year."""
    parts = []
    for party_wave, block in crosswalk.groupby("party_wave", sort=False):
        mask = np.zeros(len(frame), dtype=bool)
        for row in block.itertuples():
            mask |= row_mask(frame, row)
        if mask.any():
            parts.append(frame[mask].assign(party_wave=party_wave, ches_year=int(block["ches_year"].iloc[0])))
    return pd.concat(parts, ignore_index=True) if parts else frame.iloc[:0]


def party_wave_table(crosswalk: pd.DataFrame, blocs_path: Path | str) -> pd.DataFrame:
    """One row per CHES party-wave, with the bloc of the first faction it names."""
    first = crosswalk.drop_duplicates("party_wave").reset_index(drop=True)
    bloc = agg.project_blocs(first, pd.Series("other", index=first.index), blocs_path)
    columns = ["party_wave", "ches_year", "ches_party_id", "ches_party_name", "cluster_id", "bloc"]
    return first.assign(bloc=bloc)[columns]


def jev_estimates(unit_frames: dict[str, pd.DataFrame], crosswalk: pd.DataFrame, settings: dict[str, Any],
                  seed: int) -> tuple[pd.DataFrame, np.ndarray]:
    """aggregate's estimates per dimension and party-wave, with replicates in the same row order."""
    tables, boots = [], []
    n_boot = int(settings["aggregation"]["bootstrap_reps"])
    for dimension, unit_df in unit_frames.items():
        subset = party_wave_frame(agg.build_speeches(unit_df, "position", settings), crosswalk)
        if subset.empty:
            continue
        table, boot = agg.estimate_groups(subset, "party_wave", "position", settings, n_boot, seed, by=["ches_year"])
        tables.append(table.rename(columns={"actor_id": "party_wave"}).assign(dimension=dimension))
        boots.append(np.asarray(boot, dtype=float))
    return pd.concat(tables, ignore_index=True), np.vstack(boots)


def build_points(dimensions: Sequence[tuple[str, str]], jev: pd.DataFrame, experts: pd.DataFrame,
                 waves: pd.DataFrame, min_experts: int) -> pd.DataFrame:
    """Party-waves with enough experts and enough speech, per dimension."""
    keys = ["ches_year", "ches_party_id", "ches_variable"]
    cells = experts.groupby(keys, as_index=False).agg(ches=("value", "mean"), n_experts=("value", "size"))
    jev = jev.assign(boot_row=np.arange(len(jev)))
    parts = []
    for dimension, variable in dimensions:
        est = jev[jev["dimension"] == dimension].drop(columns=["ches_year", "dimension"])
        rated = cells[cells["ches_variable"] == variable].drop(columns="ches_variable")
        block = waves.merge(est, on="party_wave", how="left").merge(rated, on=["ches_year", "ches_party_id"],
                                                                    how="left")
        parts.append(block.assign(dimension=dimension, ches_variable=variable))
    points = pd.concat(parts, ignore_index=True)
    used = (points["n_experts"] >= min_experts) & points["sufficient"].fillna(False).astype(bool)
    points = points[used].rename(columns={"estimate": "jev", "se": "jev_se"}).reset_index(drop=True)
    return points.astype({"n_experts": int, "boot_row": int})


def pearson(x: np.ndarray, y: np.ndarray, w: np.ndarray, cluster: np.ndarray) -> np.ndarray:
    """Weighted Pearson r of B samples at once. x, y and the draw counts w are [B, n]."""
    present = ((w @ np.eye(int(cluster.max()) + 1)[cluster]) > 0).sum(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        n = w.sum(1)
        mx, my = (w * x).sum(1) / n, (w * y).sum(1) / n
        dx, dy = x - mx[:, None], y - my[:, None]
        sxx, syy, sxy = (w * dx * dx).sum(1) / n, (w * dy * dy).sum(1) / n, (w * dx * dy).sum(1) / n
        r = sxy / np.sqrt(sxx * syy)
    return np.where((present < 3) | ~(sxx > 1e-12) | ~(syy > 1e-12), np.nan, r)


def bca_interval(theta: float, boot: np.ndarray, jack: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    """BCa interval, with the acceleration from a delete-one-cluster jackknife."""
    boot, jack = boot[np.isfinite(boot)], jack[np.isfinite(jack)]
    if not np.isfinite(theta) or len(boot) < 50 or len(jack) < 3:
        return np.nan, np.nan
    share = (np.sum(boot < theta) + 0.5 * np.sum(boot == theta)) / len(boot)
    if not 0 < share < 1:
        return np.nan, np.nan
    z0 = stats.norm.ppf(share)
    d = jack.mean() - jack
    denom = 6.0 * (np.sum(d**2) ** 1.5)
    acc = float(np.sum(d**3) / denom) if denom > 0 else 0.0
    z = stats.norm.ppf([alpha / 2, 1 - alpha / 2])
    lo, hi = np.quantile(boot, stats.norm.cdf(z0 + (z0 + z) / (1 - acc * (z0 + z))))
    return float(lo), float(hi)


def r_with_interval(x: np.ndarray, y: np.ndarray, cluster: np.ndarray, x_se: np.ndarray, x_boot: np.ndarray,
                    ratings: Sequence[np.ndarray], n_boot: int, rng: np.random.Generator) -> dict[str, float]:
    """Pearson r of Jev (x) and CHES (y), with a cluster bootstrap BCa interval."""
    code = pd.factorize(cluster)[0]
    n_clusters = int(code.max()) + 1
    point = float(pearson(x[None], y[None], np.ones((1, len(x))), code)[0])
    picks = rng.integers(0, n_clusters, size=(n_boot, n_clusters))
    counts = np.zeros((n_boot, n_clusters))
    np.add.at(counts, (np.repeat(np.arange(n_boot), n_clusters), picks.ravel()), 1.0)
    y_drawn = np.tile(y, (n_boot, 1))
    for i, cell in enumerate(ratings):
        if len(cell) >= 2:
            y_drawn[:, i] = cell[rng.integers(0, len(cell), size=(n_boot, len(cell)))].mean(1)
    column = np.minimum((rng.random(n_boot) * x_boot.shape[1]).astype(int), x_boot.shape[1] - 1)
    # used where Jev's own replicate is missing
    fallback = x + np.nan_to_num(x_se) * rng.standard_normal((n_boot, len(x)))
    x_drawn = x_boot[:, column].T
    boot = pearson(np.where(np.isfinite(x_drawn), x_drawn, fallback), y_drawn, counts[:, code], code)
    jack_w = (code[None, :] != np.arange(n_clusters)[:, None]).astype(float)
    jack = pearson(np.tile(x, (n_clusters, 1)), np.tile(y, (n_clusters, 1)), jack_w, code)
    lo, hi = bca_interval(point, boot, jack)
    return {"pearson_r": point, "pearson_r_bca_lo": lo, "pearson_r_bca_hi": hi}


def bloc_floor(rows: pd.DataFrame) -> float:
    """r of CHES with the mean CHES score of the other parties in the same bloc."""
    y = rows["ches"].to_numpy(dtype=float)
    cluster, bloc = rows["cluster_id"].to_numpy(), rows["bloc"].to_numpy()
    pred = np.full(len(y), np.nan)
    for i in range(len(y)):
        other = cluster != cluster[i]
        same = other & (bloc == bloc[i])
        pred[i] = y[same].mean() if same.any() else (y[other].mean() if other.any() else np.nan)
    keep = np.isfinite(pred)
    if keep.sum() < 3 or np.ptp(pred[keep]) == 0 or np.ptp(y[keep]) == 0:
        return np.nan
    return float(np.corrcoef(pred[keep], y[keep])[0, 1])


def summarise(points: pd.DataFrame, boot: np.ndarray, experts: pd.DataFrame, dimension: str, variable: str,
              min_clusters: int, n_boot: int, seed: int) -> dict[str, Any]:
    """The validation_summary row of one dimension, both waves pooled."""
    rows = points[points["dimension"] == dimension]
    out = {"tag": "primary", "dimension": dimension, "kind": "position", "wave": "pooled", "party_set": "own",
           "n_clusters": rows["cluster_id"].nunique()}
    if out["n_clusters"] < min_clusters:
        return out | {"status": INCONCLUSIVE}
    rated = experts[experts["ches_variable"] == variable]
    by_cell = {key: g.to_numpy() for key, g in rated.groupby(["ches_year", "ches_party_id"])["value"]}
    ratings = [by_cell[(int(y), p)] for y, p in zip(rows["ches_year"], rows["ches_party_id"])]
    rng = agg.seeded_rng(seed, "validate", dimension, variable, "pooled", "own")
    out |= r_with_interval(rows["jev"].to_numpy(dtype=float), rows["ches"].to_numpy(dtype=float),
                           rows["cluster_id"].to_numpy(), rows["jev_se"].to_numpy(dtype=float),
                           boot[rows["boot_row"].to_numpy()], ratings, n_boot, rng)
    return out | {"status": "ok", "r_floor_bloc": bloc_floor(rows)}


def run(out_dir: Path | str = agg.TABLES_DIR, settings: dict[str, Any] | None = None,
        units_path: Path | str = agg.UNITS_PATH, db_path: Path | str = agg.DB_PATH,
        factions_path: Path | str = agg.FACTIONS_PATH, blocs_path: Path | str = agg.BLOCS_PATH,
        crosswalk_path: Path | str | None = None, experts_path: Path | str = CHES_EXPERTS_PATH,
        n_boot: int = N_BOOT, log=print) -> dict[str, pd.DataFrame]:
    settings = settings or load_settings()
    validation = settings["validation"]
    seed = int(settings["aggregation"]["seed"])
    dimensions = [(d.id, d.ches_variable) for d in load_dimensions() if d.kind == "position" and d.ches_variable]
    experts = load_experts(experts_path, [v for _, v in dimensions])
    crosswalk = load_crosswalk(crosswalk_path or ROOT / validation["crosswalk"])
    factions = agg.load_factions(factions_path)
    unit_frames = _unit_frames(units_path, db_path, factions, blocs_path, crosswalk, [d for d, _ in dimensions],
                               settings, log)
    jev, boot = jev_estimates(unit_frames, crosswalk, settings, seed)
    points = build_points(dimensions, jev, experts, party_wave_table(crosswalk, blocs_path),
                          int(validation["min_experts_per_cell"]))
    summary = pd.DataFrame([summarise(points, boot, experts, d, v, int(validation["min_party_clusters"]), n_boot, seed)
                            for d, v in dimensions])
    tables = {"validation_summary": summary.reindex(columns=SUMMARY_COLUMNS),
              "validation_points": points[POINT_COLUMNS]}
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(Path(out_dir) / f"{name}.csv", index=False)
    return tables


def _unit_frames(units_path: Path | str, db_path: Path | str, factions: pd.DataFrame, blocs_path: Path | str,
                 crosswalk: pd.DataFrame, dimensions: Sequence[str], settings: dict[str, Any],
                 log) -> dict[str, pd.DataFrame]:
    """Scored packs inside the crosswalk's windows, by dimension."""
    units = agg.load_units(units_path, factions, blocs_path)
    inside = np.zeros(len(units), dtype=bool)
    for row in crosswalk.itertuples():
        inside |= row_mask(units, row)
    units = units[inside].reset_index(drop=True)
    questions = build_questions()
    answers, n_used, _ = agg.load_answers(db_path, units["unit_id"], questions, settings["jev"]["model"])
    log(f"{len(units):,} packs inside a crosswalk window; {n_used:,} current answers used")
    frames = {d.id: agg.unit_scores(units, answers, d, questions) for d in load_dimensions() if d.id in dimensions}
    return {k: v for k, v in frames.items() if len(v)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="knesset-ches validate", description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=agg.TABLES_DIR, help="output directory (default: outputs/tables)")
    args = parser.parse_args(argv)
    summary = run(args.out)["validation_summary"]
    for row in summary[summary["status"] == "ok"].itertuples():
        print(f"  {row.dimension}: r = {row.pearson_r:.2f} ({row.pearson_r_bca_lo:.2f} to {row.pearson_r_bca_hi:.2f}), "
              f"{row.n_clusters} parties")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
