"""validate.py on known answers, then on the synthetic Knesset with made-up experts."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from knesset_ches import validate as v
from knesset_ches.questions import load_settings
from tests import synth

DIMENSIONS = ["lrgen", "lrecon", "galtan", "israel_palestine", "environment"]
NOISE_VARIABLE = "environment"          # experts rate it at random
THIN_CELL = (2021, "101", "galtan")     # only two experts


def test_pearson_matches_direct_computation_on_the_replicated_sample() -> None:
    rng = np.random.default_rng(1)
    cluster = np.arange(14) // 2
    x = rng.normal(5, 2, 14)
    y = rng.normal(5, 2, 14) + 0.8 * (x - 5)
    w = np.repeat(rng.integers(0, 3, 7), 2).astype(float)      # whole clusters are drawn 0, 1 or 2 times
    drawn = w.astype(int)
    assert v.pearson(x[None], y[None], w[None], cluster)[0] == pytest.approx(
        stats.pearsonr(np.repeat(x, drawn), np.repeat(y, drawn))[0])
    w[cluster > 1] = 0                                          # two clusters left: no correlation
    assert np.isnan(v.pearson(x[None], y[None], w[None], cluster)[0])


def test_r_with_interval_recovers_a_known_correlation_and_covers_it() -> None:
    rng = np.random.default_rng(3)
    cluster = np.repeat(np.arange(60), 2)
    shared = rng.normal(size=60)[cluster]                      # a party's two waves are nearly the same point
    x = 5 + shared + 0.2 * rng.normal(size=120)
    y = 5 + 0.7 * shared + np.sqrt(1 - 0.49) * rng.normal(size=60)[cluster] + 0.2 * rng.normal(size=120)
    x_boot = x[:, None] + 0.05 * rng.normal(size=(120, 200))
    ratings = [np.array([c - 0.1, c, c + 0.1]) for c in y]
    out = v.r_with_interval(x, y, cluster, np.full(120, 0.05), x_boot, ratings, 1500, np.random.default_rng(4))
    assert out["pearson_r"] == pytest.approx(stats.pearsonr(x, y)[0])
    assert out["pearson_r_bca_lo"] < 0.66 < out["pearson_r_bca_hi"]                      # the population r


def test_crosswalk_rows_are_unioned_and_respect_speakers_and_dates(tmp_path: Path) -> None:
    frame = pd.DataFrame({
        "knesset": [24, 24, 24, 24, 25], "faction_id": ["1", "1", "2", "2", "1"],
        "speaker_id": ["a", "b", "c", "d", "a"], "row": range(5),
        "date": pd.to_datetime(["2021-05-01", "2022-02-01", "2021-06-01", "2021-07-01", "2022-12-01"])})
    pd.DataFrame({
        "ches_year": [2021, 2021, 2022], "ches_party_id": ["9"] * 3, "ches_party_name": ["P"] * 3,
        "cluster_id": ["p"] * 3, "knesset": [24, 24, 24], "faction_id": ["1", "2", "1"], "faction_name": [""] * 3,
        "speaker_ids": ["", "d", ""], "date_from": ["2021-04-06", "2021-04-06", "2022-01-01"],
        "date_to": ["2021-12-31", "2021-12-31", "2022-11-14"], "note": [""] * 3,
    }).to_csv(tmp_path / "c.csv", index=False)
    selected = v.party_wave_frame(frame, v.load_crosswalk(tmp_path / "c.csv"))
    assert sorted(zip(selected["party_wave"], selected["row"])) == [("2021:9", 0), ("2021:9", 3), ("2022:9", 1)]


def _fixture_crosswalk(units: pd.DataFrame, path: Path) -> pd.DataFrame:
    """Each Knesset 24 faction is a CHES party in both waves. Faction 115 is added with no speech."""
    factions = pd.DataFrame(synth.FACTIONS).set_index(0)
    windows = {2021: [(24, "2021-04-06", "2021-12-31")],
               2022: [(24, "2022-01-01", "2022-11-14"), (25, "2022-11-15", "2022-12-31")]}
    rows = []
    for faction_id in [*sorted(units.loc[units["knesset"] == 24, "faction_id"].unique()), "115"]:
        name_he, name_en, lineage = factions.loc[faction_id, [1, 3, 4]]
        for year, parts in windows.items():
            rows += [{"ches_year": year, "ches_party_id": f"9{faction_id}", "ches_party_name": name_en,
                      "cluster_id": lineage, "knesset": knesset, "faction_id": faction_id, "faction_name": name_he,
                      "speaker_ids": "", "date_from": start, "date_to": end, "note": "synthetic"}
                     for knesset, start, end in parts]
    table = pd.DataFrame(rows)
    table.to_csv(path, index=False)
    return table


def _fixture_experts(world: synth.SynthWorld, crosswalk: pd.DataFrame, path: Path) -> None:
    """Ten experts a wave, rating the true position plus noise."""
    rng = np.random.default_rng(99)
    truth = world.positions.set_index(["faction_id", "year", "dimension"])["position"]
    rows = []
    for (year, party_id, faction_id), _ in crosswalk.groupby(["ches_year", "ches_party_id", "faction_id"]):
        for expert in range(10):
            row = {"year": year, "id": (year - 2021) * 100 + expert + 1, "party_id": int(party_id)}
            for variable in DIMENSIONS:
                if variable == NOISE_VARIABLE:
                    value = rng.uniform(0, 10)
                else:
                    value = truth[(faction_id, year, variable)] + rng.normal(0, 1.0)
                thin = (year, faction_id, variable) == THIN_CELL and expert >= 2
                row[variable] = np.nan if thin else np.clip(np.round(value), 0, 10)
            rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)


def _run(world: synth.SynthWorld, crosswalk: Path, experts: Path, out: Path, n_boot: int) -> dict[str, pd.DataFrame]:
    settings = load_settings()
    settings["aggregation"]["bootstrap_reps"] = 100
    return v.run(out, settings, world.units_path, world.db_path, world.factions_path, world.blocs_path, crosswalk,
                 experts, n_boot=n_boot, log=lambda *_: None)


@pytest.fixture(scope="module")
def e2e(tmp_path_factory: pytest.TempPathFactory) -> dict:
    root = tmp_path_factory.mktemp("validate")
    world = synth.generate(root / "world", n_sittings=1200, speeches_per_sitting=30, seed=11, dimensions=DIMENSIONS)
    crosswalk = _fixture_crosswalk(pd.read_parquet(world.units_path, columns=["knesset", "faction_id"]),
                                   root / "crosswalk.csv")
    _fixture_experts(world, crosswalk, root / "experts.csv")
    tables = _run(world, root / "crosswalk.csv", root / "experts.csv", root / "tables", 400)
    return {"root": root, "world": world, "tables": tables}


def test_e2e_recovers_the_known_positions(e2e: dict) -> None:
    summary = e2e["tables"]["validation_summary"].set_index("dimension")
    for dimension in ("lrgen", "lrecon", "galtan", "israel_palestine"):
        row = summary.loc[dimension]
        assert row["status"] == "ok" and row["n_clusters"] >= 10 and row["pearson_r"] > 0.9
        assert row["pearson_r_bca_lo"] < row["pearson_r"] <= 1 and np.isfinite(row["r_floor_bloc"])
    points = e2e["tables"]["validation_points"]
    lrgen = points[points["dimension"] == "lrgen"]
    assert summary.loc["lrgen", "pearson_r"] == pytest.approx(stats.pearsonr(lrgen["jev"], lrgen["ches"])[0])
    assert abs(summary.loc[NOISE_VARIABLE, "pearson_r"]) < 0.6


def test_e2e_points_leave_out_thin_cells_and_silent_parties(e2e: dict) -> None:
    points = pd.read_csv(e2e["root"] / "tables" / "validation_points.csv", dtype={"party_wave": str})
    assert list(points.columns) == v.POINT_COLUMNS
    assert not points["party_wave"].str.endswith(":9115").any()              # no speech
    year, faction, variable = THIN_CELL
    thin = points[points["party_wave"] == f"{year}:9{faction}"]
    assert len(thin) and variable not in set(thin["dimension"])


def test_too_few_parties_is_inconclusive(e2e: dict, tmp_path: Path) -> None:
    root = e2e["root"]
    crosswalk = pd.read_csv(root / "crosswalk.csv", dtype=str)
    crosswalk[crosswalk["faction_id"].isin(["101", "110", "115"])].to_csv(tmp_path / "small.csv", index=False)
    summary = _run(e2e["world"], tmp_path / "small.csv", root / "experts.csv", tmp_path / "tables", 50)
    summary = summary["validation_summary"]
    assert (summary["status"] == v.INCONCLUSIVE).all() and summary["pearson_r"].isna().all()
