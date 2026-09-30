"""aggregate.py on made-up numbers, then on the synthetic Knesset in tests/synth.py."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from knesset_ches import aggregate as agg
from knesset_ches.questions import build_questions, load_settings
from tests import synth

DIMENSIONS = ["lrgen", "lrecon", "israel_palestine", "ukraine_support", "lrecon_salience", "israel_palestine_salience"]
POSITIONS = ["lrgen", "lrecon", "israel_palestine"]


@pytest.fixture(scope="module")
def settings() -> dict[str, Any]:
    return load_settings()


def test_units_are_combined_into_speeches_before_weighting(settings: dict[str, Any]) -> None:
    base = {"protocol_name": "p1", "date": pd.Timestamp("2021-06-01"), "year": 2021, "knesset": 24, "speaker_id": "s1",
            "faction_id": "f1", "lineage": "f1", "bloc": "Left", "coalition": "coalition", "is_chairman": False,
            "gate": 0.9, "calib_b": 1.0}
    units = pd.DataFrame([base | r for r in [
        {"speech_id": "A", "n_words": 1000, "score10": 2.0},
        {"speech_id": "A", "n_words": 600, "score10": 6.0},
        {"speech_id": "A", "n_words": 400, "score10": 10.0, "gate": 0.2},
        {"speech_id": "B", "n_words": 400, "score10": 5.0, "is_chairman": True},
        {"speech_id": "C", "n_words": 100, "score10": 8.0, "calib_b": 0.5},        # an equated committee pack
    ]])
    position = agg.build_speeches(units, "position", settings).set_index("speech_id")
    assert list(position.index) == ["A", "C"]
    assert position.loc["A", "score10"] == pytest.approx((1000 * 2 + 600 * 6) / 1600)
    assert position.loc["A", "weight"] == pytest.approx(np.sqrt(1600))       # not sqrt(1000) + sqrt(600)
    assert position.loc["C", "weight"] == pytest.approx(10 * 0.5**2)
    salience = agg.build_speeches(units, "salience", settings).set_index("speech_id")
    assert list(salience.index) == ["A", "C"] and salience.loc["A", "weight"] == 2000
    assert salience.loc["A", "score10"] == pytest.approx(9600 / 2000)


def test_period_labels(settings: dict[str, Any]) -> None:
    frame = pd.DataFrame({"year": [1992, 1993, 1994, 1995, 2019, 2019, 2020, 2024],
                          "knesset": [13, 13, 13, 13, 21, 22, 23, 25], "protocol_name": list("abcdefgh"),
                          "date": pd.to_datetime(["1992-08-01", "1993-01-05", "1994-03-01", "1995-02-02",
                                                  "2019-05-01", "2019-11-01", "2020-04-01", "2024-02-01"])})
    assert agg.add_periods(frame, "knesset", settings)["period"].tolist()[3:6] == ["13", "21-22-23", "21-22-23"]
    rolling = agg.add_periods(frame, "rolling2", settings)
    assert sorted(rolling.loc[rolling["year"] == 1993, "period"]) == ["1993", "1994"]
    assert rolling.loc[rolling["year"] == 1992, "period"].tolist() == ["1993"]    # no half-empty window "1992"
    assert rolling.loc[rolling["year"] == 2024, "period"].tolist() == ["2024"]
    years = agg.period_table(frame, "rolling2", settings).set_index("period")
    assert years.loc["2020", ["period_start", "period_end"]].tolist() == ["2019-01-01", "2020-12-31"]
    terms = agg.period_table(frame, "knesset", settings).set_index("period")
    assert terms.loc["21-22-23", ["period_start", "period_end"]].tolist() == ["2019-05-01", "2020-04-01"]


def _random_speeches(seed: int = 0, n: int = 1500, n_protocols: int = 15) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    speaker = rng.integers(0, 40, n)
    faction = np.where(speaker == 39, 7, speaker % 6)                 # faction 7 has one member
    protocol = rng.integers(0, n_protocols, n)
    true = faction * 1.2 + rng.normal(0, 0.8, 40)[speaker]
    frame = pd.DataFrame({
        "protocol_name": [f"p{g}" for g in protocol], "speaker_id": [f"m{s}" for s in speaker],
        "faction_id": [f"f{f}" for f in faction], "bloc": np.where(faction < 3, "left", "right"),
        "score10": np.clip(true + rng.normal(0, 1.0, n_protocols)[protocol] + rng.normal(0, 1.5, n), 0, 10),
        "weight": np.sqrt(rng.integers(60, 3000, n)),
    })
    frame.loc[(frame["speaker_id"] == "m3") & (protocol >= n_protocols // 2), "faction_id"] = "f4"   # a switcher
    return frame


def _plain(frame: pd.DataFrame, group_col: str, settings: dict[str, Any]) -> tuple[pd.DataFrame, pd.Series,
                                                                                   pd.Series]:
    """The engine's formulas written out as plain loops."""
    rule = settings["filters"]["mk"]
    mks = []
    for (speaker, faction, group), part in frame.groupby(["speaker_id", "faction_id", group_col]):
        y, w = part["score10"].to_numpy(), part["weight"].to_numpy()
        mean = (w * y).sum() / w.sum()
        sums = part.assign(r=w * (y - mean)).groupby("protocol_name")["r"].sum().to_numpy()
        g = len(sums)
        mks.append({"speaker_id": speaker, "faction_id": faction, "group": group, "estimate": mean,
                    "se": np.sqrt(g / (g - 1) * (sums**2).sum() / w.sum() ** 2) if g > 1 else np.nan,
                    "n_speeches": len(part), "n_protocols": g, "n_eff": w.sum() ** 2 / (w**2).sum(),
                    "within_ss": (w * (y - mean) ** 2).sum(), "within_df": w.sum() - (w**2).sum() / w.sum()})
    mks = pd.DataFrame(mks)
    mks["sufficient"] = (mks["n_speeches"] >= rule["min_speeches"]) & (mks["n_protocols"] >= rule["min_protocols"])
    s2 = mks["within_ss"] / mks["within_df"]
    eligible = (mks["n_protocols"] >= 3) & (s2 > 0)
    deff = max(1.0, float(np.median((mks["se"] ** 2 / (s2 / mks["n_eff"]))[eligible])))
    v = deff * mks["within_ss"].sum() / mks["within_df"].sum() / mks["n_eff"]
    by_group = mks.groupby("group")["estimate"]
    k = by_group.transform("size")
    ssb = ((mks["estimate"] - by_group.transform("mean")) ** 2).sum() - (v * (1 - 1 / k)).sum()
    weight = 1 / (max(0.0, ssb / (by_group.size() - 1).sum()) + v)
    hierarchical = (weight * mks["estimate"]).groupby(mks["group"]).sum() / weight.groupby(mks["group"]).sum()
    pooled = frame.groupby(group_col).apply(lambda t: (t["weight"] * t["score10"]).sum() / t["weight"].sum())
    return mks, hierarchical, pooled


@pytest.mark.parametrize("group_col", ["faction_id", "bloc"])
def test_engine_matches_plain_formulas(settings: dict[str, Any], group_col: str) -> None:
    frame = _random_speeches()
    plain_mks, hierarchical, pooled = _plain(frame, group_col, settings)
    for kind, expected in (("position", hierarchical), ("salience", pooled)):
        table, boot = agg.estimate_groups(frame, group_col, kind, settings, n_boot=0, seed=0)
        np.testing.assert_allclose(table["estimate"], expected.reindex(table["actor_id"]), rtol=1e-10)
        assert boot is None
    mks = agg._estimate(frame, [], [group_col], settings, "position", 0, agg.seeded_rng(0))[0]
    mks = mks.set_index(["speaker_id", "faction_id"])
    plain_mks = plain_mks.set_index(["speaker_id", "faction_id"]).reindex(mks.index)
    np.testing.assert_allclose(mks["estimate"], plain_mks["estimate"], rtol=1e-9)
    assert (mks["n_speeches"] == plain_mks["n_speeches"]).all()
    assert (mks["sufficient"] == plain_mks["sufficient"]).all()
    assert sorted(mks.loc["m3"].index) == ["f3", "f4"]


def test_bootstrap_is_seeded_and_gives_the_errors(settings: dict[str, Any]) -> None:
    frame = _random_speeches(n_protocols=60)
    a, boot_a = agg.estimate_groups(frame, "faction_id", "position", settings, n_boot=300, seed=5)
    b, boot_b = agg.estimate_groups(frame, "faction_id", "position", settings, n_boot=300, seed=5)
    _, boot_c = agg.estimate_groups(frame, "faction_id", "position", settings, n_boot=300, seed=6)
    pd.testing.assert_frame_equal(a, b)
    np.testing.assert_array_equal(boot_a, boot_b)
    assert boot_a.shape == (len(a), 300) and not np.array_equal(boot_a, boot_c)
    np.testing.assert_allclose(boot_a.std(axis=1, ddof=1), a["se"], rtol=1e-5)
    alone = a[a["actor_id"] == "f7"].iloc[0]
    assert alone["n_mks"] == 1 and alone["se"] > 0.05            # a lone member still adds sampling error
    pooled, _ = agg.estimate_groups(frame, "faction_id", "salience", settings, n_boot=300, seed=5)
    pooled = pooled[pooled["n_speeches"] > 100]
    assert (pooled["se"] / pooled["se_analytic"]).between(0.6, 1.4).all()


def test_committee_scores_are_equated_onto_the_plenary_scale() -> None:
    rng = np.random.default_rng(3)
    member = np.repeat(np.arange(60), 8)
    in_committee = np.tile([False] * 4 + [True] * 4, 60)
    truth = rng.uniform(1, 9, 60)[member]
    unit_df = pd.DataFrame({"speaker_id": member.astype(str), "knesset": 24, "gate": 0.9, "is_chairman": False,
                            "protocol_type": np.where(in_committee, "committee", "plenary"),
                            "score10": np.where(in_committee, 1 + 0.8 * truth, truth) + rng.normal(0, 0.1, 480)})
    calib = agg.committee_calibration(unit_df, 0.5)
    assert calib["a"] == pytest.approx(1.0, abs=0.1) and calib["b"] == pytest.approx(0.8, abs=0.02)
    assert calib["n_pairs"] == 60
    equated = agg.apply_calibration(unit_df, calib)
    assert np.abs(equated.loc[in_committee, "score10"] - truth[in_committee]).mean() < 0.2
    assert (equated.loc[in_committee, "calib_b"] == calib["b"]).all()
    assert len(agg.apply_calibration(unit_df, None)) == 240
    assert agg.committee_calibration(unit_df[member < 20], 0.5) is None        # too few members in both


def test_project_blocs_are_keyed_by_faction_and_knesset(tmp_path: Path) -> None:
    path = tmp_path / "faction_blocs.csv"
    path.write_text("faction_id,name_en,knesset,bloc,basis\n7,A,*,Right,x\n7,A,24,Orthodox,x\n8,B,15,Sectoral,x\n")
    units = pd.DataFrame({"faction_id": ["7", "7", "8", "8"], "knesset": [20, 24, 15, 16]})
    fallback = pd.Series(["a", "b", "c", "d"])
    assert agg.project_blocs(units, fallback, path).tolist() == ["Right", "Orthodox", "Sectoral", "d"]


@pytest.fixture(scope="module")
def world(tmp_path_factory: pytest.TempPathFactory) -> synth.SynthWorld:
    return synth.generate(tmp_path_factory.mktemp("synth"), n_sittings=400, speeches_per_sitting=40, seed=11,
                          dimensions=DIMENSIONS)


@pytest.fixture(scope="module")
def tables(world: synth.SynthWorld) -> Path:
    settings = load_settings()
    settings["aggregation"] |= {"committee": "none", "bootstrap_reps": 200}
    settings["periods"]["types"] = ["year", "knesset"]
    out = world.out_dir / "tables"
    agg.run(out, settings, world.units_path, world.db_path, world.factions_path, world.blocs_path, log=lambda *_: None)
    return out


def _read(tables: Path, name: str) -> pd.DataFrame:
    dtype = {"period": str, "actor_id": str, "faction_id": str, "speaker_id": str}
    return pd.read_csv(tables / f"{name}.csv", dtype=dtype, low_memory=False)


def test_tables_have_their_columns(tables: Path) -> None:
    for name, columns in agg.TABLE_COLUMNS.items():
        table = _read(tables, name)
        assert list(table.columns) == columns and len(table), name
    used = pd.read_parquet(tables / "unit_scores.parquet")
    assert set(used["dimension"]) == set(DIMENSIONS) and used["usable"].any() and not used["usable"].all()


def test_stale_answers_are_ignored(world: synth.SynthWorld) -> None:
    units = agg.load_units(world.units_path, agg.load_factions(world.factions_path), world.blocs_path)
    model = load_settings()["jev"]["model"]
    _, n_used, n_stale = agg.load_answers(world.db_path, units["unit_id"], build_questions(), model)
    assert (n_used, n_stale) == (world.n_current_answers, world.n_stale_answers)


def test_estimates_track_the_truth(world: synth.SynthWorld, tables: Path) -> None:
    scores = _read(tables, "group_scores")
    assert set(scores["actor_type"]) == {"party", "lineage", "bloc"}
    assert scores["coalition_share"].dropna().between(0, 1).all()
    truth = world.positions.groupby(["bloc", "dimension", "year"])["position"].mean().rename("truth").reset_index()
    bloc = scores[scores["sufficient"] & scores["dimension"].isin(POSITIONS) & (scores["actor_type"] == "bloc")
                  & (scores["period_type"] == "year")]
    bloc = bloc.assign(year=bloc["period"].astype(int)).merge(
        truth, left_on=["actor_id", "dimension", "year"], right_on=["bloc", "dimension", "year"])
    assert len(bloc) > 150 and np.corrcoef(bloc["estimate"], bloc["truth"])[0, 1] > 0.9
    terms = set(scores.loc[scores["period_type"] == "knesset", "period"])
    assert "21-22-23" in terms and not {"21", "22", "23"} & terms


def test_members_and_valid_from(tables: Path) -> None:
    mks = _read(tables, "mk_scores")
    assert not mks.duplicated(["dimension", "period_type", "period", "speaker_id", "faction_id"]).any()
    assert mks["speaker_name"].notna().all() and mks["bloc"].notna().all()
    good = mks[mks["sufficient"]]
    assert len(good) > 100 and (good["n_speeches"] >= 3).all()
    ukraine = mks[mks["dimension"] == "ukraine_support"]
    assert len(ukraine) and ukraine["period_end"].str[:4].astype(int).min() >= 2022


def test_salience_recovers_talk_rates(world: synth.SynthWorld, tables: Path) -> None:
    salience = _read(tables, "salience")
    for dimension, position in (("lrecon_salience", "lrecon"), ("israel_palestine_salience", "israel_palestine")):
        rows = salience[(salience["dimension"] == dimension) & (salience["actor_type"] == "party")
                        & (salience["period_type"] == "knesset") & salience["sufficient"]]
        rates = world.talk_rates[world.talk_rates["dimension"] == position].set_index("faction_id")["rate"]
        by_party = rows.groupby("actor_id")["salience"].mean()
        assert stats.spearmanr(by_party, rates.reindex(by_party.index)).statistic > 0.8
