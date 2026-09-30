"""Checks on the hand-kept party_crosswalk.csv and factions_en.csv. See docs/CROSSWALK_NOTES.md."""
from __future__ import annotations

import csv
import json
import re
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from knesset_ches.questions import ROOT

CROSSWALK_PATH = ROOT / "config" / "party_crosswalk.csv"
FACTIONS_EN_PATH = ROOT / "config" / "factions_en.csv"
CHES_MEANS_PATH = ROOT / "data" / "ches" / "CHES_ISRAEL_means_2021_2022.csv"
RAW = ROOT / "data" / "corpus" / "raw"
UNITS_PATH = ROOT / "data" / "corpus" / "processed" / "units.parquet"

CROSSWALK_COLUMNS = ["ches_year", "ches_party_id", "ches_party_name", "cluster_id", "knesset", "faction_id",
                     "faction_name", "speaker_ids", "date_from", "date_to", "note"]
FACTIONS_EN_COLUMNS = ["faction_id", "name_he", "name_en", "short_en", "lineage", "bloc", "knessets", "note"]
BLOCS = {"right", "religious_right", "haredi", "center", "left", "arab", "other"}
CLUSTER_IDS = {"likud", "yesh_atid", "shas", "utj", "labor", "yisrael_beiteinu", "meretz", "raam", "rz", "yamina",
               "bw_nh_statecamp", "jointlist_family"}
OPEN_END = date(2099, 12, 31)     # Knesset 25 is still sitting
KNESSET_TERMS = {24: (date(2021, 4, 6), date(2022, 11, 14)), 25: (date(2022, 11, 15), OPEN_END)}
CALENDAR_WINDOWS = {2021: (date(2021, 4, 6), date(2021, 12, 31)), 2022: (date(2022, 1, 1), date(2022, 12, 31))}


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader.fieldnames or []), list(reader)


def _read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _speakers(row: dict[str, str]) -> set[str]:
    return {s for s in row["speaker_ids"].split(";") if s}


@pytest.fixture(scope="module")
def crosswalk() -> list[dict[str, str]]:
    return _read_csv(CROSSWALK_PATH)[1]


@pytest.fixture(scope="module")
def factions_en() -> list[dict[str, str]]:
    return _read_csv(FACTIONS_EN_PATH)[1]


@pytest.fixture(scope="module")
def factions_meta() -> dict[str, dict]:
    meta = {f["faction_id"]: f for f in _read_jsonl(RAW / "factions_jsons.jsonl")}
    # factions new in the 2024-26 extension
    for f in _read_csv(RAW.parent / "extension" / "new_factions.csv")[1]:
        meta.setdefault(f["id"], {"faction_id": f["id"], "faction_name": f["name"], "knesset_numbers": [f["knesset"]]})
    return meta


@pytest.fixture(scope="module")
def ches_party_waves() -> dict[tuple[int, int], str]:
    return {(int(r["year"]), int(r["party_id"])): r["party_name"] for r in _read_csv(CHES_MEANS_PATH)[1]}


@pytest.fixture(scope="module")
def units() -> pd.DataFrame:
    if not UNITS_PATH.exists():
        pytest.skip("no units.parquet")
    table = pd.read_parquet(UNITS_PATH, columns=["knesset", "date", "faction_id", "speaker_id"])
    return table[table["faction_id"].notna() & (table["faction_id"] != "")]


def test_crosswalk_schema(crosswalk) -> None:
    assert _read_csv(CROSSWALK_PATH)[0] == CROSSWALK_COLUMNS and crosswalk
    for r in crosswalk:
        assert r["ches_year"] in {"2021", "2022"} and r["knesset"] in {"24", "25"}, r
        assert r["ches_party_id"].isdigit() and r["faction_id"].isdigit(), r
        assert r["cluster_id"] in CLUSTER_IDS and re.fullmatch(r"(\d+(;\d+)*)?", r["speaker_ids"]), r
    keys = [tuple(r[c] for c in ("ches_year", "ches_party_id", "knesset", "faction_id", "speaker_ids"))
            for r in crosswalk]
    assert len(keys) == len(set(keys)), "duplicate crosswalk rows"


def test_crosswalk_parties_are_ches_party_waves(crosswalk, ches_party_waves) -> None:
    for r in crosswalk:
        assert ches_party_waves.get((int(r["ches_year"]), int(r["ches_party_id"]))) == r["ches_party_name"], r
    clusters: dict[str, set[str]] = {}
    for r in crosswalk:
        clusters.setdefault(r["ches_party_id"], set()).add(r["cluster_id"])
    assert all(len(c) == 1 for c in clusters.values()) and set().union(*clusters.values()) == CLUSTER_IDS


def test_every_party_wave_is_mapped(crosswalk, ches_party_waves) -> None:
    assert {(int(r["ches_year"]), int(r["ches_party_id"])) for r in crosswalk} == set(ches_party_waves)


def test_crosswalk_factions_and_members_exist(crosswalk, factions_meta, factions_en) -> None:
    named = {f["faction_id"] for f in factions_en}
    members = {m["person_id"]: m for m in _read_jsonl(RAW / "all_knesset_members_jsons.jsonl")}
    for r in crosswalk:
        assert factions_meta[r["faction_id"]]["faction_name"] == r["faction_name"] and r["faction_id"] in named, r
        for speaker_id in _speakers(r):
            sat = members[speaker_id]["factions_memberships"]
            assert any(m["faction_id"] == r["faction_id"] and int(m["knesset_number"]) == int(r["knesset"])
                       for m in sat), f"{speaker_id} never sat in faction {r['faction_id']} in Knesset {r['knesset']}"


def test_crosswalk_date_windows(crosswalk) -> None:
    starts: dict[tuple[str, str], date] = {}
    for r in crosswalk:
        lo, hi = date.fromisoformat(r["date_from"]), date.fromisoformat(r["date_to"])
        term_lo, term_hi = KNESSET_TERMS[int(r["knesset"])]
        win_lo, win_hi = CALENDAR_WINDOWS[int(r["ches_year"])]
        assert lo <= hi and term_lo <= lo and hi <= term_hi and win_lo <= lo and hi <= win_hi, r
        key = (r["ches_year"], r["ches_party_id"])
        starts[key] = min(starts.get(key, date.max), lo)
    # Knesset 24 sat on both start dates, so no party-wave starts late
    for (year, _), first in starts.items():
        assert first == CALENDAR_WINDOWS[int(year)][0]


def test_no_speech_is_mapped_to_two_parties(crosswalk) -> None:
    """Parties that share a faction in a wave need disjoint member lists."""
    groups: dict[tuple[str, ...], list[dict[str, str]]] = {}
    for r in crosswalk:
        groups.setdefault((r["ches_year"], r["knesset"], r["faction_id"]), []).append(r)
    for key, rows in groups.items():
        if len({r["ches_party_id"] for r in rows}) == 1:
            continue
        assert all(_speakers(r) for r in rows), f"a faction is shared without speaker_ids: {key}"
        for i, a in enumerate(rows):
            for b in rows[i + 1:]:
                assert a["ches_party_id"] == b["ches_party_id"] or not _speakers(a) & _speakers(b), key


def test_joint_list_is_split_by_constituent_party(crosswalk) -> None:
    rows = [r for r in crosswalk if r["faction_id"] == "138" and r["ches_year"] == "2022"]
    assert {r["ches_party_name"]: _speakers(r) for r in rows} == {
        "Hadash-Ta'al": {"30066", "30067", "30719", "560", "30070"},  # Odeh, Touma-Sliman, Cassif; Tibi, Saadi
        "Balad": {"30751"},                                           # Abu Shehadeh
    }


def test_factions_en_schema(factions_en, factions_meta) -> None:
    assert _read_csv(FACTIONS_EN_PATH)[0] == FACTIONS_EN_COLUMNS
    for column in ("faction_id", "short_en"):      # short_en labels chart series
        values = [f[column] for f in factions_en]
        assert len(values) == len(set(values)), f"duplicate {column}"
    blocs: dict[str, set[str]] = {}
    for f in factions_en:
        assert f["name_he"] == factions_meta[f["faction_id"]]["faction_name"], f
        assert f["name_en"].strip() and f["name_en"].isascii() and 0 < len(f["short_en"]) <= 12, f
        assert re.fullmatch(r"[a-z][a-z0-9]*(_[a-z0-9]+)*", f["lineage"]) and f["bloc"] in BLOCS, f
        knessets = [int(k) for k in f["knessets"].split(";")]
        assert knessets == sorted(set(knessets)) and all(13 <= k <= 25 for k in knessets), f
        blocs.setdefault(f["lineage"], set()).add(f["bloc"])
    assert {lineage: b for lineage, b in blocs.items() if len(b) > 1} == {}   # a lineage that changes bloc is a typo


def test_every_faction_is_named(factions_en, factions_meta, units) -> None:
    named = {f["faction_id"] for f in factions_en}
    in_metadata = {fid for fid, f in factions_meta.items() if any(13 <= int(k) <= 25 for k in f["knesset_numbers"])}
    in_corpus = set(units.loc[units["knesset"].between(13, 25), "faction_id"].astype(str))
    assert (in_metadata | in_corpus) - named == set()


def test_every_party_wave_matches_corpus_speech(crosswalk, units) -> None:
    late = units[units["knesset"] >= 24].drop_duplicates()
    late = late.assign(date=pd.to_datetime(late["date"]).dt.date, faction_id=late["faction_id"].astype(str),
                       speaker_id=late["speaker_id"].astype(str))
    matched: dict[tuple[str, str], int] = {}
    for r in crosswalk:
        lo, hi = date.fromisoformat(r["date_from"]), date.fromisoformat(r["date_to"])
        hit = ((late["knesset"] == int(r["knesset"])) & (late["faction_id"] == r["faction_id"])
               & (late["date"] >= lo) & (late["date"] <= hi))
        if _speakers(r):
            hit &= late["speaker_id"].isin(_speakers(r))
        key = (r["ches_year"], r["ches_party_name"])
        matched[key] = matched.get(key, 0) + int(hit.sum())
    assert {key: n for key, n in matched.items() if n == 0} == {}
