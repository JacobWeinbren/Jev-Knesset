"""knesset_ches.packs on hand-made units."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from knesset_ches import packs
from knesset_ches.corpus import UNITS_SCHEMA
from knesset_ches.questions import load_settings

SEP = "\n\n* * *\n\n"


def unit(protocol: str, speaker: str, faction: str, knesset: int, date: dt.date, words: int,
         chair: bool = False) -> dict:
    text = " ".join([f"מילה{protocol[-3:]}"] * words)
    return {"unit_id": f"{protocol}|{speaker}|0", "speech_id": f"{protocol}|{speaker}", "n_chunks": 1,
            "protocol_name": protocol, "protocol_type": "plenary", "knesset": knesset, "date": date,
            "year": date.year, "speaker_id": speaker, "speaker_name": f"שם {speaker}", "faction_id": faction,
            "faction_name": f"סיעה {faction}", "faction_general_name": f"סיעה {faction}", "coalition": "coalition",
            "is_chairman": chair, "is_ocr": False, "first_turn": 1, "last_turn": 1, "n_turns": 1,
            "n_sentences": 3, "n_words": words, "n_chars": len(text), "text": text}


def test_packs_hold_whole_speeches_of_one_member_year_and_are_reproducible(tmp_path: Path):
    rows = [unit(f"24_ptm_{i:03d}", "1", "31", 24, dt.date(2021, 5, 1) + dt.timedelta(days=i), 60 + i)
            for i in range(40)]
    rows += [unit(f"24_ptm_9{i:02d}", "1", "99", 24, dt.date(2022, 2, 1 + i), 80) for i in range(3)]  # new faction
    rows += [unit("24_ptm_900", "2", "31", 24, dt.date(2022, 2, 1), 70),
             unit("25_ptm_001", "2", "31", 25, dt.date(2022, 12, 1), 70),            # same year, next Knesset
             unit("24_ptm_900", "3", "31", 24, dt.date(2022, 2, 1), 90, chair=True),
             unit("25_ptm_002", "2", "31", 25, dt.date(2024, 3, 1), 70),
             unit("25_ptm_003", "2", "31", 25, dt.date(2024, 5, 1), 70)]             # after the corpus ends
    pq.write_table(pa.Table.from_pylist(rows, schema=UNITS_SCHEMA), tmp_path / "units.parquet")
    settings = load_settings()
    settings["corpus"]["pack"] = {"max_chars": 2500, "separator": SEP}
    n_packs = packs.build_packs(tmp_path / "units.parquet", tmp_path / "packs.parquet", settings)

    units = pd.DataFrame(rows)
    frame = pq.read_table(tmp_path / "packs.parquet").to_pandas()
    members = [m for ids in frame["member_unit_ids"] for m in json.loads(ids)]
    assert len(frame) == n_packs and sorted(members) == sorted(units.loc[~units["is_chairman"], "unit_id"])
    for _, row in frame.iterrows():
        inside = units[units["unit_id"].isin(json.loads(row["member_unit_ids"]))]
        assert inside[["speaker_id", "faction_id", "knesset", "year"]].drop_duplicates().shape[0] == 1
        assert row["n_chars"] == len(row["text"]) <= 2500 and row["n_words"] == inside["n_words"].sum()
        assert row["text"].split(SEP) == inside.sort_values(["date", "unit_id"])["text"].tolist()
    big = frame[(frame["speaker_id"] == "1") & (frame["year"] == 2021)]
    spans = [units[units["unit_id"].isin(json.loads(ids))]["date"].agg(["min", "max"])
             for ids in big["member_unit_ids"]]
    overlaps = sum(a["min"] <= b["max"] and b["min"] <= a["max"] for i, a in enumerate(spans) for b in spans[i + 1:])
    assert len(big) > 3 and overlaps >= len(spans)                 # dealt at random, not in date blocks
    assert {"pack|plenary|2|31|25|2024|00", "pack|plenary|2|31|25|2024x|00"} <= set(frame["unit_id"])

    assert packs.assign_packs([5000], 2000, len(SEP), np.random.default_rng(1)) == [0]   # too long, packed alone
    packs.build_packs(tmp_path / "units.parquet", tmp_path / "again.parquet", settings)
    assert pq.read_table(tmp_path / "again.parquet").equals(pq.read_table(tmp_path / "packs.parquet"))
