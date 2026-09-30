"""Draw the units for the heckle and Netanyahu questions from the plenary record.

    python -m knesset_ches.samples [--out DIR]

Writes interjections.parquet, two heckle samples and units_bibi.parquet to data/corpus/processed/. Sittings after the
corpus ends are drawn with their own seeds and appended, to keep the rows Jev has answered in place.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from knesset_ches import demographics as D
from knesset_ches.corpus import CORPUS_END

V2_SIZE = 20_000


def _turns(shard: Path) -> pd.DataFrame:
    """A shard's turns, chair aside, with the end of the turn before as context."""
    columns = ["protocol_name", "speaker_id", "is_valid_speaker", "is_chairman", "turn", "text"]
    t = pq.read_table(shard, columns=columns).to_pandas()
    t = t[~t["is_chairman"]]
    turns = t.groupby(["protocol_name", "turn"], as_index=False).agg(
        speaker_id=("speaker_id", "first"), named=("is_valid_speaker", "first"), text=("text", " ".join))
    turns = turns.sort_values(["protocol_name", "turn"])
    before = turns.groupby("protocol_name")["text"].shift(1).fillna("")
    turns["context"] = before.map(lambda s: " ".join(s.split()[-40:]))
    return turns


def _pairs(protocol: str, turns: pd.DataFrame, speeches: pd.DataFrame) -> pd.DataFrame:
    """Turns by others inside each speech of one sitting."""
    t = turns["turn"].to_numpy()
    inside = (t[:, None] > speeches["first_turn"].to_numpy()) & (t[:, None] < speeches["last_turn"].to_numpy())
    ti, si = np.nonzero(inside)
    turn, speech = turns.iloc[ti], speeches.iloc[si]
    other = turn["speaker_id"].to_numpy() != speech["speaker_id"].to_numpy()
    turn, speech = turn[other], speech[other]
    return pd.DataFrame({
        "protocol_name": protocol, "turn": turn["turn"].to_numpy(), "heckler_id": turn["speaker_id"].to_numpy(),
        "heckler_named": turn["named"].to_numpy(), "heckle": turn["text"].to_numpy(),
        "context": turn["context"].to_numpy(), "speech_id": speech["speech_id"].to_numpy(),
        "target_id": speech["speaker_id"].to_numpy(), "target_gender": speech["target_gender"].to_numpy(),
        "knesset": speech["knesset"].to_numpy(), "year": speech["year"].to_numpy(),
        "speech_words": speech["n_words"].to_numpy()})


def build_interjections(members: pd.DataFrame) -> pd.DataFrame:
    """Turns by others inside members' plenary speeches, as classifier units."""
    gender = members.set_index("speaker_id")["gender"]
    speeches = D.speeches(["speech_id", "protocol_name", "speaker_id", "knesset", "year"])
    speeches["target_gender"] = speeches["speaker_id"].map(gender).fillna("")
    spans = dict(tuple(speeches.groupby("protocol_name")))
    parts = []
    for shard in sorted(D.SENTENCES.glob("shard_*.parquet")):
        for protocol, turns in _turns(shard).groupby("protocol_name"):
            if protocol in spans:
                parts.append(_pairs(protocol, turns, spans[protocol]))
    d = pd.concat(parts, ignore_index=True)
    d["heckler_gender"] = np.where(d["heckler_named"], d["heckler_id"].map(gender).fillna(""), "")
    d["n_words"] = d["heckle"].str.split().str.len()
    d = d[d["n_words"] >= 1].copy()
    d["unit_id"] = "heckle|" + d["protocol_name"] + "|" + d["turn"].astype(str)
    d["text"] = "Speech being interrupted (last words): " + d["context"] + "\n\nInterjection: " + d["heckle"]
    # the heckler is the speaker, dated mid-year
    d["speaker_id"], d["faction_id"], d["protocol_type"], d["is_chairman"] = d["heckler_id"], "", "plenary", False
    d["date"] = pd.to_datetime(d["year"].astype(str) + "-07-01")
    d["n_chars"] = d["text"].str.len()
    return d


def heckle_sample(d: pd.DataFrame, seed: int) -> pd.DataFrame:
    """All interjections at women and a third of those at men, drawn by year."""
    rng = np.random.default_rng(seed)
    women, men = d[d["target_gender"] == "Women"], d[d["target_gender"] == "Men"]
    third = pd.concat([g.sample(frac=1 / 3, random_state=int(rng.integers(1e9))) for _, g in men.groupby("year")])
    return pd.concat([women.assign(sample_weight=1.0), third.assign(sample_weight=3.0)])


def gendered_sample(d: pd.DataFrame, rate: pd.Series) -> pd.DataFrame:
    """A share `rate` of the interjections at each gender."""
    parts = []
    for gender, g in d.groupby("target_gender"):
        n = int(round(rate[gender] * len(g)))
        parts.append(g.sample(n=n, random_state=20260923).assign(sample_weight=len(g) / n))
    return pd.concat(parts)


def netanyahu_units() -> pd.DataFrame:
    u = pd.read_parquet(D.UNITS)
    return u[~u["is_chairman"] & u["text"].str.contains("נתניהו|ביבי")]


def build(out_dir: Path) -> list[Path]:
    d = build_interjections(D.load_members())
    units = pd.read_parquet(D.UNITS, columns=["protocol_name", "date"])
    is_late = d["protocol_name"].isin(units.loc[units["date"] > CORPUS_END, "protocol_name"])
    early, late = d[~is_late], d[is_late]
    # only the first draw was shuffled
    first = heckle_sample(early, 20260922).sample(frac=1, random_state=1)
    heckles = pd.concat([first, heckle_sample(late, 20260923)], ignore_index=True)
    chair = D.presiding_keys()
    early_v2, late_v2 = D.real_heckles(early, chair), D.real_heckles(late, chair)
    rate = V2_SIZE / early_v2["target_gender"].value_counts()
    heckles_v2 = pd.concat([gendered_sample(early_v2, rate), gendered_sample(late_v2, rate)], ignore_index=True)
    files = {"interjections": d, "interjections_sample": heckles, "interjections_sample_v2": heckles_v2,
             "units_bibi": netanyahu_units()}
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, frame in files.items():
        paths.append(out_dir / f"{name}.parquet")
        frame.to_parquet(paths[-1], index=False)
    return paths


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m knesset_ches.samples", description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, default=D.PROCESSED, help="output directory")
    for path in build(ap.parse_args(argv).out):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
