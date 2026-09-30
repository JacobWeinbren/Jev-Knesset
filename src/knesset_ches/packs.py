"""Pack each member's speech for a year into a few long texts, to send the 50 questions fewer times.

    python -m knesset_ches.packs
    python -m knesset_ches.packs --dir data/corpus/processed_committee

packs.parquet has the columns of units.parquet, so the classifier reads it unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from knesset_ches.corpus import CORPUS_END, OUT_DIRS
from knesset_ches.questions import load_settings

GROUP_KEYS = ["protocol_type", "speaker_id", "faction_id", "knesset", "year"]
EXTRA_FIELDS = [pa.field("n_member_units", pa.int32()), pa.field("n_member_protocols", pa.int32()),
                pa.field("member_unit_ids", pa.string())]


def _seed(key: list[Any], seed: int) -> int:
    digest = hashlib.sha256(("|".join(map(str, key)) + f"|{seed}").encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def assign_packs(n_chars: list[int], max_chars: int, sep_len: int, rng: np.random.Generator) -> list[int]:
    """Deal units at random, each to the smallest pack, adding a pack and dealing again when one overflows."""
    order = rng.permutation(len(n_chars))
    total = sum(n_chars) + sep_len * max(0, len(n_chars) - 1)
    k = max(1, math.ceil(total / max_chars))
    while True:
        sizes, out, ok = [0] * k, [0] * len(n_chars), True
        for i in order:
            p = min(range(k), key=lambda j: sizes[j])
            if sizes[p] and sizes[p] + sep_len + n_chars[i] > max_chars:
                ok = False
                break
            sizes[p] += n_chars[i] + (sep_len if sizes[p] else 0)
            out[i] = p
        if ok or k >= len(n_chars):
            break
        k += 1
    dense = {p: j for j, p in enumerate(sorted(set(out)))}      # drop empty packs
    return [dense[p] for p in out]


def build_group(group: pd.DataFrame, extension: bool, max_chars: int, separator: str, seed: int) -> list[dict]:
    """The packs of one member-year."""
    first = group.iloc[0]
    key = [first[k] for k in GROUP_KEYS]
    if extension:
        key[-1] = f"{key[-1]}x"                                    # keeps the ids of packs scored before the extension
    rng = np.random.default_rng(_seed(key, seed))
    group = group.sort_values("unit_id", kind="stable")            # fixed order before the deal
    pack_of = assign_packs(group["n_chars"].tolist(), max_chars, len(separator), rng)
    rows = []
    for k in sorted(set(pack_of)):
        members = group[[p == k for p in pack_of]].sort_values(["date", "unit_id"], kind="stable")
        text = separator.join(members["text"])
        unit_id = "pack|" + "|".join(map(str, key)) + f"|{k:02d}"
        coalition = members["coalition"].dropna()
        rows.append({
            "unit_id": unit_id, "speech_id": unit_id, "n_chunks": 1, "protocol_name": unit_id,
            "protocol_type": first["protocol_type"], "knesset": int(first["knesset"]),
            "date": members["date"].sort_values().iloc[len(members) // 2], "year": int(first["year"]),
            "speaker_id": first["speaker_id"], "speaker_name": first["speaker_name"],
            "faction_id": first["faction_id"],
            "faction_name": members["faction_name"].mode().iat[0] if members["faction_name"].notna().any() else None,
            "faction_general_name": first["faction_general_name"],
            "coalition": coalition.mode().iat[0] if len(coalition) else None,
            "is_chairman": False, "is_ocr": bool(members["is_ocr"].any()), "first_turn": 0, "last_turn": 0,
            "n_turns": int(members["n_turns"].sum()), "n_sentences": int(members["n_sentences"].sum()),
            "n_words": int(members["n_words"].sum()), "n_chars": len(text), "text": text,
            "n_member_units": len(members), "n_member_protocols": int(members["protocol_name"].nunique()),
            "member_unit_ids": json.dumps(members["unit_id"].tolist(), ensure_ascii=False),
        })
    return rows


def build_packs(units_path: Path, out_path: Path, settings: dict[str, Any]) -> int:
    """Write packs.parquet one Knesset at a time and return the number of packs."""
    max_chars = int(settings["corpus"]["pack"]["max_chars"])
    separator = str(settings["corpus"]["pack"]["separator"])
    seed = int(settings["aggregation"]["seed"])
    schema = pa.schema(list(pq.read_schema(units_path)) + EXTRA_FIELDS)
    knessets = sorted(set(pq.read_table(units_path, columns=["knesset"]).column("knesset").to_pylist()))
    tmp = out_path.with_suffix(".parquet.tmp")
    n_packs = 0
    with pq.ParquetWriter(tmp, schema, compression="zstd") as writer:
        for knesset in knessets:
            units = pq.read_table(units_path, filters=[("knesset", "=", knesset)]).to_pandas()
            units = units[~units["is_chairman"].astype(bool)]
            units = units.assign(extension=units["date"] > CORPUS_END)
            rows: list[dict] = []
            for keys, group in units.groupby(GROUP_KEYS + ["extension"], sort=True, dropna=False):
                rows.extend(build_group(group.drop(columns="extension"), keys[-1], max_chars, separator, seed))
            if rows:
                writer.write_table(pa.Table.from_pandas(pd.DataFrame(rows), schema=schema, preserve_index=False))
                n_packs += len(rows)
    tmp.replace(out_path)
    return n_packs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="knesset-ches packs", description="Pack units.parquet into packs.parquet.")
    parser.add_argument("--dir", type=Path, default=OUT_DIRS["plenary"],
                        help="folder that holds units.parquet; packs.parquet is written next to it")
    args = parser.parse_args(argv)
    n_packs = build_packs(args.dir / "units.parquet", args.dir / "packs.parquet", load_settings())
    print(f"wrote {n_packs:,} packs to {args.dir / 'packs.parquet'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
