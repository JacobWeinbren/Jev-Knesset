"""Cut the Knesset Corpus sentence shards into classification units (units.parquet).

    python -m knesset_ches.corpus
    python -m knesset_ches.corpus --protocol-type committee

Pass 1 turns each raw shard into a slim parquet file. Pass 2 groups the sentences into speeches and cuts long ones
into chunks. Thresholds are in the corpus block of config/settings.yaml.
"""

from __future__ import annotations

import argparse
import bz2
import datetime as dt
import json
import math
import os
import re
import shutil
from bisect import bisect_left
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from knesset_ches.questions import ROOT, load_settings

RAW_DIR = ROOT / "data" / "corpus" / "raw"
OUT_DIRS = {
    "plenary": ROOT / "data" / "corpus" / "processed",
    "committee": ROOT / "data" / "corpus" / "processed_committee",
}
SHARD_DIR_TEMPLATE = "protocols_sentences/{protocol_type}_no_morph_sentences_shards_bzip2_files"
CORPUS_END = dt.date(2024, 4, 3)        # last sitting in the corpus
EXTRACT_BATCH_ROWS = 50_000
UNITS_ROW_GROUP_ROWS = 20_000

SENTENCE_SCHEMA = pa.schema([
    ("shard", pa.int16()), ("line", pa.int32()),        # line is 0-based in the raw file
    ("protocol_name", pa.string()), ("protocol_type", pa.string()), ("knesset", pa.int16()), ("date", pa.date32()),
    ("is_ocr", pa.bool_()), ("speaker_id", pa.string()), ("speaker_name", pa.string()),
    ("is_valid_speaker", pa.bool_()), ("is_chairman", pa.bool_()), ("is_mk", pa.bool_()),
    ("turn", pa.int32()), ("sent", pa.int32()),
    ("faction_id", pa.string()), ("faction_name", pa.string()), ("faction_general_name", pa.string()),
    ("coalition", pa.string()), ("text", pa.string()),
])

UNITS_SCHEMA = pa.schema([
    ("unit_id", pa.string()), ("speech_id", pa.string()), ("n_chunks", pa.int32()),
    ("protocol_name", pa.string()), ("protocol_type", pa.string()), ("knesset", pa.int32()), ("date", pa.date32()),
    ("year", pa.int32()), ("speaker_id", pa.string()), ("speaker_name", pa.string()),
    ("faction_id", pa.string()), ("faction_name", pa.string()), ("faction_general_name", pa.string()),
    ("coalition", pa.string()), ("is_chairman", pa.bool_()), ("is_ocr", pa.bool_()),
    ("first_turn", pa.int32()), ("last_turn", pa.int32()), ("n_turns", pa.int32()), ("n_sentences", pa.int32()),
    ("n_words", pa.int32()), ("n_chars", pa.int32()), ("text", pa.string()),
])


# Pass 1: raw shards to slim sentence files


def shard_number(path: Path) -> int:
    return int(re.search(r"shard_(\d+)", path.name).group(1))


def list_shards(raw_dir: Path, protocol_type: str) -> list[Path]:
    folder = raw_dir / SHARD_DIR_TEMPLATE.format(protocol_type=protocol_type)
    shards = sorted(folder.glob("*.jsonl.bz2"), key=shard_number)
    if not shards:
        raise FileNotFoundError(f"no .jsonl.bz2 shards under {folder}")
    return shards


def _text_or_none(value: Any) -> str | None:
    text = "" if value is None else str(value).strip()
    return text or None


def slim_record(raw: dict[str, Any], shard: int, line: int) -> dict[str, Any]:
    """The fields the build needs. speaker_is_knesset_member is a string in old shards and a bool in new ones."""
    date = raw.get("protocol_date")
    return {
        "shard": shard,
        "line": line,
        "protocol_name": raw["protocol_name"],
        "protocol_type": raw["protocol_type"],
        "knesset": int(raw["knesset_number"]),
        "date": dt.date.fromisoformat(date[:10]) if date else None,
        "is_ocr": bool(raw.get("is_ocr_output")),
        "speaker_id": _text_or_none(raw.get("speaker_id")),
        "speaker_name": _text_or_none(raw.get("speaker_name")),
        "is_valid_speaker": bool(raw.get("is_valid_speaker")),
        "is_chairman": bool(raw.get("is_chairman")),
        "is_mk": str(raw.get("speaker_is_knesset_member")) == "True",
        "turn": int(raw["turn_num_in_protocol"]),
        "sent": int(raw["sent_num_in_turn"]),
        "faction_id": _text_or_none(raw.get("faction_id")),
        "faction_name": _text_or_none(raw.get("current_faction_name")),
        "faction_general_name": _text_or_none(raw.get("faction_general_name")),
        "coalition": _text_or_none(raw.get("member_of_coalition_or_opposition")),
        "text": raw.get("sentence_text") or "",
    }


def extract_shard(raw_path: Path, out_path: Path) -> int:
    """Write one raw shard as a slim parquet file and return the row count."""
    shard = shard_number(raw_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(".parquet.tmp")
    n_rows = 0
    batch: list[dict[str, Any]] = []
    with bz2.open(raw_path, "rb") as lines, pq.ParquetWriter(tmp_path, SENTENCE_SCHEMA, compression="zstd") as writer:
        for line_no, line in enumerate(lines):
            if not line.strip():
                continue
            record = slim_record(json.loads(line), shard, line_no)
            if record["date"] is None:      # some committee protocols are undated
                continue
            batch.append(record)
            if len(batch) >= EXTRACT_BATCH_ROWS:
                writer.write_table(pa.Table.from_pylist(batch, schema=SENTENCE_SCHEMA))
                n_rows += len(batch)
                batch = []
        if batch:
            writer.write_table(pa.Table.from_pylist(batch, schema=SENTENCE_SCHEMA))
            n_rows += len(batch)
    tmp_path.replace(out_path)
    return n_rows


def _extract_task(task: tuple[Path, Path]) -> tuple[str, int]:
    raw_path, out_path = task
    return out_path.name, extract_shard(raw_path, out_path)


def extract_all(raw_dir: Path, sentences_dir: Path, protocol_type: str, workers: int) -> list[Path]:
    """Pass 1 for shards that changed. Returns all slim files, extension shards too."""
    slim_paths: list[Path] = []
    tasks: list[tuple[Path, Path]] = []
    for raw_path in list_shards(raw_dir, protocol_type):
        out_path = sentences_dir / f"shard_{shard_number(raw_path):02d}.parquet"
        slim_paths.append(out_path)
        if not out_path.exists() or out_path.stat().st_mtime < raw_path.stat().st_mtime:
            tasks.append((raw_path, out_path))
    extension = sorted(set(sentences_dir.glob("shard_*.parquet")) - set(slim_paths), key=shard_number)
    print(f"extract: {len(slim_paths)} shards, {len(tasks)} to do; read as they are: "
          f"{', '.join(path.name for path in extension) or 'none'}", flush=True)
    if tasks:
        tasks.sort(key=lambda task: -task[0].stat().st_size)
        with ProcessPoolExecutor(max_workers=min(workers, len(tasks))) as pool:
            for name, n_rows in pool.map(_extract_task, tasks):
                print(f"  {name}: {n_rows:,} sentences", flush=True)
    return slim_paths + extension


# Pass 2: sentences to turns, speeches and units


@dataclass(frozen=True)
class UnitParams:
    mk_only: bool
    min_words: int
    max_chars: int
    max_interruption_words: int

    @classmethod
    def from_settings(cls, settings: dict[str, Any]) -> UnitParams:
        corpus = settings["corpus"]
        return cls(bool(corpus["mk_only"]), int(corpus["min_words"]), int(corpus["max_chars"]),
                   int(corpus["max_interruption_words"]))


SPEAKER_FIELDS = ("speaker_id", "speaker_name", "is_valid_speaker", "is_mk", "faction_id", "faction_name",
                  "faction_general_name", "coalition", "is_ocr")


@dataclass
class Turn:
    """Speaker fields come from the turn's first sentence."""

    turn: int
    flagged_chair: bool                 # the corpus' is_chairman
    speaker_id: str | None
    speaker_name: str | None
    is_valid_speaker: bool
    is_mk: bool
    faction_id: str | None
    faction_name: str | None
    faction_general_name: str | None
    coalition: str | None
    is_ocr: bool
    sentences: list[str] = field(default_factory=list)
    n_words: int = 0
    presiding: bool = False             # set by detect_presiding


_WORD_RE = re.compile(r"\S*\w\S*")


def count_words(text: str) -> int:
    return len(_WORD_RE.findall(text))


def group_turns(rows: list[dict[str, Any]]) -> list[Turn]:
    """Group one protocol's sorted sentence rows into turns, dropping empty and repeated sentences."""
    turns: list[Turn] = []
    previous_key = None
    for row in rows:
        key = (row["turn"], row["sent"])
        if key == previous_key:
            continue
        previous_key = key
        if not turns or turns[-1].turn != row["turn"]:
            turns.append(Turn(row["turn"], row["is_chairman"], *(row[name] for name in SPEAKER_FIELDS)))
        current = turns[-1]
        text = " ".join(row["text"].split())
        if not text:
            continue
        current.sentences.append(text)
        current.n_words += count_words(text)
        current.flagged_chair = current.flagged_chair or row["is_chairman"]
        current.is_ocr = current.is_ocr or row["is_ocr"]
    return turns


# Who presides in plenary protocols. The patterns come from short chair turns set against short floor turns.
CHAIR_ANCHOR_MAX_WORDS = 40      # longest turn for weak patterns and the next-turn cue
CHAIR_MIN_RUN = 3                # anchors in a row for a stint in the chair
CHAIR_EDGE_CHARS = 150           # long turns are searched for strong patterns at the ends only
_ADDRESS_CHARS = 80
_ADDRESS_RE = re.compile(r'(?:אדוני|גברתי|גבירתי|כבוד)\s+היו(?:שבת?[-־– ]ראש|"ר|״ר)')
_CHAIR_STRONG_RE = re.compile(
    "|".join(
        [
            r"רשות הדיבור ל",                                                   # "the floor goes to ..."
            r"אחרי[וה]\s*[-–—,]?\s*(?:חבר|השר|סגן|סגנית|ראש)",                    # "... and after him, MK ..."
            r"תודה(?: רבה)? ל(?:חבר|שר|סגן|סגנית|מזכיר|ראש|יושב)",               # "thank you to MK ..."
            r"מי בעד|מי נגד|מי נמנע|נא להצביע|ההצבעה החלה|אנחנו מצביעים",
            r"הצבעה מס'|אין מתנגדים|אין נמנעים",                                 # vote results
            r"אני קובעת? (?:ש|כי)",                                             # declaring a result
            r"קוראת? אותך לסדר",                                                # call to order
            r"מתכבדת? (?:לפתוח|להזמין)|אני (?:פותח|פותחת|נועל|נועלת) את|ישיבה זו נעולה|הישיבה נעולה",
            r"מזמינה? את (?:חבר|השר|סגן|סגנית|ראש|יושב)",                        # inviting a speaker
            r"שאילת[אה] מס'|נוכחת?\s*[;,–-]\s*לפרוטוקול",                        # question time
            r"(?:עוברים|נעבור|ניגשים|ניגש)(?: עכשיו| כעת| עתה)? ל(?:הצבעה|הצבעות|נושא|סעיף|הצע|שאילת|סדר|דיון)",
            r"ממשיכים בסדר-היום",                                               # next on the agenda
            r"נא לשבת|שאלה נוספת ל",
        ]
    )
)
# "thank you" and "please" count alone or with a form of address. Floor speakers say them to hecklers too.
_CHAIR_WEAK_RE = re.compile(r"^\W*תודה|בבקשה")
_CHAIR_BARE_RE = re.compile(r"^\W*(?:תודה(?: רבה)?|בבקשה)\W*$")
_VOCATIVE_RE = re.compile(r"חברת? הכנסת|חברי הכנסת|השרה?|אדוני|גברתי|גבירתי|רבותי")


def _addresses_chair(turn: Turn) -> bool:
    """Opens with "אדוני היושב-ראש" or the like."""
    return bool(turn.sentences) and bool(_ADDRESS_RE.search(" ".join(turn.sentences)[:_ADDRESS_CHARS]))


def _has_chair_formula(turn: Turn, next_turn: Turn | None, next_addresses_chair: bool) -> bool:
    """The turn, or the reply to it, sounds like the chair."""
    if not turn.is_valid_speaker or turn.n_words == 0:
        return False
    text = " ".join(turn.sentences)
    if turn.n_words > CHAIR_ANCHOR_MAX_WORDS:
        edges = text[:CHAIR_EDGE_CHARS] + " " + text[-CHAIR_EDGE_CHARS:]
        return bool(_CHAIR_STRONG_RE.search(edges))
    if _CHAIR_STRONG_RE.search(text) or _CHAIR_BARE_RE.search(text):
        return True
    if _CHAIR_WEAK_RE.search(text) and _VOCATIVE_RE.search(text):
        return True
    # the next speaker addresses the chair
    return next_turn is not None and next_turn.speaker_id != turn.speaker_id and next_addresses_chair


def detect_presiding(turns: list[Turn], use_heuristic: bool = True) -> None:
    """Set Turn.presiding for one protocol. The corpus flags one person per sitting, floor speeches included, and
    none of the deputy speakers. In plenaries, runs of chair formulas correct this."""
    if not use_heuristic:
        for turn in turns:
            turn.presiding = turn.flagged_chair and turn.is_valid_speaker
        return

    n = len(turns)
    addresses = [_addresses_chair(turn) for turn in turns]
    formulas = [
        not addresses[i]
        and _has_chair_formula(turns[i], turns[i + 1] if i + 1 < n else None, i + 1 < n and addresses[i + 1])
        for i in range(n)
    ]
    flagged = [turn.flagged_chair and turn.is_valid_speaker and turn.n_words > 0 for turn in turns]

    # floor mode
    on_floor = [False] * n
    on_floor_since: dict[str | None, int] = {}                        # speaker: index of their address turn
    last_evidence: tuple[str | None, int, bool] = (None, -1, False)   # speaker, index, was a formula
    for i, turn in enumerate(turns):
        speaker_id = turn.speaker_id
        if addresses[i]:
            on_floor_since[speaker_id] = i
        elif formulas[i] and speaker_id in on_floor_since:
            previous_speaker, previous_index, previous_was_formula = last_evidence
            if (previous_speaker == speaker_id and previous_was_formula
                    and previous_index > on_floor_since[speaker_id]):
                for j in range(previous_index, i):                    # back in the chair from the first of the two
                    on_floor[j] = on_floor[j] and turns[j].speaker_id != speaker_id
                del on_floor_since[speaker_id]
        if formulas[i] or (flagged[i] and speaker_id not in on_floor_since):
            last_evidence = (speaker_id, i, formulas[i])
        on_floor[i] = speaker_id in on_floor_since

    # stints, dropping stray runs shortest first
    anchors = [(formulas[i] or flagged[i]) and not on_floor[i] for i in range(n)]
    kept = [i for i in range(n) if anchors[i]]
    runs: list[list[int]] = []
    for max_stray in range(1, CHAIR_MIN_RUN + 1):
        while True:
            runs = []
            for i in kept:
                if runs and turns[runs[-1][0]].speaker_id == turns[i].speaker_id:
                    runs[-1].append(i)
                else:
                    runs.append([i])
            stray = {i for run in runs if len(run) < max_stray for i in run}
            if not stray:
                break
            kept = [i for i in kept if i not in stray]
    in_stint = [False] * n
    for run in runs:
        # widen the run, since long announcements are rarely anchors themselves
        speaker_id = turns[run[0]].speaker_id
        start, end = run[0], run[-1]
        while start > 0 and not (anchors[start - 1] and turns[start - 1].speaker_id != speaker_id):
            start -= 1
        while end < n - 1 and not (anchors[end + 1] and turns[end + 1].speaker_id != speaker_id):
            end += 1
        for i in range(start, end + 1):
            in_stint[i] = in_stint[i] or turns[i].speaker_id == speaker_id

    for i, turn in enumerate(turns):
        turn.presiding = turn.is_valid_speaker and (in_stint[i] or turn.flagged_chair) and not on_floor[i]


def merge_speeches(turns: list[Turn], max_interruption_words: int) -> list[list[Turn]]:
    """Group turns into speeches. A turn of max_interruption_words or more closes other speakers' speeches."""
    open_speeches: dict[tuple[str | None, bool], list[Turn]] = {}
    speeches: list[list[Turn]] = []
    for turn in turns:
        key = (turn.speaker_id, turn.presiding)
        if turn.n_words >= max_interruption_words:
            open_speeches = {key: open_speeches[key]} if key in open_speeches else {}
        if key in open_speeches:
            open_speeches[key].append(turn)
        else:
            speech = [turn]
            speeches.append(speech)
            open_speeches[key] = speech
    return speeches


def split_long_sentence(sentence: str, max_chars: int) -> list[str]:
    """Cut a run-on sentence at spaces into pieces of at most max_chars."""
    if len(sentence) <= max_chars:
        return [sentence]
    pieces: list[str] = []
    current = ""
    for word in sentence.split(" "):
        while len(word) > max_chars:
            if current:
                pieces.append(current)
                current = ""
            pieces.append(word[:max_chars])
            word = word[max_chars:]
        if current and len(current) + 1 + len(word) > max_chars:
            pieces.append(current)
            current = word
        else:
            current = f"{current} {word}" if current else word
    if current:
        pieces.append(current)
    return pieces


def _balanced_ranges(lengths: list[int], k: int) -> list[tuple[int, int]]:
    """k consecutive ranges of roughly equal length."""
    n = len(lengths)
    cumulative: list[int] = []
    running = 0
    for length in lengths:
        running += length + 1                              # plus a separator
        cumulative.append(running)
    ranges: list[tuple[int, int]] = []
    start = 0
    for j in range(1, k):
        target = running * j / k
        i = bisect_left(cumulative, target)
        end = i + 1
        if i > 0 and target - cumulative[i - 1] <= cumulative[i] - target:
            end = i
        end = min(max(end, start + 1), n - (k - j))
        ranges.append((start, end))
        start = end
    ranges.append((start, n))
    return ranges


def split_chunks(lengths: list[int], max_chars: int) -> list[tuple[int, int]]:
    """The fewest roughly equal sentence ranges that each fit in max_chars."""
    n = len(lengths)
    total = sum(lengths) + n - 1
    if total <= max_chars:
        return [(0, n)]
    k = math.ceil(total / max_chars)
    while True:
        ranges = _balanced_ranges(lengths, k)
        if k >= n or all(sum(lengths[a:b]) + (b - a - 1) <= max_chars for a, b in ranges):
            return ranges
        k += 1


def _join(pieces: list[tuple[int, str]]) -> str:
    """Spaces within a turn, newlines between turns."""
    parts = [pieces[0][1]]
    for (previous_turn, _), (turn, piece) in zip(pieces, pieces[1:]):
        parts.append(" " if turn == previous_turn else "\n")
        parts.append(piece)
    return "".join(parts)


def speech_units(speech: list[Turn], protocol: dict[str, Any], params: UnitParams) -> list[dict[str, Any]]:
    """Unit rows of one speech that pass the filters."""
    head = speech[0]
    if not head.is_valid_speaker or (params.mk_only and not (head.is_mk and head.faction_id)):
        return []
    pieces = [(turn.turn, piece) for turn in speech for sentence in turn.sentences
              for piece in split_long_sentence(sentence, params.max_chars)]
    if not pieces:
        return []
    ranges = split_chunks([len(piece) for _, piece in pieces], params.max_chars)
    speech_id = f"{protocol['protocol_name']}|{head.turn}"
    units: list[dict[str, Any]] = []
    for chunk, (start, end) in enumerate(ranges):
        text = _join(pieces[start:end])
        n_words = count_words(text)
        if n_words < params.min_words:
            continue
        units.append({
            "unit_id": f"{speech_id}|{chunk}",
            "speech_id": speech_id,
            "n_chunks": len(ranges),
            **protocol,
            "year": protocol["date"].year,
            "speaker_id": head.speaker_id,
            "speaker_name": head.speaker_name,
            "faction_id": head.faction_id,
            "faction_name": head.faction_name,
            "faction_general_name": head.faction_general_name,
            "coalition": head.coalition,
            "is_chairman": head.presiding,
            "is_ocr": any(turn.is_ocr for turn in speech),
            "first_turn": head.turn,
            "last_turn": speech[-1].turn,
            "n_turns": len({turn for turn, _ in pieces[start:end]}),
            "n_sentences": end - start,
            "n_words": n_words,
            "n_chars": len(text),
            "text": text,
        })
    return units


def protocol_units(rows: list[dict[str, Any]], params: UnitParams) -> list[dict[str, Any]]:
    protocol = {key: rows[0][key] for key in ("protocol_name", "protocol_type", "knesset", "date")}
    turns = group_turns(rows)
    detect_presiding(turns, use_heuristic=protocol["protocol_type"] == "plenary")
    speeches = merge_speeches(turns, params.max_interruption_words)
    return [unit for speech in speeches for unit in speech_units(speech, protocol, params)]


def knesset_units(slim_paths: list[Path], knesset: int, params: UnitParams) -> list[dict[str, Any]]:
    """A protocol can span two shards, so all of a Knesset's shards are read together."""
    table = pa.concat_tables([pq.read_table(path, filters=[("knesset", "=", knesset)]) for path in slim_paths])
    table = table.sort_by([(name, "ascending") for name in ("protocol_name", "turn", "sent", "shard", "line")])
    names = table.column("protocol_name").to_numpy(zero_copy_only=False)
    bounds = [0, *(np.flatnonzero(names[1:] != names[:-1]) + 1).tolist(), len(names)]
    units: list[dict[str, Any]] = []
    for start, end in zip(bounds[:-1], bounds[1:]):
        units.extend(protocol_units(table.slice(start, end - start).to_pylist(), params))
    units.sort(key=lambda unit: (unit["date"], unit["protocol_name"]))   # stable: keeps turn and chunk order
    return units


def _build_task(task: tuple[int, list[Path], Path, UnitParams]) -> int:
    knesset, slim_paths, part_path, params = task
    units = knesset_units(slim_paths, knesset, params)
    table = pa.Table.from_pylist(units, schema=UNITS_SCHEMA)
    pq.write_table(table, part_path, compression="zstd", row_group_size=UNITS_ROW_GROUP_ROWS)
    return len(units)


def build_units(slim_paths: list[Path], out_path: Path, params: UnitParams, workers: int) -> int:
    """Pass 2, one worker per Knesset, parts joined in Knesset order."""
    by_knesset: dict[int, list[Path]] = {}
    for path in slim_paths:
        for knesset in pc.unique(pq.read_table(path, columns=["knesset"]).column("knesset")).to_pylist():
            by_knesset.setdefault(knesset, []).append(path)
    parts_dir = out_path.parent / "_parts"
    shutil.rmtree(parts_dir, ignore_errors=True)
    parts_dir.mkdir(parents=True)
    tasks = [(k, by_knesset[k], parts_dir / f"{k:02d}.parquet", params) for k in sorted(by_knesset)]
    tmp_path = out_path.with_suffix(".parquet.tmp")
    n_units = 0
    with ProcessPoolExecutor(max_workers=min(workers, len(tasks))) as pool, \
            pq.ParquetWriter(tmp_path, UNITS_SCHEMA, compression="zstd") as writer:
        for (knesset, _, part_path, _), n in zip(tasks, pool.map(_build_task, tasks)):
            print(f"  Knesset {knesset}: {n:,} units", flush=True)
            n_units += n
            part = pq.ParquetFile(part_path)
            for group in range(part.num_row_groups):
                writer.write_table(part.read_row_group(group))
    tmp_path.replace(out_path)
    shutil.rmtree(parts_dir)
    return n_units


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="knesset-ches corpus", description="Build units.parquet from the corpus.")
    parser.add_argument("--protocol-type", choices=list(OUT_DIRS), default="plenary")
    parser.add_argument("--out-dir", type=Path, help="default: data/corpus/processed or processed_committee")
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--workers", type=int, default=max(1, min(12, (os.cpu_count() or 2) - 2)))
    args = parser.parse_args(argv)
    out_dir = args.out_dir or OUT_DIRS[args.protocol_type]
    sentences_dir = out_dir / "_sentences" / args.protocol_type
    slim_paths = extract_all(args.raw_dir, sentences_dir, args.protocol_type, args.workers)
    params = UnitParams.from_settings(load_settings())
    n_units = build_units(slim_paths, out_dir / "units.parquet", params, args.workers)
    print(f"wrote {n_units:,} units to {out_dir / 'units.parquet'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
