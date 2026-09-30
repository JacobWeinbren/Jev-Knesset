"""knesset_ches.corpus on small synthetic protocols."""

from __future__ import annotations

import bz2
import datetime as dt
import json
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from knesset_ches import corpus
from knesset_ches.corpus import UnitParams

PROTOCOL = "25_ptm_000001.doc"
PARAMS = UnitParams(mk_only=True, min_words=60, max_chars=8000, max_interruption_words=25)


def words(n: int, token: str = "מילה") -> str:
    return " ".join([token] * n) + "."


def row(turn: int, sent: int, speaker: str, text: str, **overrides: Any) -> dict[str, Any]:
    return {"protocol_name": PROTOCOL, "protocol_type": "plenary", "knesset": 25, "date": dt.date(2023, 1, 30),
            "turn": turn, "sent": sent, "speaker_id": speaker, "speaker_name": speaker, "is_valid_speaker": True,
            "is_chairman": False, "is_mk": True, "faction_id": "31", "faction_name": None,
            "faction_general_name": None, "coalition": None, "is_ocr": False, "text": text, **overrides}


def build(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return corpus.protocol_units(rows, PARAMS)


def spans(units: list[dict[str, Any]]) -> list[tuple]:
    return [(u["speaker_id"], u["first_turn"], u["last_turn"], u["n_words"]) for u in units]


def test_a_speech_runs_on_across_short_interruptions_and_drops_them():
    rows = [
        row(0, 0, "A", "  " + words(40, "ראשון") + "   "),
        row(0, 0, "A", words(40, "ראשון")),                  # repeated by the next shard
        row(0, 1, "A", "   "),
        row(1, 0, "B", words(5, "הפרעה")),
        row(2, 0, "A", words(30, "שני")),
        row(2, 1, "A", words(10, "שלישי")),
        row(3, 0, "C", words(24, "קריאה")),                  # just under 25 words
        row(4, 0, "A", words(20, "רביעי")),
    ]
    [unit] = build(rows)
    assert (unit["first_turn"], unit["last_turn"], unit["n_turns"], unit["n_sentences"], unit["n_words"]) == (
        0, 4, 3, 4, 100)
    assert unit["text"] == "\n".join([words(40, "ראשון"), words(30, "שני") + " " + words(10, "שלישי"),
                                      words(20, "רביעי")])


def test_long_interruptions_end_a_speech_and_short_or_non_member_speech_is_dropped():
    rows = [row(0, 0, "A", words(70)), row(1, 0, "B", words(25, "ארוך")), row(2, 0, "A", words(80))]
    assert spans(build(rows)) == [("A", 0, 0, 70), ("A", 2, 2, 80)]          # B is under min_words
    rows = [row(turn, 0, "A" if turn % 2 == 0 else "B", words(20, f"t{turn}")) for turn in range(8)]
    assert spans(build(rows)) == [("A", 0, 6, 80), ("B", 1, 7, 80)]
    rows = [row(0, 0, "MK", words(100)), row(1, 0, "OFFICIAL", words(100), is_mk=False),
            row(2, 0, "MINISTER", words(100), faction_id=None), row(3, 0, "HECKLE", words(100), is_valid_speaker=False)]
    assert spans(build(rows)) == [("MK", 0, 0, 100)]


def chaired_sitting() -> list[dict[str, Any]]:
    """The flagged Speaker opens, the unflagged deputy DEP takes over and later speaks from the floor."""
    floor_speech = "אדוני היושב-ראש, כנסת נכבדה, " + words(80, "עמדה")
    return [
        row(0, 0, "SPEAKER", "חברי הכנסת, אני מתכבד לפתוח את ישיבת הכנסת. " + words(70, "הודעה"), is_chairman=True),
        row(1, 0, "A", floor_speech),
        row(2, 0, "SPEAKER", "תודה לחבר הכנסת. חבר הכנסת בי, בבקשה.", is_chairman=True),
        row(3, 0, "B", floor_speech),
        row(4, 0, "DEP", "תודה רבה לחבר הכנסת בי. חבר הכנסת סי, בבקשה."),
        row(5, 0, "C", floor_speech),
        row(6, 0, "H", "בבקשה, תענה לשאלה!"),                                     # a heckler's stray "please"
        row(7, 0, "DEP", "חבר הכנסת, נא לשבת. אני קורא אותך לסדר."),
        row(8, 0, "C", words(30, "המשך")),
        row(9, 0, "DEP", "תודה. אנחנו עוברים להצבעה. מי בעד? מי נגד? " + words(70, "סעיף")),
        row(10, 0, "DEP", "אני קובע שהצעת החוק התקבלה. רשות הדיבור לחבר הכנסת די."),
        row(11, 0, "D", floor_speech),
        row(12, 0, "SPEAKER", "תודה. חבר הכנסת דפ, בבקשה.", is_chairman=True),
        row(13, 0, "DEP", floor_speech),
        row(14, 0, "H", "לא נכון!"),
        row(15, 0, "DEP", words(40, "טיעון")),
        row(16, 0, "SPEAKER", "תודה לחבר הכנסת דפ. ישיבה זו נעולה.", is_chairman=True),
    ]


def chair_spans(units: list[dict[str, Any]], speaker: str) -> dict[tuple[int, int], bool]:
    return {(u["first_turn"], u["last_turn"]): u["is_chairman"] for u in units if u["speaker_id"] == speaker}


def test_a_presiding_deputy_is_detected_and_floor_speech_is_not():
    units = build(chaired_sitting())
    assert chair_spans(units, "SPEAKER") == {(0, 0): True}
    assert chair_spans(units, "DEP") == {(9, 10): True, (13, 15): False}
    assert not any(u["is_chairman"] for u in units if u["speaker_id"] in {"A", "B", "C", "D", "H"})
    committee = build([dict(r, protocol_type="committee") for r in chaired_sitting()])   # the corpus flag alone
    assert {u["speaker_id"] for u in committee if u["is_chairman"]} == {"SPEAKER"}


def test_the_flagged_speaker_on_the_floor_and_the_deputy_back_in_the_chair():
    rows = chaired_sitting()[:12] + [
        row(12, 0, "DEP", "תודה. חבר הכנסת, בבקשה."),
        row(13, 0, "SPEAKER", "גברתי היושבת-ראש, כנסת נכבדה, " + words(90, "נאום"), is_chairman=True),
        row(14, 0, "DEP", "תודה ליושב-ראש הכנסת. ישיבה זו נעולה."),
    ]
    assert chair_spans(build(rows), "SPEAKER")[(13, 13)] is False
    rows = chaired_sitting()[:16] + [
        row(16, 0, "DEP", "תודה. חבר הכנסת אי, בבקשה."),
        row(17, 0, "E", "אדוני היושב-ראש, כנסת נכבדה, " + words(80, "עמדה")),
        row(18, 0, "DEP", "תודה לחבר הכנסת אי. אנחנו עוברים להצבעה. מי בעד? מי נגד? " + words(70, "סעיף")),
        row(19, 0, "DEP", "אני קובע שההצעה התקבלה. ישיבה זו נעולה."),
    ]
    assert chair_spans(build(rows), "DEP") == {(9, 10): True, (13, 15): False, (18, 19): True}


def test_a_floor_speaker_saying_please_to_hecklers_is_not_presiding():
    rows = [row(0, 0, "SPEAKER", "אני מתכבד לפתוח את ישיבת הכנסת. חבר הכנסת איי, בבקשה.", is_chairman=True),
            row(1, 0, "A", words(50, "פתיחה"))]
    for k in range(4):
        rows += [row(2 + 3 * k, 0, "H", "– – –"), row(3 + 3 * k, 0, "A", "אל תפריעי לי, בבקשה."),
                 row(4 + 3 * k, 0, "A", words(20, f"טיעון{k}"))]
    assert chair_spans(build(rows), "A") == {(1, 13): False}


def raw_record(protocol: str, turn: int, sent: int, speaker: str, text: str, date: str = "2023-01-30 00:00",
               **overrides: Any) -> dict[str, Any]:
    return {"knesset_number": protocol[:2], "protocol_name": protocol, "protocol_type": "plenary",
            "protocol_date": date, "speaker_id": speaker, "is_valid_speaker": True, "speaker_is_knesset_member": True,
            "turn_num_in_protocol": turn, "sent_num_in_turn": sent, "faction_id": "31", "sentence_text": text,
            **overrides}


@pytest.fixture()
def tiny_corpus(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Two shards, with 24_ptm_2 split between them mid-turn."""
    may, june = "2022-05-02 00:00", "2022-06-01 00:00"
    old_type = {"speaker_is_knesset_member": "True"}                     # a string in the older shards
    shard0 = [raw_record("24_ptm_1.doc", 0, s, "A", words(10, "אלף"), may, **old_type) for s in range(3)]
    shard0 += [raw_record("24_ptm_2.doc", 0, s, "B", words(10, "בית"), june) for s in range(2)]
    shard1 = [raw_record("24_ptm_2.doc", 0, s, "B", words(10, "בית"), june) for s in range(2, 4)]
    shard1 += [raw_record("24_ptm_2.doc", 1, 0, "C", words(3, "הפרעה"), june),
               raw_record("24_ptm_2.doc", 2, 0, "B", words(10, "המשך"), june)]
    shard1 += [raw_record("25_ptm_3.doc", 0, s, "A", words(10, "גימל")) for s in range(3)]
    folder = tmp_path / "raw" / corpus.SHARD_DIR_TEMPLATE.format(protocol_type="plenary")
    folder.mkdir(parents=True)
    for number, records in enumerate([shard0, shard1]):
        with bz2.open(folder / f"plenary_shard_{number:02d}.jsonl.bz2", "wt", encoding="utf-8") as f:
            f.writelines(json.dumps(record, ensure_ascii=False) + "\n" for record in records)
    thresholds = {"mk_only": True, "min_words": 20, "max_chars": 8000, "max_interruption_words": 5}
    monkeypatch.setattr(corpus, "load_settings", lambda: {"corpus": thresholds})
    return tmp_path


def test_the_build_regroups_a_protocol_split_across_shards_and_reads_extension_shards(tiny_corpus: Path, capsys):
    def build_units() -> pa.Table:
        argv = ["--raw-dir", str(tiny_corpus / "raw"), "--out-dir", str(tiny_corpus / "out"), "--workers", "2"]
        assert corpus.main(argv) == 0
        return pq.read_table(tiny_corpus / "out" / "units.parquet")

    units = build_units()
    assert units.schema.equals(corpus.UNITS_SCHEMA)
    frame = units.to_pandas().set_index("unit_id")
    assert frame.index.tolist() == ["24_ptm_1.doc|0|0", "24_ptm_2.doc|0|0", "25_ptm_3.doc|0|0"]
    assert tuple(frame.loc["24_ptm_2.doc|0|0", ["n_words", "n_sentences", "n_turns"]]) == (50, 5, 2)
    assert frame.loc["24_ptm_2.doc|0|0", "text"] == " ".join([words(10, "בית")] * 4) + "\n" + words(10, "המשך")

    # extend writes sentence shards with no raw shard behind them
    slim = tiny_corpus / "out" / "_sentences" / "plenary"
    later = pq.read_table(slim / "shard_01.parquet").to_pandas()
    later = later[later["protocol_name"] == "25_ptm_3.doc"].assign(protocol_name="25_ptm_9.doc",
                                                                   date=dt.date(2024, 5, 1), shard=90)
    pq.write_table(pa.Table.from_pandas(later, schema=corpus.SENTENCE_SCHEMA, preserve_index=False),
                   slim / "shard_90.parquet")
    capsys.readouterr()
    assert build_units().column("unit_id").to_pylist()[-1] == "25_ptm_9.doc|0|0"
    assert "2 shards, 0 to do; read as they are: shard_90.parquet" in capsys.readouterr().out
