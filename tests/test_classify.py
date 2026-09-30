"""classify.py with the mock client, offline."""

from __future__ import annotations

import dataclasses
import datetime as dt
import re
import socket
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import yaml

from knesset_ches import classify
from knesset_ches.jev import MockJevClient
from knesset_ches.questions import ROOT, SETTINGS_PATH, Question, build_questions, load_dimensions, load_settings

SITTINGS = [(20, 2016), (24, 2021), (24, 2022), (25, 2022), (25, 2023)]   # knesset, year: 8 protocols each
SPEECHES = 6       # per protocol
N_QUESTIONS = 50
TO_ASK = 240       # 242 units less the chair and one too long to send


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(socket.socket, "connect", lambda *a, **k: pytest.fail("a test opened a network connection"))


def unit_row(unit_id: str, knesset: int, date: dt.date, text: str, chair: bool = False) -> dict[str, Any]:
    return {"unit_id": unit_id, "protocol_type": "plenary", "knesset": knesset, "date": date, "year": date.year,
            "faction_id": "31", "is_chairman": chair, "n_chars": len(text), "text": text}


@pytest.fixture(scope="module")
def units_frame() -> pd.DataFrame:
    rows = []
    for knesset, year in SITTINGS:
        for p in range(8):
            for s in range(SPEECHES):
                n = len(rows) + 1
                text = f"גברתי היושבת-ראש, זהו נאום מספר {n} על תקציב המדינה ועל חינוך. " * (3 + n % 4)
                rows.append(unit_row(f"{knesset}_ptm_{year}{p}|{s}|0", knesset, dt.date(year, 1 + p, 10 + p), text))
    rows.append(unit_row("24_ptm_chair|1|0", 24, dt.date(2021, 6, 1), "אנא סיים. תודה רבה. ", chair=True))
    rows.append(unit_row("25_ptm_huge|1|0", 25, dt.date(2023, 3, 1), "מילה " * 14000))
    return pd.DataFrame(rows)


@pytest.fixture()
def units_path(tmp_path: Path, units_frame: pd.DataFrame) -> Path:
    units_frame.to_parquet(tmp_path / "units.parquet", index=False)
    return tmp_path / "units.parquet"


def battery(units: Path, stage: str = "unit", dimensions: str = "dimensions.yaml", only: list | None = None) -> dict:
    entry = {"stage": stage, "units": str(units), "dimensions": str(ROOT / "config" / dimensions),
             "protocol_types": ["plenary"]}
    return {**entry, "only": only} if only else entry


def write_settings(tmp_path: Path, batteries: list[dict]) -> Path:
    settings = {**yaml.safe_load(SETTINGS_PATH.read_text(encoding="utf-8")), "prompts_approved": False,
                "batteries": batteries}
    path = tmp_path / "settings.yaml"
    path.write_text(yaml.safe_dump(settings, allow_unicode=True), encoding="utf-8")
    return path


def run(tmp_path: Path, settings: Path, *argv: str, db: str = "results.mock.sqlite") -> int:
    return classify.main([*argv, "--settings", str(settings), "--db", str(tmp_path / db)])


def query(tmp_path: Path, sql: str, db: str = "results.mock.sqlite") -> list[tuple]:
    with closing(sqlite3.connect(tmp_path / db)) as conn:
        return conn.execute(sql).fetchall()


class RecordingMock(MockJevClient):
    calls: list[tuple[dict, dict]] = []

    async def decide(self, state: dict[str, Any], questions: dict[str, dict[str, Any]]) -> dict[str, Any]:
        RecordingMock.calls.append((state, questions))
        return await super().decide(state, questions)


@pytest.fixture()
def recorded(monkeypatch: pytest.MonkeyPatch) -> list[tuple[dict, dict]]:
    RecordingMock.calls = []
    monkeypatch.setattr(classify, "MockJevClient", RecordingMock)
    return RecordingMock.calls


def test_dry_run_is_the_default_and_writes_nothing(tmp_path: Path, units_path: Path, recorded: list,
                                                   capsys: pytest.CaptureFixture[str]) -> None:
    assert run(tmp_path, write_settings(tmp_path, [battery(units_path)])) == 0
    out = capsys.readouterr().out
    assert "mode: dry-run" in out and "Dry run: nothing was sent" in out and "£" in out
    assert f"241 units: 0 already answered, 1 too long to send, {TO_ASK} to ask" in out and "25_ptm_huge|1|0" in out
    assert recorded == [] and list(tmp_path.glob("*.sqlite*")) == []


def test_live_is_refused_while_the_prompts_are_not_approved(tmp_path: Path, units_path: Path,
                                                           monkeypatch: pytest.MonkeyPatch,
                                                           capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setenv(load_settings()["jev"]["api_key_env"], "sk-test")
    monkeypatch.setattr(classify, "JevClient", lambda *a, **k: pytest.fail("JevClient must not be built"))
    settings = write_settings(tmp_path, [battery(units_path)])
    assert run(tmp_path, settings, "--live", "--yes", db="results.sqlite") == 2
    assert "prompts_approved" in capsys.readouterr().err and list(tmp_path.glob("*.sqlite*")) == []


def test_mock_results_never_go_into_a_live_file(tmp_path: Path, units_path: Path,
                                                capsys: pytest.CaptureFixture[str]) -> None:
    settings = write_settings(tmp_path, [battery(units_path)])
    classify.ResultsDB(tmp_path / "live.sqlite", "live").close()
    assert run(tmp_path, settings, "--mock", "--limit", "2", db="live.sqlite") == 2
    assert query(tmp_path, "SELECT COUNT(*) FROM answers", db="live.sqlite") == [(0,)]
    existed = classify.LIVE_DB_PATH.exists()
    assert classify.main(["--mock", "--settings", str(settings), "--db", str(classify.LIVE_DB_PATH)]) == 2
    assert classify.LIVE_DB_PATH.exists() == existed and "refusing to write mock" in capsys.readouterr().err


def test_a_mock_run_stores_every_answer_and_a_rerun_asks_nothing(tmp_path: Path, units_path: Path, recorded: list,
                                                                 capsys: pytest.CaptureFixture[str]) -> None:
    settings = write_settings(tmp_path, [battery(units_path)])
    assert run(tmp_path, settings, "--mock") == 0
    # ukraine_support is only asked from 24 February 2022
    early = 48 + 48 + 2 * 2 * SPEECHES
    assert len(recorded) == TO_ASK and all(set(state) == {"text"} for state, _ in recorded)
    assert query(tmp_path, "SELECT COUNT(*) FROM answers") == [(TO_ASK * N_QUESTIONS - 2 * early,)]
    recorded.clear()
    assert run(tmp_path, settings, "--mock") == 0
    assert recorded == [] and f"{TO_ASK} already answered" in capsys.readouterr().out


def test_done_counts_only_the_current_prompt_model_and_stage(tmp_path: Path) -> None:
    questions = [Question(f"q{i}", f"d{i}", "gate", {"type": "noul"}, f"hash{i}") for i in range(130)]
    db = classify.ResultsDB(tmp_path / "many.sqlite", "mock")
    answered = [0, 59, 60, 61, 125, 129]   # more bits than one SQLite integer holds
    rows = [("u", f"q{i}", f"hash{i}", "m", "unit") for i in answered]
    rows += [("u", "q3", "an old prompt", "m", "unit"), ("u", "q4", "hash4", "another model", "unit"),
             ("u", "q5", "hash5", "m", "another stage")]
    db.conn.executemany("INSERT INTO answers VALUES (?,?,?,?,?,'noul',NULL,NULL,0.5,NULL,'{}','now')", rows)
    db.close()
    with closing(classify.open_readonly(tmp_path / "many.sqlite")) as conn:
        assert classify.load_done(conn, questions, "m", "unit") == {"u": sum(1 << i for i in answered)}


def test_invalid_answers_are_logged_not_stored_and_asked_again(tmp_path: Path, units_path: Path, recorded: list,
                                                               monkeypatch: pytest.MonkeyPatch) -> None:
    class Corrupting(MockJevClient):
        async def decide(self, state: dict[str, Any], questions: dict[str, dict[str, Any]]) -> dict[str, Any]:
            body = await super().decide(state, questions)
            body["answers"]["lrgen__score"]["probabilities"] = {"0": 0.9, "1": 0.9}
            body["answers"]["galtan__gate"] = {"type": "score", "score": 1.0}
            return body

    settings = write_settings(tmp_path, [battery(units_path, only=["lrgen", "galtan"])])
    monkeypatch.setattr(classify, "MockJevClient", Corrupting)
    assert run(tmp_path, settings, "--mock", "--knessets", "20", "--limit", "1") == 0
    assert sorted(query(tmp_path, "SELECT qid FROM answers")) == [("galtan__score",), ("lrgen__gate",)]
    ((error,),) = query(tmp_path, "SELECT error FROM requests")
    assert "lrgen__score" in error and "galtan__gate" in error and "sum to" in error
    monkeypatch.setattr(classify, "MockJevClient", RecordingMock)
    assert run(tmp_path, settings, "--mock", "--knessets", "20", "--limit", "1") == 0
    assert [sorted(questions) for _, questions in recorded] == [["galtan__gate", "lrgen__score"]]


def test_one_pass_over_all_batteries_stores_what_separate_passes_would(tmp_path: Path, units_frame: pd.DataFrame,
                                                                      recorded: list,
                                                                      capsys: pytest.CaptureFixture[str]) -> None:
    """Bibi items on every speech, heckle items on the 2023 ones, and one 2023 text under a second id."""
    speeches = units_frame[units_frame["unit_id"] != "25_ptm_huge|1|0"]
    in_2023 = speeches[speeches["year"] == 2023]
    pd.concat([speeches, in_2023.head(1).assign(unit_id="25_ptm_copy|1|0")]).to_parquet(tmp_path / "s.parquet")
    in_2023.to_parquet(tmp_path / "h.parquet")
    settings = write_settings(tmp_path, [
        battery(tmp_path / "s.parquet", "bibi", "dimensions_bibi.yaml"),
        battery(tmp_path / "h.parquet", "heckle", "dimensions_heckles.yaml", ["heckle_hostile", "heckle_personal"]),
    ])
    assert run(tmp_path, settings, "--mock", db="all.sqlite") == 0
    out = capsys.readouterr().out
    assert re.search(r"text sent by another battery +48\n", out) and re.search(r"text repeated in a battery +1\n", out)
    # 240 distinct texts, and the heckle questions go in the same requests
    assert len(recorded) == 240 and sorted({len(questions) for _, questions in recorded}) == [3, 5]

    for stage in ("bibi", "heckle"):
        assert run(tmp_path, settings, "--mock", "--stages", stage, db="apart.sqlite") == 0
    answers = ("SELECT unit_id, qid, prompt_hash, stage, score, noul, probabilities FROM answers"
               " ORDER BY unit_id, qid, stage")
    together = query(tmp_path, answers, db="all.sqlite")
    assert together == query(tmp_path, answers, db="apart.sqlite") and len(together) == 241 * 3 + 48 * 2
    assert query(tmp_path, "SELECT COUNT(*) FROM requests", db="all.sqlite") == [(240,)]
    assert query(tmp_path, "SELECT COUNT(*) FROM requests", db="apart.sqlite") == [(240 + 48,)]

    recorded.clear()
    capsys.readouterr()
    assert run(tmp_path, settings, "--mock", db="all.sqlite") == 0
    assert recorded == [] and re.search(r"units to ask +0\n", capsys.readouterr().out)


def test_a_qid_asked_with_two_payloads_goes_into_separate_requests() -> None:
    questions = build_questions(load_dimensions(ROOT / "config" / "dimensions_bibi.yaml"))
    changed = dataclasses.replace(questions[0], payload={**questions[0].payload, "instructions": "changed"},
                                  prompt_hash="changed")
    batteries = [classify.Battery("bibi", "a", Path("a"), [], questions, {}, pending={"u": 0b111}),
                 classify.Battery("other", "b", Path("b"), [], [changed], {}, pending={"u": 0b1})]
    first, second = classify.Merged(batteries, load_settings()).requests([(0, "u"), (1, "u")])
    assert first[:2] == (tuple(questions), (("u", "bibi", frozenset(q.qid for q in questions)),))
    assert second[:2] == ((changed,), (("u", "other", frozenset({changed.qid})),))
