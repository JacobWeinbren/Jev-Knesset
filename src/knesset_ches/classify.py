"""Ask Jev each battery's questions about its texts. Resumable, and a dry run unless told otherwise.

    python -m knesset_ches.classify [--dry-run | --mock | --live] [--stages STAGE ...]
                                    [--knessets K ...] [--years Y ...] [--sample N] [--limit N]

--dry-run, the default, prints what is left to ask and the cost. --mock writes made-up answers to results.mock.sqlite.
--live needs `prompts_approved: true`, the API key and a typed confirmation (or --yes).
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import signal
import sqlite3
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Collection, Iterable, Iterator

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm

from knesset_ches.jev import (
    JevAuthError,
    JevClient,
    JevCreditsError,
    JevError,
    LiveCallsNotApproved,
    MockJevClient,
    live_gate_problems,
    parse_response,
)
from knesset_ches.questions import (
    ROOT,
    SETTINGS_PATH,
    Question,
    build_questions,
    load_dimensions,
    load_settings,
    questions_payload,
)

LIVE_DB_PATH = ROOT / "data" / "jev" / "results.sqlite"
MOCK_DB_PATH = ROOT / "data" / "jev" / "results.mock.sqlite"
UNIT_COLUMNS = ["unit_id", "protocol_type", "knesset", "date", "year", "faction_id", "is_chairman", "n_chars"]
REQUEST_SECONDS = 4.0
MAX_FAILURES_IN_A_ROW = 25
COMMIT_EVERY_REQUESTS = 100
COMMIT_EVERY_SECONDS = 2.0
MASK_BITS = 60                 # SQLite sums the masks in 64-bit integers

SCHEMA = """
CREATE TABLE IF NOT EXISTS answers (
  unit_id TEXT NOT NULL, qid TEXT NOT NULL, prompt_hash TEXT NOT NULL, model TEXT NOT NULL,
  stage TEXT NOT NULL DEFAULT 'unit', type TEXT NOT NULL, score REAL, confidence REAL, noul REAL,
  probabilities TEXT, raw TEXT NOT NULL, created_at TEXT NOT NULL,
  PRIMARY KEY (unit_id, qid, prompt_hash, model, stage)
);
CREATE TABLE IF NOT EXISTS requests (
  id INTEGER PRIMARY KEY AUTOINCREMENT, unit_id TEXT, n_questions INTEGER, status INTEGER,
  input_tokens INTEGER, output_tokens INTEGER, cost_usd REAL, latency_ms REAL, error TEXT, created_at TEXT
);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""

Target = tuple[str, str, frozenset[str]]   # (unit_id, stage, qids it needs from the request)


@dataclass
class Battery:

    stage: str
    title: str
    units_path: Path
    protocol_types: list[str]
    questions: list[Question]
    valid_from: dict[str, pd.Timestamp]
    pending: dict[str, int] = field(default_factory=dict)            # unit_id: bitmask of questions to ask
    n_units: int = 0
    n_done: int = 0
    too_long: list[tuple[str, int]] = field(default_factory=list)    # (unit_id, estimated tokens)


@dataclass(frozen=True)
class WorkItem:
    """One request: a text, its questions and where the answers go."""

    unit_id: str
    text: str
    questions: tuple[Question, ...]
    targets: tuple[Target, ...]


@dataclass(frozen=True)
class Outcome:
    item: WorkItem
    status: int | None
    latency_ms: float
    body: dict[str, Any] | None
    error: str | None
    fatal: bool = False


@dataclass
class RunStats:
    requests_ok: int = 0
    requests_failed: int = 0
    answers_stored: int = 0
    answers_invalid: int = 0
    input_tokens: int = 0
    cost_usd: float = 0.0
    requests_without_cost: int = 0
    elapsed_s: float = 0.0
    interrupted: bool = False
    fatal: str | None = None


class ResultsDB:
    """Writes a results file. `kind`, live or mock, must match the file's."""

    def __init__(self, path: Path, kind: str):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path, timeout=60)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.executescript(SCHEMA)
        self.conn.execute("INSERT OR IGNORE INTO meta (key, value) VALUES ('kind', ?)", (kind,))
        self.conn.commit()
        (stored,) = self.conn.execute("SELECT value FROM meta WHERE key = 'kind'").fetchone()
        if stored != kind:
            self.conn.close()
            raise ValueError(f"{path} holds {stored} results; refusing to write {kind} results into it")

    def record(self, outcome: Outcome, model: str, stats: RunStats) -> None:
        item, error, now = outcome.item, outcome.error, datetime.now(timezone.utc).isoformat(timespec="seconds")
        input_tokens = output_tokens = cost = None
        if outcome.body is None:
            stats.requests_failed += 1
        else:
            parsed = parse_response(outcome.body, questions_payload(item.questions))
            asked = {q.qid: q for q in item.questions}
            rows = [answer_row(unit_id, stage, asked[qid], answer, model, now)
                    for unit_id, stage, qids in item.targets
                    for qid, answer in parsed.answers.items() if qid in qids]
            self.conn.executemany("INSERT OR IGNORE INTO answers VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", rows)
            if parsed.invalid:
                error = "invalid answers (not stored): " + json.dumps(parsed.invalid, ensure_ascii=False)
            input_tokens, output_tokens, cost = parsed.input_tokens, parsed.output_tokens, parsed.cost_usd
            stats.requests_ok += 1
            stats.answers_stored += len(rows)
            stats.answers_invalid += len(parsed.invalid)
            stats.input_tokens += input_tokens or 0
            stats.cost_usd += cost or 0.0
            stats.requests_without_cost += cost is None
        self.conn.execute(
            "INSERT INTO requests (unit_id, n_questions, status, input_tokens, output_tokens, cost_usd, latency_ms,"
            " error, created_at) VALUES (?,?,?,?,?,?,?,?,?)",
            (item.unit_id, len(item.questions), outcome.status, input_tokens, output_tokens, cost, outcome.latency_ms,
             error, now),
        )

    def commit(self) -> None:
        self.conn.commit()

    def close(self) -> None:
        self.conn.commit()
        self.conn.close()


def answer_row(unit_id: str, stage: str, question: Question, answer: dict[str, Any], model: str, now: str) -> tuple:
    probabilities = json.dumps(answer["probabilities"]) if "probabilities" in answer else None
    criteria = question.payload["criteria"]
    if isinstance(criteria, list) and answer.get("legend") == {str(i): text for i, text in enumerate(criteria)}:
        answer = {key: value for key, value in answer.items() if key != "legend"}   # it just repeats the levels
    return (unit_id, question.qid, question.prompt_hash, model, stage, question.payload["type"], answer.get("score"),
            answer.get("confidence"), answer.get("noul"), probabilities, json.dumps(answer, ensure_ascii=False), now)


def open_readonly(path: Path) -> sqlite3.Connection | None:
    """None when there are no results yet."""
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True) if path.exists() else None


def load_done(conn: sqlite3.Connection | None, questions: list[Question], model: str, stage: str) -> dict[str, int]:
    """By unit, a bitmask of the questions already answered."""
    done: dict[str, int] = {}
    if conn is None:
        return done
    for offset in range(0, len(questions), MASK_BITS):
        block = questions[offset : offset + MASK_BITS]
        parameters = [value for i, q in enumerate(block) for value in (q.qid, q.prompt_hash, 1 << i)]
        sql = (f"WITH asked(qid, prompt_hash, bit) AS (VALUES {','.join(['(?,?,?)'] * len(block))}) "
               "SELECT a.unit_id, SUM(asked.bit) FROM answers a "
               "JOIN asked ON a.qid = asked.qid AND a.prompt_hash = asked.prompt_hash "
               "WHERE a.model = ? AND a.stage = ? GROUP BY a.unit_id")
        for unit_id, mask in conn.execute(sql, [*parameters, model, stage]):
            done[unit_id] = done.get(unit_id, 0) | (mask << offset)
    return done


def load_batteries(settings: dict[str, Any], stages: Collection[str] | None) -> list[Battery]:
    batteries = []
    for entry in settings["batteries"]:
        if stages and entry["stage"] not in stages:
            continue
        dimensions = load_dimensions(ROOT / entry["dimensions"])
        only = entry.get("only")
        questions = [q for q in build_questions(dimensions) if not only or q.dimension in only]
        valid_from = {d.id: pd.Timestamp(d.valid_from) for d in dimensions if d.valid_from}
        title = f"{entry['stage']}: {entry['units']}" + (f", {len(only)} items" if only else "")
        batteries.append(Battery(entry["stage"], title, ROOT / entry["units"], entry["protocol_types"], questions,
                                 valid_from))
    return batteries


def select_units(battery: Battery, settings: dict[str, Any], knessets: list[int] | None, years: list[int] | None,
                 sample: int | None, limit: int | None) -> pd.DataFrame:
    units = pd.read_parquet(battery.units_path, columns=UNIT_COLUMNS)
    units["date"] = pd.to_datetime(units["date"])
    keep = units["protocol_type"].isin(battery.protocol_types)
    if settings["filters"]["exclude_chair_units"]:
        keep &= ~units["is_chairman"].astype(bool)
    if knessets:
        keep &= units["knesset"].isin(knessets)
    if years:
        keep &= units["year"].isin(years)
    units = units[keep]
    if sample is not None:
        units = stratified_sample(units, sample, settings["aggregation"]["seed"])
    if limit is not None:
        units = units.head(limit)
    return units.drop_duplicates("unit_id")   # the heckle samples hold some heckles twice


def stratified_sample(units: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """`n` units spread over year x faction strata in proportion, at least one per stratum."""
    if n >= len(units):
        return units
    rng = np.random.default_rng(seed)
    ordered = units.sort_values("unit_id")
    members = list(ordered.groupby(["year", "faction_id"], sort=True, dropna=False).indices.values())
    sizes = np.array([len(m) for m in members])
    if n < len(members):
        quota = np.zeros(len(members), dtype=int)
        quota[rng.choice(len(members), size=n, replace=False)] = 1
    else:
        quota = proportional_quota(sizes, n)
    picks = [rng.choice(m, size=q, replace=False) for m, q in zip(members, quota) if q]
    return ordered.iloc[np.sort(np.concatenate(picks))]


def proportional_quota(sizes: np.ndarray, n: int) -> np.ndarray:
    share = sizes * n / sizes.sum()
    quota = np.minimum(sizes, np.maximum(1, np.floor(share).astype(int)))
    while quota.sum() < n:
        quota[np.argmax(np.where(quota < sizes, share - quota, -np.inf))] += 1
    while quota.sum() > n:   # the minimum of one per stratum overshot
        quota[np.argmax(np.where(quota > 1, quota - share, -np.inf))] -= 1
    return quota


def applicable(units: pd.DataFrame, battery: Battery) -> list[int]:
    """By unit, a bitmask of the questions that apply, given each dimension's valid_from."""
    masks = [(1 << len(battery.questions)) - 1] * len(units)
    for dimension, first_day in battery.valid_from.items():
        bits = sum(1 << i for i, q in enumerate(battery.questions) if q.dimension == dimension)
        too_early = ~(units["date"] >= first_day).to_numpy()   # an unknown date counts as too early
        masks = [mask & ~bits if early else mask for mask, early in zip(masks, too_early)]
    return masks


def question_tokens(question: Question, settings: dict[str, Any]) -> int:
    chars = len(question.qid) + len(json.dumps(question.payload, ensure_ascii=False))
    return math.ceil(chars / settings["jev"]["est_chars_per_token_questions"])


def text_tokens(n_chars: int, settings: dict[str, Any]) -> int:
    return math.ceil(n_chars / settings["jev"]["est_chars_per_token_hebrew"])


def plan_battery(battery: Battery, units: pd.DataFrame, done: dict[str, int], settings: dict[str, Any]) -> None:
    """Fill in what is left to ask. Texts too long for the context are set aside."""
    jev = settings["jev"]
    tokens = [question_tokens(q, settings) for q in battery.questions]
    longest: dict[int, int] = {}
    for unit_id, n_chars, mask in zip(units["unit_id"], units["n_chars"], applicable(units, battery)):
        battery.n_units += 1
        mask &= ~done.get(unit_id, 0)
        if not mask:
            battery.n_done += 1
            continue
        if mask not in longest:
            longest[mask] = max(t for i, t in enumerate(tokens) if mask >> i & 1)
        size = text_tokens(n_chars, settings) + jev["est_request_overhead_tokens"] + longest[mask]
        if size > jev["context_limit_tokens"]:
            battery.too_long.append((unit_id, size))
        else:
            battery.pending[unit_id] = mask


def plan(settings: dict[str, Any], db_path: Path, stages: Collection[str] | None = None,
         knessets: list[int] | None = None, years: list[int] | None = None, sample: int | None = None,
         limit: int | None = None) -> list[Battery]:
    batteries = load_batteries(settings, stages)
    conn = open_readonly(db_path)
    try:
        for battery in batteries:
            units = select_units(battery, settings, knessets, years, sample, limit)
            done = load_done(conn, battery.questions, settings["jev"]["model"], battery.stage)
            plan_battery(battery, units, done, settings)
    finally:
        if conn is not None:
            conn.close()
    return batteries


def iter_texts(path: Path, wanted: Collection[str]) -> Iterator[tuple[str, str]]:
    remaining = set(wanted)
    for batch in pq.ParquetFile(path).iter_batches(batch_size=512, columns=["unit_id", "text"]):
        if not remaining:
            return
        texts = batch.column("text")
        for row, unit_id in enumerate(batch.column("unit_id").to_pylist()):
            if unit_id in remaining:
                remaining.discard(unit_id)
                yield unit_id, texts[row].as_py()


def split_requests(asked: dict[Any, list[Question]]) -> list[tuple[tuple[Question, ...], dict[Any, frozenset[str]]]]:
    """Pack the questions into as few requests as possible. A qid worded two ways needs two requests."""
    layers: list[dict[str, Question]] = []
    for question in (q for questions in asked.values() for q in questions):
        for layer in layers:
            if question.qid not in layer:
                layer[question.qid] = question
                break
            if layer[question.qid].prompt_hash == question.prompt_hash:
                break
        else:
            layers.append({question.qid: question})
    requests = []
    for layer in layers:
        sent = {(q.qid, q.prompt_hash) for q in layer.values()}
        needs = {key: frozenset(q.qid for q in qs if (q.qid, q.prompt_hash) in sent) for key, qs in asked.items()}
        requests.append((tuple(layer.values()), needs))
    return requests


class Merged:
    """All batteries' pending questions, one request per distinct text."""

    def __init__(self, batteries: list[Battery], settings: dict[str, Any]):
        self.batteries = batteries
        self.settings = settings
        self.sent: list[tuple[int, str]] = []                 # (battery, unit) that sends each distinct text
        self.others: dict[int, list[tuple[int, str]]] = {}    # other units with the same text
        self.n_shared = 0
        self.n_repeated = 0
        self.n_requests = self.n_questions = self.n_tokens = 0
        self._layouts: dict[tuple, list] = {}

    def requests(self, units: list[tuple[int, str]]) -> list[tuple[tuple[Question, ...], tuple[Target, ...], int]]:
        """Requests for one text: questions, answer targets and question tokens."""
        wants = [(i, unit_id, self.batteries[i].pending[unit_id]) for i, unit_id in units]
        key = tuple(sorted({(i, mask) for i, _, mask in wants}))
        if key not in self._layouts:
            asked = {(i, mask): [q for bit, q in enumerate(self.batteries[i].questions) if mask >> bit & 1]
                     for i, mask in key}
            self._layouts[key] = [(questions, needs, sum(question_tokens(q, self.settings) for q in questions))
                                  for questions, needs in split_requests(asked)]
        out = []
        for questions, needs, tokens in self._layouts[key]:
            targets = tuple((unit_id, self.batteries[i].stage, needs[i, mask])
                            for i, unit_id, mask in wants if needs[i, mask])
            out.append((questions, targets, tokens))
        return out


def merge(batteries: list[Battery], settings: dict[str, Any]) -> Merged:
    """Find shared and repeated texts and count the requests."""
    merged = Merged(batteries, settings)
    first: dict[bytes, int] = {}   # text hash: index in merged.sent
    chars: list[int] = []
    for i, battery in enumerate(batteries):
        for unit_id, text in iter_texts(battery.units_path, battery.pending):
            key = hashlib.blake2b(text.encode("utf-8"), digest_size=16).digest()
            if key not in first:
                first[key] = len(merged.sent)
                merged.sent.append((i, unit_id))
                chars.append(len(text))
                continue
            r = first[key]
            merged.others.setdefault(r, []).append((i, unit_id))
            if merged.sent[r][0] == i:
                merged.n_repeated += 1
            else:
                merged.n_shared += 1
    for r, unit in enumerate(merged.sent):
        text_and_overhead = text_tokens(chars[r], settings) + settings["jev"]["est_request_overhead_tokens"]
        for questions, _, tokens in merged.requests([unit, *merged.others.get(r, [])]):
            merged.n_requests += 1
            merged.n_questions += len(questions)
            merged.n_tokens += text_and_overhead + tokens
    return merged


def iter_items(merged: Merged) -> Iterator[WorkItem]:
    """Yield the requests, reading the texts again rather than keeping them all in memory."""
    for i, battery in enumerate(merged.batteries):
        sent_here = {unit_id: r for r, (b, unit_id) in enumerate(merged.sent) if b == i}
        for unit_id, text in iter_texts(battery.units_path, sent_here):
            for questions, targets, _ in merged.requests([(i, unit_id), *merged.others.get(sent_here[unit_id], [])]):
                yield WorkItem(unit_id, text, questions, targets)


async def send(client: JevClient | MockJevClient, item: WorkItem) -> Outcome:
    started = time.perf_counter()
    try:
        body = await client.decide({"text": item.text}, questions_payload(item.questions))
    except JevError as exc:
        fatal = isinstance(exc, (JevAuthError, JevCreditsError, LiveCallsNotApproved))
        latency_ms = (time.perf_counter() - started) * 1000
        return Outcome(item, getattr(exc, "status", None), latency_ms, None, str(exc), fatal)
    return Outcome(item, 200, (time.perf_counter() - started) * 1000, body, None)


async def execute(items: Iterable[WorkItem], total: int, client: JevClient | MockJevClient, db: ResultsDB, *,
                  model: str, concurrency: int, stop: asyncio.Event | None = None) -> RunStats:
    """Send `items`, at most `concurrency` at a time. Setting `stop` lets those in flight finish."""
    stop = stop or asyncio.Event()
    work: asyncio.Queue[WorkItem | None] = asyncio.Queue(maxsize=2 * concurrency)
    results: asyncio.Queue[Outcome | None] = asyncio.Queue()
    stats = RunStats()
    started = time.monotonic()

    async def produce() -> None:
        iterator = iter(items)
        while not stop.is_set():
            item = await asyncio.to_thread(next, iterator, None)   # parquet reads stay off the event loop
            if item is None:
                break
            await work.put(item)
        for _ in range(concurrency):
            await work.put(None)

    async def work_loop() -> None:
        while (item := await work.get()) is not None:
            if not stop.is_set():   # once stopped, drain without sending
                await results.put(await send(client, item))

    async def write() -> None:
        failures_in_a_row = uncommitted = 0
        last_commit = time.monotonic()
        with tqdm(total=total, unit="req", disable=None, smoothing=0.05) as bar:   # no bar off a terminal
            while (outcome := await results.get()) is not None:
                db.record(outcome, model, stats)
                bar.update(1)
                bar.set_postfix(failed=stats.requests_failed, invalid=stats.answers_invalid, refresh=False)
                failures_in_a_row = failures_in_a_row + 1 if outcome.body is None else 0
                if outcome.fatal or failures_in_a_row >= MAX_FAILURES_IN_A_ROW:
                    streak = f"{failures_in_a_row} requests failed in a row; last: {outcome.error}"
                    stats.fatal = outcome.error if outcome.fatal else streak
                    stop.set()
                uncommitted += 1
                if uncommitted >= COMMIT_EVERY_REQUESTS or time.monotonic() - last_commit >= COMMIT_EVERY_SECONDS:
                    db.commit()
                    uncommitted, last_commit = 0, time.monotonic()

    writer: asyncio.Task[None] | None = None
    try:
        async with asyncio.TaskGroup() as writing:
            writer = writing.create_task(write())
            async with asyncio.TaskGroup() as sending:
                sending.create_task(produce())
                for _ in range(concurrency):
                    sending.create_task(work_loop())
            await results.put(None)
    finally:
        writer_failed = writer is not None and writer.done() and not writer.cancelled() and writer.exception()
        while not writer_failed and not results.empty():   # keep what arrived before a crash
            if (outcome := results.get_nowait()) is not None:
                db.record(outcome, model, stats)
        db.commit()
    stats.interrupted = stop.is_set() and stats.fatal is None
    stats.elapsed_s = time.monotonic() - started
    return stats


async def send_all(merged: Merged, client: JevClient | MockJevClient, db: ResultsDB,
                   settings: dict[str, Any]) -> RunStats:
    """`execute` with Ctrl-C handling. The first stops new requests, a second aborts."""
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()

    def on_sigint() -> None:
        print("\nCtrl-C: no new requests; waiting for the ones in flight, then committing. Ctrl-C again aborts.",
              file=sys.stderr)
        stop.set()
        loop.remove_signal_handler(signal.SIGINT)

    loop.add_signal_handler(signal.SIGINT, on_sigint)
    try:
        return await execute(iter_items(merged), merged.n_requests, client, db, model=settings["jev"]["model"],
                             concurrency=settings["jev"]["concurrency"], stop=stop)
    finally:
        loop.remove_signal_handler(signal.SIGINT)
        await client.aclose()


def pounds(amount: float) -> str:
    return f"£{amount:,.2f}"


def duration(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.0f} s"
    if seconds < 5400:
        return f"{seconds / 60:.0f} min"
    if seconds < 48 * 3600:
        return f"{seconds / 3600:.1f} h"
    return f"{seconds / 86400:.1f} days"


def describe(batteries: list[Battery], merged: Merged, settings: dict[str, Any]) -> str:
    jev = settings["jev"]
    lines = []
    for b in batteries:
        lines.append(f"{b.title}\n  {b.n_units:,} units: {b.n_done:,} already answered, {len(b.too_long):,} too long"
                     f" to send, {len(b.pending):,} to ask")
        lines += [f"    skipped: {unit_id}, about {size:,} tokens against {jev['context_limit_tokens']:,}"
                  for unit_id, size in b.too_long]
    cost = merged.n_tokens * jev["price_per_m_input_tokens"] / 1e6
    eta = duration(merged.n_requests * REQUEST_SECONDS / jev["concurrency"])
    counts = [
        ("units to ask", sum(len(b.pending) for b in batteries)),
        ("text sent by another battery", merged.n_shared),
        ("text repeated in a battery", merged.n_repeated),
        ("requests", merged.n_requests),
        ("questions", merged.n_questions),
        ("input tokens", merged.n_tokens),
    ]
    lines.append("All batteries, one request per distinct text")
    lines += [f"  {label:<30}{count:>15,}" for label, count in counts]
    lines.append(f"  {'estimated cost':<30}{pounds(cost):>15}")
    lines.append(f"  {'ETA':<30}{eta:>15}  at {jev['concurrency']} requests at a time and {REQUEST_SECONDS:g} s each")
    return "\n".join(lines)


def format_stats(stats: RunStats, settings: dict[str, Any]) -> str:
    list_price = stats.input_tokens * settings["jev"]["price_per_m_input_tokens"] / 1e6
    reported = stats.cost_usd / settings["jev"]["usd_per_gbp"]
    lines = [
        f"{stats.requests_ok:,} requests answered and {stats.requests_failed:,} failed in {duration(stats.elapsed_s)};"
        f" {stats.answers_stored:,} answers stored, {stats.answers_invalid:,} invalid answers rejected",
        f"{stats.input_tokens:,} input tokens: {pounds(reported)} as reported by the API"
        f" ({stats.requests_without_cost:,} responses gave no cost), {pounds(list_price)} at the list price",
    ]
    if stats.requests_failed or stats.answers_invalid:
        lines.append("The failures are in the requests table (status, error); running again retries them.")
    if stats.interrupted:
        lines.append("Interrupted: everything received was committed; run again to resume.")
    if stats.fatal:
        lines.append(f"Stopped: {stats.fatal}")
    return "\n".join(lines)


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m knesset_ches.classify", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", dest="mode", action="store_const", const="dry-run", help="the default")
    mode.add_argument("--mock", dest="mode", action="store_const", const="mock")
    mode.add_argument("--live", dest="mode", action="store_const", const="live")
    parser.set_defaults(mode="dry-run")
    parser.add_argument("--stages", nargs="+", metavar="STAGE", help="only the batteries with these stages")
    parser.add_argument("--knessets", type=int, nargs="+", metavar="K")
    parser.add_argument("--years", type=int, nargs="+", metavar="Y")
    parser.add_argument("--sample", type=int, metavar="N", help="N units of each battery by year and faction")
    parser.add_argument("--limit", type=int, metavar="N", help="the first N units of each battery")
    parser.add_argument("--yes", action="store_true", help="with --live, no typed confirmation")
    parser.add_argument("--db", type=Path, help="the results file (default: data/jev/results.sqlite, or "
                        "results.mock.sqlite with --mock)")
    parser.add_argument("--settings", type=Path, default=SETTINGS_PATH)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    settings = load_settings(args.settings)
    db_path = args.db or (MOCK_DB_PATH if args.mode == "mock" else LIVE_DB_PATH)
    if args.mode == "mock" and db_path.resolve() == LIVE_DB_PATH.resolve():
        print(f"refusing to write mock results into {LIVE_DB_PATH}", file=sys.stderr)
        return 2
    batteries = plan(settings, db_path, args.stages, args.knessets, args.years, args.sample, args.limit)
    merged = merge(batteries, settings)
    print(f"mode: {args.mode}   model: {settings['jev']['model']}   results: {db_path}")
    print(describe(batteries, merged, settings))
    if args.mode == "dry-run":
        print("Dry run: nothing was sent and nothing was written.")
        return 0
    if args.mode == "live":
        problems = live_gate_problems(settings, live=True)
        if problems:
            print("Refusing to contact Jev:\n  - " + "\n  - ".join(problems), file=sys.stderr)
            return 2
        if not args.yes and input("Send these requests to Jev? Type 'yes' to continue: ").strip().lower() != "yes":
            print("Aborted; nothing was sent.")
            return 1
    try:
        db = ResultsDB(db_path, args.mode)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    client = JevClient(settings, live=True) if args.mode == "live" else MockJevClient()
    try:
        stats = asyncio.run(send_all(merged, client, db, settings))
    finally:
        db.close()
    print(format_stats(stats, settings))
    if stats.fatal:
        return 1
    return 130 if stats.interrupted else 0


if __name__ == "__main__":
    raise SystemExit(main())
