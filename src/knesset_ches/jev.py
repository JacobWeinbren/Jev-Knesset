"""The client for TypeSafe Jev (OpenRouter's Decisions API), an offline mock of it, and the response parser.

JevClient refuses to send unless the settings approve live calls, live=True is passed and the API key is set.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import os
import random
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

import httpx

log = logging.getLogger(__name__)

TOLERANCE = 1e-6
PROBABILITY_SUM_TOLERANCE = 0.05   # the API rounds probabilities
BODY_MAX_CHARS = 2000


class JevError(Exception):
    pass


class LiveCallsNotApproved(JevError):
    pass


class JevHTTPError(JevError):
    """A failed request. `status` is None when nothing came back."""

    def __init__(self, status: int | None, body: str, attempts: int = 1):
        self.status, self.body, self.attempts = status, body[:BODY_MAX_CHARS], attempts
        super().__init__(f"Jev request failed (HTTP {status}, {attempts} attempt(s)): {self.body}")


class JevAuthError(JevHTTPError):
    """401: the key was rejected."""


class JevCreditsError(JevHTTPError):
    """402: out of credits."""


ERRORS = {401: JevAuthError, 402: JevCreditsError}


def live_gate_problems(settings: dict[str, Any], live: bool) -> list[str]:
    """Why a live request is not allowed. Empty when it is."""
    problems = []
    if settings.get("prompts_approved") is not True:
        problems.append("`prompts_approved` is not true in config/settings.yaml (live runs are switched off)")
    if live is not True:
        problems.append("live=True was not passed (command-line flag --live)")
    key_env = settings["jev"]["api_key_env"]
    if not os.environ.get(key_env):
        problems.append(f"the environment variable {key_env} is not set")
    return problems


class JevClient:
    def __init__(self, settings: dict[str, Any], *, live: bool = False,
                 transport: httpx.AsyncBaseTransport | None = None,
                 sleep: Callable[[float], Awaitable[None]] = asyncio.sleep):
        self._settings = settings
        self._live = live
        self._api_key()   # refuse before the HTTP client exists
        jev = settings["jev"]
        self.endpoint, self.model, self.max_retries = jev["endpoint"], jev["model"], int(jev["max_retries"])
        self._sleep = sleep
        self._http = httpx.AsyncClient(transport=transport, timeout=httpx.Timeout(float(jev["timeout_s"])))

    def _api_key(self) -> str:
        problems = live_gate_problems(self._settings, self._live)
        if problems:
            raise LiveCallsNotApproved("live Jev calls are not allowed: " + "; ".join(problems))
        return os.environ[self._settings["jev"]["api_key_env"]]

    async def aclose(self) -> None:
        await self._http.aclose()

    async def decide(self, state: dict[str, Any], questions: dict[str, dict[str, Any]]) -> dict[str, Any]:
        request = {"model": self.model, "state": state, "questions": questions}
        content = json.dumps(request, ensure_ascii=False).encode("utf-8")
        attempts = self.max_retries + 1
        for attempt in range(attempts):
            headers = {"Authorization": f"Bearer {self._api_key()}", "Content-Type": "application/json"}
            retry_after = None
            try:
                response = await self._http.post(self.endpoint, content=content, headers=headers)
            except httpx.TransportError as exc:
                status, body = None, f"{type(exc).__name__}: {exc}"
            else:
                if response.is_success:
                    return _decode(response)
                status, body = response.status_code, response.text
                if status != 429 and status < 500:
                    raise ERRORS.get(status, JevHTTPError)(status, body, attempt + 1)
                retry_after = _seconds(response.headers.get("Retry-After"))
            if attempt + 1 < attempts:
                delay = _backoff(attempt, retry_after)
                log.warning("Jev attempt %d/%d failed (%s); retrying in %.1f s",
                            attempt + 1, attempts, status or body, delay)
                await self._sleep(delay)
        raise JevHTTPError(status, body, attempts)


def _decode(response: httpx.Response) -> dict[str, Any]:
    try:
        body = response.json()
    except ValueError:
        body = None
    if not isinstance(body, dict):
        raise JevError(f"the response is not a JSON object: {response.text[:BODY_MAX_CHARS]}")
    return body


def _backoff(attempt: int, retry_after: float | None) -> float:
    """Exponential backoff with jitter, capped at 60 s, but no shorter than Retry-After (up to 300 s)."""
    delay = min(60.0, 2.0**attempt) * random.uniform(0.5, 1.0)
    return delay if retry_after is None else max(delay, min(retry_after, 300.0))


def _seconds(header: str | None) -> float | None:
    try:
        return max(0.0, float(header))
    except (TypeError, ValueError):
        return None


@dataclass(frozen=True)
class ParsedResponse:
    answers: dict[str, dict[str, Any]]
    invalid: dict[str, str]              # qid: why it was rejected
    input_tokens: int | None
    output_tokens: int | None
    cost_usd: float | None


def parse_response(body: dict[str, Any], questions: dict[str, dict[str, Any]]) -> ParsedResponse:
    """Check each answer against its question. Answers and usage may sit at the top level or under `data`."""
    found = _find_object(body, "answers")
    if found is None and isinstance(body.get("data"), dict) and set(body["data"]) & set(questions):
        found = body["data"]
    answers, invalid = {}, {}
    for qid, payload in questions.items():
        answer = (found or {}).get(qid)
        if answer is None:
            invalid[qid] = "no answer in the response"
        elif (problem := validate_answer(answer, payload)) is None:
            answers[qid] = answer
        else:
            invalid[qid] = f"{problem}; raw={json.dumps(answer, ensure_ascii=False)[:500]}"
    usage = _find_object(body, "usage") or {}
    cost = _first_number(usage, ("cost", "cost_usd", "total_cost"))
    if cost is None:
        cost = _first_number(body, ("cost", "cost_usd"))
    input_tokens = _first_number(usage, ("input_tokens", "prompt_tokens"))
    output_tokens = _first_number(usage, ("output_tokens", "completion_tokens"))
    return ParsedResponse(answers, invalid, None if input_tokens is None else int(input_tokens),
                          None if output_tokens is None else int(output_tokens), cost)


def validate_answer(answer: Any, payload: dict[str, Any]) -> str | None:
    """None if `answer` is valid, else why not."""
    if not isinstance(answer, dict):
        return "answer is not an object"
    expected = payload["type"]
    if answer.get("type", expected) != expected:
        return f"answer type {answer.get('type')!r} does not match question type {expected!r}"
    if expected == "noul":
        return None if _in_range(answer.get("noul"), 1) else "`noul` is not a number in [0, 1]"
    top = len(payload["criteria"]) - 1
    if not _in_range(answer.get("score"), top):
        return f"`score` is not a number in [0, {top}]"
    if not _in_range(answer.get("confidence"), 1):
        return "`confidence` is not a number in [0, 1]"
    probabilities = answer.get("probabilities")
    if not isinstance(probabilities, dict) or not probabilities:
        return "`probabilities` is missing"
    for level, p in probabilities.items():
        if not str(level).isdigit() or int(level) > top:
            return f"`probabilities` has a level outside 0..{top}: {level!r}"
        if not _in_range(p, 1):
            return f"probability of level {level} is not a number in [0, 1]"
    total = sum(probabilities.values())
    if abs(total - 1.0) > PROBABILITY_SUM_TOLERANCE:
        return f"`probabilities` sum to {total:.4f}, not 1"
    return None


def _in_range(value: Any, top: float) -> bool:
    number = _number(value)
    return number is not None and -TOLERANCE <= number <= top + TOLERANCE


def _find_object(body: dict[str, Any], key: str) -> dict[str, Any] | None:
    for holder in (body, body.get("data")):
        if isinstance(holder, dict) and isinstance(holder.get(key), dict):
            return holder[key]
    return None


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    return float(value)


def _first_number(holder: dict[str, Any], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        number = _number(holder.get(key))
        if number is not None:
            return number
    return None


class MockJevClient:
    """Offline stand-in for Jev. Answers are fixed by the text and qid, and marked "mock": true."""

    def __init__(self, latency_s: float = 0.0):
        self.latency_s = latency_s
        self.n_calls = 0

    async def aclose(self) -> None:
        pass

    async def decide(self, state: dict[str, Any], questions: dict[str, dict[str, Any]]) -> dict[str, Any]:
        self.n_calls += 1
        await asyncio.sleep(self.latency_s)
        text = json.dumps(state, ensure_ascii=False, sort_keys=True)
        answers = {qid: _mock_answer(text, qid, payload) for qid, payload in questions.items()}
        tokens = len(text) + len(json.dumps(questions, ensure_ascii=False)) // 4
        return {"model": "mock/jev", "answers": answers,
                "usage": {"input_tokens": tokens, "output_tokens": 0, "cost": tokens * 0.042 / 1e6}}


def _mock_answer(text: str, qid: str, payload: dict[str, Any]) -> dict[str, Any]:
    if payload["type"] == "noul":
        return {"type": "noul", "noul": round(_draw(text, qid, "noul"), 4), "mock": True}
    levels = payload["criteria"]
    n = len(levels)
    mode = int(_draw(text, qid, "level") * n)
    confidence = _draw(text, qid, "confidence")
    top = (confidence * (n - 1) + 1) / n   # the modal probability that gives this confidence
    probabilities = [top if level == mode else (1 - top) / (n - 1) for level in range(n)]
    return {
        "type": "score",
        "score": sum(level * p for level, p in enumerate(probabilities)),
        "legend": {str(level): label for level, label in enumerate(levels)},
        "probabilities": {str(level): p for level, p in enumerate(probabilities)},
        "confidence": confidence,
        "mock": True,
    }


def _draw(text: str, qid: str, salt: str) -> float:
    digest = hashlib.sha256(f"{salt}\x1f{qid}\x1f{text}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2**64
