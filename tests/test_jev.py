"""jev.py offline: the live-call gate, retries and the response parser."""

from __future__ import annotations

import json
import socket
from typing import Any

import httpx
import pytest

from knesset_ches.jev import (JevAuthError, JevClient, JevCreditsError, JevError, JevHTTPError, LiveCallsNotApproved,
                              parse_response, validate_answer)

KEY_ENV = "KNESSET_CHES_TEST_KEY"
ENDPOINT = "https://jev.invalid/decisions"
SCORE_Q = {"type": "score", "instructions": "Where does the speaker stand?", "criteria": ["a", "b", "c", "d", "e"]}
NOUL_Q = {"type": "noul", "instructions": "Does the text express a view?", "criteria": {"true": "yes", "false": "no"}}
QUESTIONS = {"dim__gate": NOUL_Q, "dim__score": SCORE_Q}
GOOD_BODY = {
    "answers": {"dim__gate": {"type": "noul", "noul": 0.87},
                "dim__score": {"type": "score", "score": 1.05, "probabilities": {"1": 0.95, "2": 0.05},
                               "confidence": 0.92}},
    "usage": {"input_tokens": 1234, "output_tokens": 0},
}


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(socket.socket, "connect", lambda *a, **k: pytest.fail("a test opened a network connection"))


@pytest.fixture(autouse=True)
def key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(KEY_ENV, "sk-test")


def make_settings(approved: Any = True) -> dict[str, Any]:
    jev = {"endpoint": ENDPOINT, "model": "typesafe/jev-1.13", "api_key_env": KEY_ENV, "timeout_s": 5,
           "max_retries": 3}
    return {"prompts_approved": approved, "jev": jev}


class Wire:
    """A live client that replays canned responses and records the requests and delays."""

    def __init__(self, *responses: httpx.Response | Exception):
        self.responses, self.requests, self.delays = list(responses), [], []
        self.settings = make_settings()
        self.client = JevClient(self.settings, live=True, transport=httpx.MockTransport(self.handle), sleep=self.sleep)

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    async def sleep(self, seconds: float) -> None:
        self.delays.append(seconds)

    async def decide(self) -> dict[str, Any]:
        try:
            return await self.client.decide({"text": "שלום עולם"}, QUESTIONS)
        finally:
            await self.client.aclose()


@pytest.mark.parametrize("approved, live, has_key", [
    (False, True, True), (True, False, True), (True, True, False),
    ("true", True, True), (1, True, True), (True, "yes", True),       # truthy is not enough
])
def test_no_client_without_approval_live_and_a_key(monkeypatch: pytest.MonkeyPatch, approved: Any, live: Any,
                                                   has_key: bool) -> None:
    if not has_key:
        monkeypatch.delenv(KEY_ENV)
    with pytest.raises(LiveCallsNotApproved):
        JevClient(make_settings(approved), live=live)


@pytest.mark.parametrize("revoked", ["prompts_approved", "live", "key"])
async def test_the_gate_is_checked_at_every_send(monkeypatch: pytest.MonkeyPatch, revoked: str) -> None:
    wire = Wire(httpx.Response(200, json=GOOD_BODY))
    if revoked == "prompts_approved":
        wire.settings["prompts_approved"] = False
    elif revoked == "live":
        wire.client._live = False
    else:
        monkeypatch.delenv(KEY_ENV)
    with pytest.raises(LiveCallsNotApproved):
        await wire.decide()
    assert wire.requests == []


async def test_429_5xx_and_timeouts_are_retried_with_backoff() -> None:
    wire = Wire(httpx.Response(429, headers={"Retry-After": "17"}), httpx.Response(503),
                httpx.ReadTimeout("timed out"), httpx.Response(200, json=GOOD_BODY))
    assert await wire.decide() == GOOD_BODY
    assert wire.delays[0] == 17 and 1 <= wire.delays[1] <= 2 and 2 <= wire.delays[2] <= 4   # 2 s and 4 s, jittered
    request = wire.requests[-1]
    assert len(wire.requests) == 4 and request.headers["Authorization"] == "Bearer sk-test"
    assert json.loads(request.content) == {"model": "typesafe/jev-1.13", "state": {"text": "שלום עולם"},
                                           "questions": QUESTIONS}
    assert "שלום עולם" in request.content.decode("utf-8")   # UTF-8, not \u escapes


@pytest.mark.parametrize("status, error, n_sent", [
    (500, JevHTTPError, 4), (400, JevHTTPError, 1), (401, JevAuthError, 1), (402, JevCreditsError, 1),
    (200, JevError, 1),   # body is not a JSON object
])
async def test_other_failures_are_not_retried_and_carry_the_body(status: int, error: type, n_sent: int) -> None:
    wire = Wire(*[httpx.Response(status, text="instructions must be a string")] * 5)
    with pytest.raises(error, match="instructions must be a string"):
        await wire.decide()
    assert len(wire.requests) == n_sent


def test_the_parser_reads_the_documented_and_the_nested_forms() -> None:
    parsed = parse_response(GOOD_BODY, QUESTIONS)
    assert set(parsed.answers) == set(QUESTIONS) and (parsed.input_tokens, parsed.cost_usd) == (1234, None)
    usage = {"prompt_tokens": 10, "completion_tokens": 2, "cost": 0.0001}
    nested = parse_response({"data": {"answers": GOOD_BODY["answers"], "usage": usage}}, QUESTIONS)
    assert set(nested.answers) == set(QUESTIONS)
    assert (nested.input_tokens, nested.output_tokens, nested.cost_usd) == (10, 2, 0.0001)
    assert set(parse_response({"data": GOOD_BODY["answers"]}, QUESTIONS).answers) == set(QUESTIONS)
    partial = parse_response({"answers": {"dim__gate": {"noul": 0.2}, "unasked": {"noul": 0.9}}}, QUESTIONS)
    assert set(partial.answers) == {"dim__gate"} and "no answer" in partial.invalid["dim__score"]
    assert validate_answer(score_answer(probabilities={"1": 0.33, "2": 0.33, "3": 0.33}), SCORE_Q) is None  # rounded


def score_answer(**changes: Any) -> dict[str, Any]:
    return {"type": "score", "score": 2.0, "probabilities": {"1": 0.25, "2": 0.5, "3": 0.25}, "confidence": 0.4,
            **changes}


@pytest.mark.parametrize("answer, payload, reason", [
    ("0.9", NOUL_Q, "not an object"),
    ({"type": "score", "score": 1.0}, NOUL_Q, "does not match"),
    ({"type": "noul", "noul": 1.2}, NOUL_Q, "noul"),
    ({"type": "noul", "noul": True}, NOUL_Q, "noul"),
    (score_answer(score=4.5), SCORE_Q, "score"),
    (score_answer(score=float("nan")), SCORE_Q, "score"),
    (score_answer(confidence=None), SCORE_Q, "confidence"),
    (score_answer(probabilities=None), SCORE_Q, "probabilities"),
    (score_answer(probabilities={"1": 0.5, "2": 0.3}), SCORE_Q, "sum"),
    (score_answer(probabilities={"1": 0.5, "7": 0.5}), SCORE_Q, "outside"),
    (score_answer(probabilities={"1": -0.5, "2": 1.5}), SCORE_Q, "probability"),
])
def test_malformed_answers_are_rejected_with_a_reason(answer: Any, payload: dict[str, Any], reason: str) -> None:
    parsed = parse_response({"answers": {"q": answer}}, {"q": payload})
    assert parsed.answers == {} and reason in parsed.invalid["q"]
