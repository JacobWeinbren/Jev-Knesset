"""Load the settings and dimension files, and turn the dimensions into Jev question payloads.

Answers are stored under a hash of each question's payload, so an edited prompt gets asked again.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import yaml

ROOT = Path(__file__).resolve().parents[2]
DIMENSIONS_PATH = ROOT / "config" / "dimensions.yaml"
SETTINGS_PATH = ROOT / "config" / "settings.yaml"


@dataclass(frozen=True)
class Question:
    qid: str                 # e.g. "lrecon__score"
    dimension: str
    role: str                # gate, score, salience or flag
    payload: dict[str, Any]  # sent as is under questions[qid]
    prompt_hash: str
    n_levels: int | None = None


@dataclass(frozen=True)
class Dimension:
    id: str
    kind: str                     # position, salience or flag
    ches_variable: str | None
    gate_share_of: str | None     # salience: the position dimension on its subject
    valid_from: str | None        # not asked of earlier text
    raw: dict[str, Any] = field(hash=False, compare=False, repr=False)


def _clean(text: str) -> str:
    """Collapse the whitespace from YAML folding. This is what Jev sees."""
    return " ".join(str(text).split())


def _hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def _noul_payload(block: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "noul",
        "instructions": _clean(block["instructions"]),
        "criteria": {"true": _clean(block["when_true"]), "false": _clean(block["when_false"])},
    }


def _score_payload(block: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "score",
        "instructions": _clean(block["instructions"]),
        "criteria": [_clean(level) for level in block["levels"]],
    }


def load_settings(path: Path | str = SETTINGS_PATH) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def _speech_items(doc: dict[str, Any]) -> list[dict[str, Any]]:
    """Speech-level items as dimensions. Items with levels are salience, the rest flags."""
    out = []
    for group in doc.values():
        for item in group:
            if "levels" in item:
                score = {"instructions": item["instructions"], "levels": item["levels"]}
                out.append({"id": item["id"], "kind": "salience", "score": score})
            else:
                gate = {"instructions": item["question"], "when_true": item["when_true"],
                        "when_false": item["when_false"]}
                out.append({"id": item["id"], "kind": "flag", "gate": gate})
    return out


def load_dimensions(path: Path | str = DIMENSIONS_PATH) -> list[Dimension]:
    doc = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    items = doc["dimensions"] if "dimensions" in doc else _speech_items(doc)
    return [Dimension(raw["id"], raw["kind"], raw.get("ches_variable"), raw.get("gate_share_of"),
                      raw.get("valid_from"), raw) for raw in items]


def build_questions(dimensions: list[Dimension] | None = None) -> list[Question]:
    out: list[Question] = []
    for d in load_dimensions() if dimensions is None else dimensions:
        if d.kind in ("position", "flag"):
            gate = _noul_payload(d.raw["gate"])
            role, suffix = ("gate", "gate") if d.kind == "position" else ("flag", "flag")
            out.append(Question(f"{d.id}__{suffix}", d.id, role, gate, _hash(gate)))
        if d.kind in ("position", "salience"):
            score = _score_payload(d.raw["score"])
            role = "score" if d.kind == "position" else "salience"
            out.append(Question(f"{d.id}__score", d.id, role, score, _hash(score), len(score["criteria"])))
    return out


def questions_payload(questions: Iterable[Question]) -> dict[str, dict[str, Any]]:
    """The `questions` object of a request."""
    return {q.qid: q.payload for q in questions}

