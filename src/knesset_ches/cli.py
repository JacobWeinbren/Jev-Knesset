"""Run the pipeline one step at a time.

    python -m knesset_ches.cli <step> [options]      (each step takes --help)

Steps, in order:
    download      fetch the Knesset Corpus and member metadata from Hugging Face
    extend        add the Knesset's own plenary protocols from April 2024 to July 2026
    corpus        cut the sentences into speeches
    packs         pack each member's speech for a year into a few long texts
    samples       draw the heckle samples and the speeches that name Netanyahu
    classify      ask Jev the questions (a dry run unless --live)
    aggregate     average the answers to members, parties and blocs
    validate      compare party positions with the Chapel Hill experts
    speech        summarise the speech-level answers
    demographics  members' gender, religion and origin, and the heckle tables
    blog          draw the charts
"""
from __future__ import annotations

import importlib
import os
import sys

from knesset_ches.questions import ROOT

STEPS = {
    "download": "knesset_ches.download",
    "extend": "knesset_ches.extend",
    "corpus": "knesset_ches.corpus",
    "packs": "knesset_ches.packs",
    "samples": "knesset_ches.samples",
    "classify": "knesset_ches.classify",
    "aggregate": "knesset_ches.aggregate",
    "validate": "knesset_ches.validate",
    "speech": "knesset_ches.speech_level",
    "demographics": "knesset_ches.demographics",
    "blog": "knesset_ches.blog",
}


def load_dotenv() -> None:
    """Read KEY=VALUE lines from .env without overriding variables already set."""
    env = ROOT / ".env"
    if not env.exists():
        return
    for line in env.read_text(encoding="utf-8").splitlines():
        key, sep, value = line.strip().partition("=")
        if sep and not key.startswith("#"):
            os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    step, rest = argv[0], argv[1:]
    if step not in STEPS:
        print(f"unknown step {step!r}; choose from: {', '.join(STEPS)}", file=sys.stderr)
        return 2
    load_dotenv()
    return int(importlib.import_module(STEPS[step]).main(rest) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
