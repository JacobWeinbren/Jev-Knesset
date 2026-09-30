"""Draw every chart of the blog post into outputs/charts/blog/.

    python -m knesset_ches.blog            # all charts
    python -m knesset_ches.blog chamber    # one module
"""
from __future__ import annotations

import importlib
import sys
from typing import Sequence

from knesset_ches.tables import TABLES, load_tables

MODULES = ["opening", "chamber", "extra", "courts", "groups", "tone", "women"]


def main(argv: Sequence[str] | None = None) -> int:
    names = list(argv if argv is not None else sys.argv[1:]) or MODULES
    tables = load_tables(TABLES)
    failed = 0
    for name in names:
        module = importlib.import_module(f"knesset_ches.blog_{name}")
        for slug, draw in module.CHARTS.items():
            try:
                paths = draw(tables)
                print(f"{slug}: {len(paths)} files")
            except Exception as e:           # carry on with the rest
                failed += 1
                print(f"{slug}: FAILED: {e}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
