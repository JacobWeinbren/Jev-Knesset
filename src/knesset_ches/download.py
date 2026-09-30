"""Download the Knesset Corpus sentence shards and metadata from Hugging Face.

    python -m knesset_ches.download
    python -m knesset_ches.download --only plenary

Only the no_morph shards are fetched. Interrupted downloads resume.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from knesset_ches.questions import ROOT

REPO = "HaifaCLGroup/KnessetCorpus"
API = f"https://huggingface.co/api/datasets/{REPO}/tree/main"
RESOLVE = f"https://huggingface.co/datasets/{REPO}/resolve/main"
HEADERS = {"User-Agent": "knesset-ches-jev/0.1"}
RAW = ROOT / "data" / "corpus" / "raw"

SHARD_DIRS = {
    "plenary": "protocols_sentences/plenary_no_morph_sentences_shards_bzip2_files",
    "committee": "protocols_sentences/committee_no_morph_sentences_shards_bzip2_files",
}
METADATA_FILES = [
    "README.md",
    "all_knesset_members_jsons.jsonl",
    "factions_jsons.jsonl",
    "meta_data/all_knesset_members_personal_and_factions_data.csv",
    "meta_data/knesset_members_factions_data.csv",
    "meta_data/knesset_members_personal_data.csv",
    "meta_data/factions_list.csv",
    "meta_data/factions_coalition_opposition_membership.csv",
    "meta_data/all_plenary_protocols_knessets_13-24.csv",
    "meta_data/all_committees_protocols_knessets_13-24.csv",
]


def list_dir(path: str = "") -> dict[str, int]:
    """Path and LFS size of each file in a dataset folder."""
    url = f"{API}/{urllib.parse.quote(path)}" if path else API
    with urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=60) as response:
        entries = json.load(response)
    return {e["path"]: (e.get("lfs") or {}).get("size", e.get("size", 0)) for e in entries if e["type"] == "file"}


def download(path: str, size: int | None, dest_root: Path, retries: int = 8) -> str:
    dest = dest_root / path
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size == size:
        return f"skip   {path}"
    part = dest.with_suffix(dest.suffix + ".part")
    url = f"{RESOLVE}/{urllib.parse.quote(path)}"
    for attempt in range(retries):
        try:
            have = part.stat().st_size if part.exists() else 0
            headers = {**HEADERS, "Range": f"bytes={have}-"} if have else HEADERS
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=120) as response:
                if have and response.status != 206:      # range ignored, start again
                    have = 0
                with open(part, "ab" if have else "wb") as f:
                    while chunk := response.read(1 << 20):
                        f.write(chunk)
            got = part.stat().st_size
            if size is not None and got != size:
                raise OSError(f"size mismatch {got} != {size}")
            part.replace(dest)
            return f"ok     {path} ({got / 1e6:.1f} MB)"
        except Exception as error:                       # back off and resume
            failure = error
            time.sleep(min(60, 2**attempt))
    return f"FAILED {path}: {failure}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="knesset-ches download", description="Download the Knesset Corpus.")
    parser.add_argument("--only", choices=["plenary", "committee", "metadata"], help="fetch just one part")
    parser.add_argument("--out-dir", type=Path, default=RAW)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)

    jobs: list[tuple[str, int | None]] = []
    if args.only in (None, "metadata"):
        sizes = list_dir() | list_dir("meta_data")
        jobs += [(path, sizes.get(path)) for path in METADATA_FILES]
    for kind, folder in SHARD_DIRS.items():
        if args.only in (None, kind):
            jobs += list(list_dir(folder).items())

    print(f"{len(jobs)} files, {sum(s or 0 for _, s in jobs) / 1e9:.2f} GB into {args.out_dir}", flush=True)
    failed = 0
    with ThreadPoolExecutor(args.workers) as pool:
        futures = [pool.submit(download, path, size, args.out_dir) for path, size in jobs]
        for i, future in enumerate(as_completed(futures), 1):
            message = future.result()
            failed += message.startswith("FAILED")
            print(f"[{i}/{len(jobs)}] {message}", flush=True)
    print(f"done with {failed} failures; run again to resume" if failed else "done", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
