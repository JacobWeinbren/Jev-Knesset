"""Add the plenary sittings after the corpus ends, April 2024 to July 2026, from the Knesset's own protocols.

    python -m knesset_ches.extend
    python -m knesset_ches.extend --steps parse --out-dir <folder>

Downloads are cached in data/corpus/extension/raw/ and converted with macOS textutil. parse writes shard_90 in the
corpus' format. If the Knesset revises a protocol, delete it from raw/protocols/ and run with --refresh-listings.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import re
import subprocess
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable

import pyarrow as pa
import pyarrow.parquet as pq

from knesset_ches.corpus import CORPUS_END, SENTENCE_SCHEMA
from knesset_ches.questions import ROOT

DATA_DIR = ROOT / "data" / "corpus"
RAW_DIR = DATA_DIR / "extension" / "raw"
ODATA_DIR = RAW_DIR / "odata"
PROTOCOL_DIR = RAW_DIR / "protocols"
TEXT_DIR = RAW_DIR / "text"
CORPUS_RAW = DATA_DIR / "raw"
STEPS = ["download", "convert", "parse"]

SHARD = 90
ODATA = "https://knesset.gov.il/Odata/ParliamentInfo.svc"
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/126.0 Safari/537.36 knesset-ches-jev research extension"
)
PAUSE_SECONDS = 0.8
PROTOCOL_GROUP_TYPE = 28                # "דברי הכנסת", the edited record the corpus uses
UUID_NAMESPACE = uuid.UUID("6f1f7c1e-5d0b-4c43-9a57-2a7f3e0b9a90")
FAR_FUTURE = dt.date(2099, 12, 31)


# Listings and files

_last_request = 0.0


def http_get(url: str, retries: int = 4) -> bytes:
    global _last_request
    for attempt in range(retries):
        wait = PAUSE_SECONDS - (time.time() - _last_request)
        if wait > 0:
            time.sleep(wait)
        _last_request = time.time()
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "*/*"})
            with urllib.request.urlopen(request, timeout=120) as response:
                return response.read()
        except (urllib.error.URLError, TimeoutError, ConnectionError) as error:
            if isinstance(error, urllib.error.HTTPError) and error.code == 404 or attempt == retries - 1:
                raise
            time.sleep(5 * (attempt + 1))


def odata_all(entity: str, filter_: str | None = None, orderby: str | None = None) -> list[dict[str, Any]]:
    params = {"$format": "json"}
    if filter_:
        params["$filter"] = filter_
    if orderby:
        params["$orderby"] = orderby
    url = f"{ODATA}/{entity}?" + urllib.parse.urlencode(params, quote_via=urllib.parse.quote, safe="$',()")
    rows: list[dict[str, Any]] = []
    while url:
        payload = json.loads(http_get(url))
        rows.extend(payload.get("value", []))
        url = payload.get("odata.nextLink", "")
        if url and not url.startswith("http"):
            url = f"{ODATA}/{url}"
        if url and "$format" not in url:
            url += "&$format=json"
    return rows


def listings(refresh: bool = False) -> dict[str, list[dict[str, Any]]]:
    """OData listings, cached under raw/odata/."""

    def cached(name: str, entity: str, filter_: str | None = None, orderby: str | None = None) -> list[dict]:
        path = ODATA_DIR / f"{name}.json"
        if refresh or not path.exists():
            ODATA_DIR.mkdir(parents=True, exist_ok=True)
            rows = odata_all(entity, filter_, orderby)
            path.write_text(json.dumps(rows, ensure_ascii=False, indent=0), encoding="utf-8")
        return json.loads(path.read_text(encoding="utf-8"))

    since = f"datetime'{CORPUS_END}T00:00:00'"
    sessions = cached("sessions", "KNS_PlenumSession", f"KnessetNum ge 25 and StartDate ge {since}", "StartDate")
    first = min(row["PlenumSessionID"] for row in sessions)
    documents = f"PlenumSessionID ge {first} and GroupTypeID eq {PROTOCOL_GROUP_TYPE}"
    return {
        "sessions": sessions,
        "documents": cached("documents", "KNS_DocumentPlenumSession", documents),
        "persons": cached("persons", "KNS_Person"),
        "positions": cached("person_to_position", "KNS_PersonToPosition", "KnessetNum ge 24"),
    }


@dataclass
class Sitting:
    knesset: int
    date: dt.date
    protocol_name: str
    file_url: str

    @property
    def raw_path(self) -> Path:
        return PROTOCOL_DIR / self.protocol_name

    @property
    def text_path(self) -> Path:
        return TEXT_DIR / f"{Path(self.protocol_name).stem}.txt"


def plenary_sittings(lists: dict[str, list[dict[str, Any]]]) -> list[Sitting]:
    """One protocol file per sitting after the corpus ends."""
    documents = defaultdict(list)
    for doc in lists["documents"]:
        if doc["GroupTypeID"] == PROTOCOL_GROUP_TYPE and doc.get("FilePath"):
            documents[doc["PlenumSessionID"]].append(doc)
    sittings = []
    for session in sorted(lists["sessions"], key=lambda s: (s["StartDate"], s["PlenumSessionID"])):
        date = dt.date.fromisoformat(session["StartDate"][:10])
        docs = documents[session["PlenumSessionID"]]
        if date <= CORPUS_END or not docs:
            continue
        # prefer .docx, then the latest update
        doc = sorted(docs, key=lambda d: (d["FilePath"].lower().endswith(".docx"), d.get("LastUpdatedDate") or ""))[-1]
        url = doc["FilePath"].strip()
        suffix = Path(urllib.parse.urlparse(url).path).suffix.lower() or ".doc"
        name = f"{session['KnessetNum']}_ptm_{doc['DocumentPlenumSessionID']}{suffix}"
        sittings.append(Sitting(int(session["KnessetNum"]), date, name, url))
    return sittings


def download(sittings: list[Sitting]) -> None:
    """Fetch missing protocol files. Failures are retried on the next run."""
    PROTOCOL_DIR.mkdir(parents=True, exist_ok=True)
    for sitting in sittings:
        path = sitting.raw_path
        if path.exists() and path.stat().st_size > 0:
            continue
        try:
            data = http_get(sitting.file_url)
        except OSError as error:
            print(f"  failed {sitting.protocol_name}: {error}", flush=True)
            continue
        tmp = path.with_suffix(path.suffix + ".part")
        tmp.write_bytes(data)
        tmp.replace(path)
        print(f"  {sitting.date} {sitting.protocol_name} ({len(data) / 1e6:.1f} MB)", flush=True)


def convert(sittings: list[Sitting]) -> None:
    TEXT_DIR.mkdir(parents=True, exist_ok=True)
    for sitting in sittings:
        source, out = sitting.raw_path, sitting.text_path
        if not source.exists():
            continue
        if out.exists() and out.stat().st_size > 0 and out.stat().st_mtime >= source.stat().st_mtime:
            continue
        tmp = out.with_suffix(".tmp.txt")
        result = subprocess.run(["textutil", "-convert", "txt", "-encoding", "UTF-8", "-output", str(tmp), str(source)],
                                capture_output=True, text=True)
        if result.returncode == 0 and tmp.exists():
            tmp.replace(out)
        else:
            print(f"  failed {sitting.protocol_name}: {result.stderr.strip()[:200]}", flush=True)


# Coalition status from press reports. A change counts from its announcement, or the next day if after the sitting.

TOI = "https://www.timesofisrael.com/"
WIKI_GOV37 = "https://en.wikipedia.org/wiki/Thirty-seventh_government_of_Israel"
COALITION_FIELDS = ["faction_id", "faction_name", "start_date", "end_date", "status", "source_url", "note"]
COALITION = [
    ("31", "הליכוד", "2024-04-04", "", "coalition", WIKI_GOV37,
     "Netanyahu's party; in the 37th government throughout (cross-checked with the Knesset faction list)"),
    ("111", "האיחוד הלאומי", "2024-04-04", "", "coalition",
     TOI + "otzma-yehudit-to-quit-coalition-sunday-religious-zionism-condemns-deal-but-remains/",
     "Religious Zionism (Smotrich); condemned the January 2025 Gaza deal but stayed; in the coalition throughout"),
    ("148", "עוצמה יהודית", "2024-04-04", "2025-01-18", "coalition", WIKI_GOV37,
     "Otzma Yehudit (Ben Gvir) in the government"),
    ("148", "עוצמה יהודית", "2025-01-19", "2025-03-17", "opposition",
     TOI + "otzma-yehudit-exits-coalition-over-gaza-deal-blasting-it-as-victory-for-terrorism/",
     "Quit the coalition on Sunday 19 Jan 2025 over the hostage-ceasefire deal (resignations took effect 21 Jan); "
     "said it would not topple the government. Outside the coalition, counted as opposition"),
    ("148", "עוצמה יהודית", "2025-03-18", "", "coalition",
     TOI + "far-right-otzma-yehudit-returning-to-the-government-after-resumption-of-war-in-gaza/",
     "Returned on 18 Mar 2025 when the fighting in Gaza resumed; the Knesset approved the reappointments on 19 Mar "
     "(also https://www.jns.org/ben-gvirs-otzma-yehudit-rejoins-israeli-govt/)"),
    ("75", "התאחדות הספרדים שומרי תורה", "2024-04-04", "", "coalition",
     TOI + "shas-bolts-government-over-haredi-enlistment-remains-part-of-pms-coalition/",
     "Shas. Its ministers quit the GOVERNMENT on 16-17 Jul 2025 over the draft law but the party said it stayed in "
     "the COALITION; it also gave up its committee chairs on 23 Oct 2025 "
     "(https://www.jpost.com/israel-news/politics-and-diplomacy/article-871383) while saying it would not leave the "
     "coalition. Coded coalition throughout; a judgement call"),
    ("90", "יהדות התורה", "2024-04-04", "2025-07-14", "coalition", WIKI_GOV37,
     "United Torah Judaism in the government"),
    ("90", "יהדות התורה", "2025-07-15", "", "opposition",
     TOI + "united-torah-judaism-quits-govt-to-protest-new-proposal-on-army-enlistment-exemptions/",
     "Degel HaTorah and then Agudat Yisrael quit the government AND the coalition on the evening of Mon 14 Jul 2025 "
     "over the draft law; did not return before the Knesset dissolved (17 Jul 2026). Outside the coalition, counted "
     "as opposition"),
    ("150", "נעם", "2024-04-04", "2025-07-16", "coalition",
     TOI + "far-right-lawmaker-avi-maoz-quits-coalition-leaving-pm-with-a-minority-government/",
     "Noam (Avi Maoz, one MK). Resigned as deputy minister in March 2025 but stayed in the coalition"),
    ("150", "נעם", "2025-07-17", "", "opposition",
     TOI + "far-right-lawmaker-avi-maoz-quits-coalition-leaving-pm-with-a-minority-government/",
     "Maoz declared on the evening of 16 Jul 2025 that he was no longer part of the coalition. Counted as opposition"),
    ("146", "הימין הממלכתי", "2024-04-04", "2024-09-28", "opposition",
     TOI + "saar-rejoins-government-bolstering-netanyahu-as-longtime-rivals-say-rifts-mended/",
     "New Hope (Sa'ar; Knesset name 'HaYamin HaMamlakhti') left National Unity and the emergency government in March "
     "2024; the corpus codes it opposition from 27 Mar 2024"),
    ("146", "הימין הממלכתי", "2024-09-29", "", "coalition",
     TOI + "saar-rejoins-government-bolstering-netanyahu-as-longtime-rivals-say-rifts-mended/",
     "Sa'ar rejoined the government on 29 Sep 2024 (foreign minister from Nov 2024). Party merger with Likud signed "
     "Mar 2025 and approved by Likud on 13 Aug 2025 "
     "(https://www.timesofisrael.com/ruling-likud-party-approves-merger-with-gideon-saars-new-hope/), but the "
     "Knesset factions stayed separate (KNS_Faction 1108 still current)"),
    ("142", "כחול לבן", "2024-04-04", "2024-06-09", "coalition",
     TOI + "urging-elections-gantz-quits-wartime-government-accuses-pm-of-botching-war-effort/",
     "Gantz's National Unity (Knesset faction 1098; the corpus codes it General_ID 142 'Blue and White' after the 13 "
     "Mar 2024 split) sat in the emergency unity government from 11-12 Oct 2023. The corpus leaves this faction's "
     "coalition field null for Oct 2023-Apr 2024"),
    ("142", "כחול לבן", "2024-06-10", "", "opposition",
     TOI + "urging-elections-gantz-quits-wartime-government-accuses-pm-of-botching-war-effort/",
     "Gantz announced the exit on the evening of Sun 9 Jun 2024. From 8 Jul 2025 the faction is Knesset faction 1110 "
     "'כחול לבן - המחנה הממלכתי' (Eisenkot and Kahana had left), same General_ID"),
    ("136", "יש עתיד", "2024-04-04", "", "opposition", WIKI_GOV37,
     "Yesh Atid (Lapid), leader of the opposition throughout"),
    ("110", "ישראל ביתנו", "2024-04-04", "", "opposition", WIKI_GOV37,
     "Yisrael Beiteinu (Liberman), opposition throughout"),
    ("35", "העבודה", "2024-04-04", "", "opposition", WIKI_GOV37,
     "Labor faction (the party merged with Meretz outside the Knesset as 'The Democrats' in 2024; the Knesset "
     "faction kept its name), opposition throughout"),
    ("118", "החזית הדמוקרטית לשלום ולשוויון - תנועה ערבית להתחדשות", "2024-04-04", "", "opposition", WIKI_GOV37,
     "Hadash-Ta'al, opposition throughout"),
    ("101", "הרשימה הערבית המאוחדת (2)", "2024-04-04", "", "opposition", WIKI_GOV37,
     "Ra'am, opposition throughout"),
    ("9001", 'חה"כ עידן רול', "2025-01-14", "2025-08-13", "opposition",
     TOI + "knesset-approves-mk-idan-rolls-request-to-leave-yesh-atid-party/",
     "Idan Roll's one-MK faction ('National Majority') after the House Committee let him leave Yesh Atid on 14 Jan "
     "2025; not in the coalition; he left the Knesset on 13 Aug 2025 (KNS_Faction 1109)"),
]


def coalition_spells() -> dict[str, list[tuple[dt.date, dt.date, str]]]:
    spells: dict[str, list[tuple[dt.date, dt.date, str]]] = defaultdict(list)
    for faction_id, _, start, end, status, *_ in COALITION:
        end_date = dt.date.fromisoformat(end) if end else FAR_FUTURE
        spells[faction_id].append((dt.date.fromisoformat(start), end_date, status))
    return spells


def coalition_status(spells: dict[str, list[tuple[dt.date, dt.date, str]]], faction_id: str | None,
                     date: dt.date) -> str | None:
    for start, end, status in spells.get(faction_id, []):
        if start <= date <= end:
            return status
    return None


# People and factions

MK_POSITIONS = {43, 61}                        # חבר הכנסת / חברת הכנסת
FACTION_MEMBER_POSITION = 54                   # חבר/ת סיעה, carries the FactionID
# ministers, the prime minister, the Speaker and their deputies
SERVING_POSITIONS = {31, 39, 40, 45, 50, 51, 57, 59, 65, 73, 285079, 70, 71, 122, 123, 285078}
SPLIT_2024 = dt.date(2024, 3, 13)              # National Unity split: 1098 is 149 before, 142 after
# FactionIDs newer than the corpus
FACTION_OVERRIDES: dict[int, str] = {
    1109: "9001",   # Idan Roll's one-MK faction
    1110: "142",    # National Unity renamed, July 2025
}
NEW_FACTION_FIELDS = ["id", "name", "knesset", "kns_faction_id", "bloc_suggestion", "basis"]
NEW_FACTIONS = [
    ("9001", 'חה"כ עידן רול', 25, 1109, "Secular Centre",
     "Idan Roll left Yesh Atid (Secular Centre) on 14 Jan 2025 to sit alone as 'National Majority', calling for a "
     "liberal, post-October-7 centre; stayed in the opposition; left the Knesset 13 Aug 2025. Source: "
     "https://www.timesofisrael.com/knesset-approves-mk-idan-rolls-request-to-leave-yesh-atid-party/"),
]
FACTION_NAMES = {
    "31": "הליכוד", "35": "העבודה", "75": "התאחדות הספרדים שומרי תורה", "90": "יהדות התורה",
    "101": "הרשימה הערבית המאוחדת (2)", "110": "ישראל ביתנו", "111": "האיחוד הלאומי",
    "118": "החזית הדמוקרטית לשלום ולשוויון - תנועה ערבית להתחדשות", "136": "יש עתיד", "142": "כחול לבן",
    "146": "הימין הממלכתי", "148": "עוצמה יהודית", "149": "המחנה הממלכתי", "150": "נעם", "9001": 'חה"כ עידן רול',
}
GENERAL_NAMES = {"146": "תקווה חדשה"}
FACTION_1110_NAME = "כחול לבן - המחנה הממלכתי"

_NIQQUD_RE = re.compile(r"[֑-ׇ]")
_SPACE_RE = re.compile(r"\s+")


def norm_name(text: str) -> str:
    """Name key without niqqud, punctuation or dashes."""
    text = unicodedata.normalize("NFC", text)
    text = _NIQQUD_RE.sub(lambda m: "" if m.group() not in "־׳״" else m.group(), text)
    text = text.replace("־", " ").replace("–", " ").replace("—", " ").replace("-", " ")
    text = re.sub(r"[\"'`׳״’‘“”()\[\].,:;]", "", text)
    return _SPACE_RE.sub(" ", text).strip()


def stable_uuid(name: str) -> str:
    return str(uuid.uuid5(UUID_NAMESPACE, norm_name(name) or name))


def _date(value: str | None, default: dt.date) -> dt.date:
    return dt.date.fromisoformat(value[:10]) if value else default


def _spells(rows: Iterable[dict[str, Any]]) -> list[tuple[dt.date, dt.date]]:
    return [(_date(r["StartDate"], dt.date(1900, 1, 1)), _date(r.get("FinishDate"), FAR_FUTURE)) for r in rows]


@dataclass
class Person:
    person_id: str
    name: str                                   # the corpus' spelling if it has one
    is_mk: bool
    variants: set[str] = field(default_factory=set)
    last_keys: set[str] = field(default_factory=set)
    first_keys: set[str] = field(default_factory=set)
    spells: list[tuple[dt.date, dt.date]] = field(default_factory=list)     # as MK, minister or Speaker
    memberships: list[tuple[dt.date, dt.date, int]] = field(default_factory=list)   # start, end, KNS FactionID

    def active(self, date: dt.date) -> bool:
        pad = dt.timedelta(days=3)
        return any(start - pad <= date <= end + pad for start, end in self.spells)

    def faction_on(self, date: dt.date) -> int | None:
        """KNS FactionID on the date. On a changeover day the newer membership wins."""
        hits = [(start, faction) for start, end, faction in self.memberships if start <= date <= end]
        return max(hits)[1] if hits else None


# misprints seen in the protocols
NAME_ALIASES: dict[str, list[str]] = {
    "30894": ["סמיר אל סעיד"],      # Samir Bin Said, 23 Jun 2025
}


def _name_variants(first: str, last: str) -> tuple[set[str], set[str], set[str]]:
    """Full, last and first name keys for one spelling."""
    firsts = {first}
    bracket = re.search(r"\(([^)]*)\)", first)
    if bracket:                                           # 'טטיאנה (טניה)' gives 'טטיאנה' and 'טניה'
        firsts = {re.sub(r"\([^)]*\)", "", first), bracket.group(1)}
    lasts = {re.sub(r"\([^)]*\)", "", last)}
    bracket = re.search(r"\(([^)]*)\)", last)
    if bracket:
        lasts.add(bracket.group(1))
    full: set[str] = set()
    first_keys: set[str] = set()
    for f in firsts:
        f_key = norm_name(f)
        tokens = f_key.split()
        first_keys.update(tokens)
        options = {f_key}
        if len(tokens) > 1:                               # 'מכלוף מיקי' also gives 'מכלוף' and 'מיקי'
            options.update(tokens)
        for option in options:
            for l in lasts:
                if norm_name(l):
                    full.add(norm_name(f"{option} {l}"))
    return full, {norm_name(l) for l in lasts if norm_name(l)}, first_keys


def _corpus_members() -> list[dict[str, Any]]:
    with (CORPUS_RAW / "all_knesset_members_jsons.jsonl").open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_people(lists: dict[str, list[dict[str, Any]]]) -> dict[str, Person]:
    """KNS persons with their corpus names, spells in office and factions."""
    corpus = {str(row["person_id"]): row for row in _corpus_members()}
    positions: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in lists["positions"]:
        positions[str(row["PersonID"])].append(row)
    people: dict[str, Person] = {}
    for row in lists["persons"]:
        pid = str(row["PersonID"])
        first, last = (row.get("FirstName") or "").strip(), (row.get("LastName") or "").strip()
        known = corpus.get(pid)
        rows = positions.get(pid, [])
        mk_rows = [r for r in rows if r["PositionID"] in MK_POSITIONS]
        is_mk = bool(mk_rows) or bool(known and known.get("is_knesset_member") in (True, "True", "true"))
        name = (known.get("full_name") if known else None) or f"{first} {last}".strip()
        person = Person(pid, name, is_mk)
        spellings = [(first, last)]
        if known:
            spellings.append((known.get("first_name") or "", known.get("last_name") or ""))
        for f, l in spellings:
            full, lasts, firsts = _name_variants(f, l)
            person.variants |= full
            person.last_keys |= lasts
            person.first_keys |= firsts
        if known and known.get("full_name"):
            person.variants.add(norm_name(re.sub(r"\([^)]*\)", "", known["full_name"])))
        for alias in NAME_ALIASES.get(pid, []):
            person.variants.add(norm_name(alias))
        person.variants.discard("")
        person.spells = _spells(r for r in rows if r["PositionID"] in MK_POSITIONS | SERVING_POSITIONS)
        person.memberships = [
            (_date(r["StartDate"], dt.date(1900, 1, 1)), _date(r.get("FinishDate"), FAR_FUTURE), int(r["FactionID"]))
            for r in rows if r["PositionID"] == FACTION_MEMBER_POSITION and r.get("FactionID") and r["KnessetNum"] >= 25
        ]
        people[pid] = person
    return people


def faction_general_ids() -> dict[int, str]:
    mapping: dict[int, str] = {}
    with (CORPUS_RAW / "meta_data" / "knesset_members_factions_data.csv").open(encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            if row.get("FactionID", "").strip().isdigit() and row.get("General_ID", "").strip():
                mapping.setdefault(int(row["FactionID"]), row["General_ID"].strip())
    for member in _corpus_members():
        for m in member.get("factions_memberships") or []:
            kid, gid = str(m.get("knesset_faction_id") or ""), str(m.get("faction_id") or "")
            if kid.isdigit() and gid:
                mapping.setdefault(int(kid), gid)
    return mapping | {1098: "149"} | FACTION_OVERRIDES      # 1098 before the split, see general_id


def general_id(mapping: dict[int, str], kns_faction_id: int | None, date: dt.date) -> str | None:
    if kns_faction_id is None:
        return None
    if kns_faction_id == 1098 and date >= SPLIT_2024:
        return "142"
    return mapping.get(kns_faction_id)


def faction_labels(faction_id: str, kns_faction_id: int | None) -> tuple[str | None, str | None]:
    """faction_name and faction_general_name, as in the corpus."""
    name = FACTION_NAMES.get(faction_id)
    general = GENERAL_NAMES.get(faction_id, name)
    if faction_id == "142" and kns_faction_id == 1110:
        name = FACTION_1110_NAME
    return name, general


# Turns and sentences

SPEAKER_TAGS = {"יור": "chair", "דובר": "speaker", "דובר_המשך": "speaker", "קריאה": "interjection",
                "קריאות": "interjection", "אורח": "speaker", "דור_המשך": "speaker", "דשובר": "speaker",
                "דובר_זהמשך": "speaker", "דובר_המש ך": "speaker", "קרתיאה": "interjection"}
HEADING_TAGS = {"נושא", "הצח", "הלסי", "שאילתה"}    # headings, not speech
END_TAGS = {"סיום", "הפסקה"}                          # sitting closed or suspended
NOTE_TAGS = {"מנהל", "אחרי_כן"}                       # editorial notes
_TAG_ALT = "|".join(re.escape(w) for w in sorted(set(SPEAKER_TAGS) | HEADING_TAGS | END_TAGS | NOTE_TAGS,
                                                  key=len, reverse=True))
_SPEAKER_ALT = "|".join(re.escape(w) for w in sorted(SPEAKER_TAGS, key=len, reverse=True))
_OPEN_RE = re.compile(rf"^\s*<?\s*<?\s*({_TAG_ALT})\s*>\s*>?")
_CLOSE_RE = re.compile(rf"<\s*<?\s*(?:{_TAG_ALT})?\s*(?:>\s*>?)?\s*$")
_ANY_TAG_RE = re.compile(rf"<\s*<\s*(?:{_TAG_ALT})\s*>\s*>|<<|>>")
_EMBEDDED_TAG_RE = re.compile(rf"^(?P<head>.*?\S)\s*(?P<tail><<\s*(?:{_SPEAKER_ALT})\s*>>.*)$")
_PAREN_LINE_RE = re.compile(r"^\(.*\)\.?$")
_LIST_ITEM_RE = re.compile(r"^\d+\s*\.")
# A vote header, typo הצביעה included. "הצבעה." with a full stop is the chair calling a vote, not a header.
# The result block after a header runs to the next blank line and is dropped whole.
_VOTE_START_RE = re.compile(r"^(?:הצבעה|הצביעה)(?:(?: מס['׳]| מספר)?\s*\d+\.?| מס['׳])?$")


@dataclass
class RawTurn:
    kind: str                 # chair, speaker or interjection
    line: str                 # the speaker line as printed
    lines: list[str] = field(default_factory=list)


def classify_tag_line(line: str) -> tuple[str, str] | None:
    """(tag, text) for a tag line, malformed ones included."""
    opening = _OPEN_RE.match(line)
    if opening is None:
        return None
    rest = line[opening.end():]
    closing = _CLOSE_RE.search(rest)
    inner = rest[: closing.start()] if closing and closing.group(0).strip() else rest
    inner = _ANY_TAG_RE.sub(" ", inner)
    return opening.group(1), _SPACE_RE.sub(" ", inner).strip()


def _split_embedded_tags(lines: list[str]) -> list[str]:
    """Move a speaker tag printed at the end of a paragraph onto its own line."""
    out: list[str] = []
    for raw in lines:
        if "<<" in raw and not _OPEN_RE.match(raw):
            match = _EMBEDDED_TAG_RE.match(raw)
            if match and len(match.group("head").strip()) >= 5 and re.search(r"\w{2,}", match.group("head")):
                tagged = classify_tag_line(match.group("tail"))
                if tagged and tagged[0] in SPEAKER_TAGS and tagged[1].endswith(":") and "<<" not in match.group("head"):
                    out.extend([match.group("head"), match.group("tail")])
                    continue
        out.append(raw)
    return out


def split_turns(text: str) -> list[RawTurn]:
    text = text.replace("\x0c", "\n").replace(" ", "\n").replace(" ", "\n").replace("\r", "\n")
    lines = _split_embedded_tags(text.split("\n"))
    start = next((i for i, line in enumerate(lines) if classify_tag_line(line)), len(lines))
    turns: list[RawTurn] = []
    current: RawTurn | None = None
    last_chair: RawTurn | None = None
    vote = None                     # None, "header" or "block"
    for raw in lines[start:]:
        tagged = classify_tag_line(raw)
        if tagged is None and _CLOSE_RE.search(raw) and "<<" in raw:
            # opening tag lost: "... << נושא >>"
            word = re.search(rf"({_TAG_ALT})", raw[raw.rfind("<"):] if "<" in raw else "")
            inner = _SPACE_RE.sub(" ", _ANY_TAG_RE.sub(" ", raw[: raw.rfind("<<")])).strip()
            if word and (word.group(1) in HEADING_TAGS or inner.endswith(":")):
                tagged = (word.group(1), inner)
        if tagged is not None:
            word, inner = tagged
            vote = None
            if word in SPEAKER_TAGS:
                current = RawTurn(SPEAKER_TAGS[word], inner)
                turns.append(current)
                if current.kind == "chair" and inner:
                    last_chair = current
            elif word in HEADING_TAGS or word in END_TAGS:
                current = None
            elif word == "אחרי_כן" and inner and not _PAREN_LINE_RE.match(inner) and current is not None:
                current.lines.append(inner)
            continue
        line = _ANY_TAG_RE.sub(" ", raw) if "<<" in raw or ">>" in raw else raw
        line = _SPACE_RE.sub(" ", line.replace("\t", " ")).strip()
        if not line:
            if vote == "block":
                vote = None
            continue
        if _VOTE_START_RE.match(line):
            vote = "header"
            continue
        if vote is not None:
            vote = "block"
            continue
        if current is None:
            # Text after a heading is dropped unless the chair carries on from their own turn. It then joins that
            # turn rather than opening a new one, which would shift the turn numbers in unit ids.
            resumable = (
                last_chair is not None and turns and turns[-1] is last_chair
                and "<<" not in raw and ">>" not in raw
                and not line.startswith(("(", ")", "[", "]")) and not _LIST_ITEM_RE.match(line)
                and len(line.split()) >= 3
            )
            if not resumable:
                continue
            current = last_chair
        if _PAREN_LINE_RE.match(line):
            continue
        current.lines.append(line)
    return turns


_SENT_SPLIT_RE = re.compile(r'(?<=[.!?])\s+|(?<=[.!?]["״”])\s+')
_FINAL_STOP_RE = re.compile(r'[.\s]+(?=["״”]*$)')


def sentences_of(line: str) -> list[str]:
    """Split a paragraph after . ? ! and drop final full stops, as the corpus does."""
    line = line.replace("– – –", "- - -").replace("—", "–")
    out: list[str] = []
    for part in _SENT_SPLIT_RE.split(line):
        part = _FINAL_STOP_RE.sub("", part.strip()).strip()
        if part and re.search(r"\w|-", part):
            out.append(part)
    return out


# Speakers

ANONYMOUS = {"קריאה", "קריאות"}
_CHAIR_PREFIXES = ("היור ", "היוריי ", "היושב ראש ", "היושבת ראש ", "יור הכנסת ", "יושב ראש הכנסת ", "יושבת ראש הכנסת ")
_ON_BEHALF_RE = re.compile(r"\s+[–-]?\s*בשם\s+(?:ה?וועד|ועד|יושב|יו\"ר|הסיע).*$")
# titles of speakers not in KNS_Person, stripped from the name
_TITLE_RE = re.compile(
    r"^(?:סג(?:ן|נית) מזכיר(?:ת)? הכנסת|מזכיר(?:ת)? הכנסת|נשיא(?:ת)? המדינה|"
    r"נשיא(?:ת)? ארצות הברית(?: של אמריקה)?|נשיא(?:ת)? הרפובליקה של \S+|נשיא(?:ת)? \S+|ראש(?:ת)? ממשלת \S+|"
    r"יושב(?:ת)? ראש (?:בית )?\S+(?: \S+)?|היועצ(?:ת)? המשפטי(?:ת)? (?:לממשלה|לכנסת)|נציב(?:ת)? \S+)\s+"
)
# party labels in speaker brackets, to tell namesakes apart
PARTY_HINTS = {
    "יש עתיד": "136", "הליכוד": "31", "סיעת הליכוד": "31", "העבודה": "35", "חדש תעל": "118", "ישראל ביתנו": "110",
    "רעם הרשימה הערבית המאוחדת": "101", "שס": "75", "המחנה הממלכתי": "142", "הציונות הדתית": "111",
    "כחול לבן המחנה הממלכתי": "142", "כחול לבן": "142", "עוצמה יהודית": "148", "יהדות התורה": "90",
    "הימין הממלכתי": "146", "נעם": "150",
}


@dataclass
class Speaker:
    speaker_id: str
    speaker_name: str
    is_valid: bool
    is_chair: bool
    is_mk: bool
    person: Person | None


def _ratio(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio() if a and b else 0.0


class Resolver:
    """Matches printed speaker lines to KNS persons."""

    def __init__(self, people: dict[str, Person], factions: dict[int, str]):
        self.people = people
        self.factions = factions
        self.by_variant: dict[str, set[str]] = defaultdict(set)
        self.by_last: dict[str, set[str]] = defaultdict(set)
        for pid, person in people.items():
            for key in person.variants:
                self.by_variant[key].add(pid)
            for key in person.last_keys:
                self.by_last[key].add(pid)
        self.cache: dict[tuple[str, str, dt.date], Speaker] = {}

    def _party(self, pid: str, date: dt.date) -> str | None:
        return general_id(self.factions, self.people[pid].faction_on(date), date)

    def _pick(self, pids: Iterable[str], date: dt.date, party: str | None) -> list[str]:
        pids = list(pids)
        pool = [pid for pid in pids if self.people[pid].active(date)] or pids
        if len(pool) > 1 and party:
            pool = [pid for pid in pool if self._party(pid, date) == party] or pool
        if len(pool) > 1:
            pool = [pid for pid in pool if self.people[pid].is_mk] or pool
        return pool

    def match(self, tokens: list[str], date: dt.date, party: str | None) -> tuple[Person | None, int]:
        """The person named by the last tokens, and how many tokens the name uses."""
        for k in range(min(6, len(tokens)), 0, -1):
            key = " ".join(tokens[-k:])
            if key in self.by_variant:
                pool = self._pick(self.by_variant[key], date, party)
                if len(pool) == 1:
                    return self.people[pool[0]], k
        # a known surname after an unknown first name or nickname
        for k in range(min(3, len(tokens)), 0, -1):
            key = " ".join(tokens[-k:])
            active = [pid for pid in self.by_last.get(key, ()) if self.people[pid].active(date)]
            if not active:
                continue
            before = tokens[-k - 1] if len(tokens) > k else ""
            scored = sorted(((max(_ratio(before, f) for f in self.people[pid].first_keys or {""}), pid)
                             for pid in active), reverse=True)
            if party and len(scored) > 1:
                scored = [s for s in scored if self._party(s[1], date) == party] or scored
            if len(scored) == 1 or scored[0][0] >= 0.75 and scored[0][0] - scored[1][0] >= 0.15:
                return self.people[scored[0][1]], k + (1 if before and scored[0][0] >= 0.6 else 0)
        # fuzzy, against everyone serving that day, if all close matches are one person
        best: list[tuple[float, str, int]] = []
        for pid, person in self.people.items():
            if not person.active(date):
                continue
            for variant in person.variants:
                width = len(variant.split())
                for k in {width - 1, width, width + 1}:
                    if 1 <= k <= len(tokens):
                        score = _ratio("".join(tokens[-k:]), variant.replace(" ", ""))
                        if score >= 0.84:
                            best.append((score, pid, k))
        best.sort(reverse=True)
        if best and len({pid for _, pid, _ in best}) == 1:
            return self.people[best[0][1]], best[0][2]
        return None, 0

    def resolve(self, line: str, kind: str, date: dt.date) -> Speaker:
        key = (kind, line, date)
        if key not in self.cache:
            self.cache[key] = self._resolve(line, kind, date)
        return self.cache[key]

    def _resolve(self, line: str, kind: str, date: dt.date) -> Speaker:
        clean = line.strip().rstrip(":").strip()
        brackets = re.findall(r"\(([^)]*)\)", clean)
        clean = re.sub(r"\([^)]*\)?", " ", clean)
        clean = _ON_BEHALF_RE.sub("", clean)
        clean = re.sub(r"\s+[–-]\s*$", "", clean)
        key = norm_name(clean)
        if key in ANONYMOUS:
            return Speaker(stable_uuid(key), key, False, False, False, None)
        is_chair = kind == "chair" or key.startswith(_CHAIR_PREFIXES) or key in {"היור", "היושב ראש"}
        for prefix in _CHAIR_PREFIXES:
            if key.startswith(prefix):
                key = key[len(prefix):].strip()
                break
        party = None
        for bracket in brackets:
            party = PARTY_HINTS.get(norm_name(bracket)) or party
        tokens = key.split()
        person, _ = self.match(tokens, date, party) if tokens else (None, 0)
        if person is not None:
            speaker_id = person.person_id if person.is_mk else stable_uuid(person.name)
            return Speaker(speaker_id, person.name, True, is_chair, person.is_mk, person)
        # officials, guests or parsing debris
        name = _TITLE_RE.sub("", key).strip() or key
        debris = not name or is_chair or len(name.split()) > 6 or bool(re.search(r"\d", name))
        return Speaker(stable_uuid(name or line), name or line.strip(), not debris, is_chair, False, None)


# Rows

def build_rows(sittings: list[Sitting], resolver: Resolver) -> list[dict[str, Any]]:
    """Sentence rows in corpus.SENTENCE_SCHEMA."""
    coalition = coalition_spells()
    rows: list[dict[str, Any]] = []
    for sitting in sittings:
        last_chair: Speaker | None = None
        for turn_no, raw in enumerate(split_turns(sitting.text_path.read_text(encoding="utf-8"))):
            speaker = resolver.resolve(raw.line, raw.kind, sitting.date)
            if not raw.line.strip() and raw.kind == "chair" and last_chair is not None:
                speaker = last_chair                     # chair tag without a name: same chair
            elif speaker.is_chair and speaker.person is not None:
                last_chair = speaker
            kns_faction = speaker.person.faction_on(sitting.date) if speaker.person and speaker.is_mk else None
            faction_id = general_id(resolver.factions, kns_faction, sitting.date)
            name, general = faction_labels(faction_id, kns_faction) if faction_id else (None, None)
            status = coalition_status(coalition, faction_id, sitting.date)
            sentences = [s for line in raw.lines for s in sentences_of(line)]
            for sent_no, sentence in enumerate(sentences):
                rows.append({
                    "shard": SHARD, "line": len(rows), "protocol_name": sitting.protocol_name,
                    "protocol_type": "plenary", "knesset": sitting.knesset, "date": sitting.date, "is_ocr": False,
                    "speaker_id": speaker.speaker_id, "speaker_name": speaker.speaker_name,
                    "is_valid_speaker": speaker.is_valid, "is_chairman": speaker.is_chair and speaker.is_valid,
                    "is_mk": speaker.is_mk, "turn": turn_no, "sent": sent_no,
                    "faction_id": faction_id, "faction_name": name, "faction_general_name": general,
                    "coalition": status, "text": sentence,
                })
    return rows


def write_csv(path: Path, header: list[str], rows: list[tuple]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def parse(sittings: list[Sitting], lists: dict[str, list[dict[str, Any]]], out_dir: Path) -> None:
    resolver = Resolver(load_people(lists), faction_general_ids())
    rows = build_rows(sittings, resolver)
    shard = out_dir / "processed" / "_sentences" / "plenary" / f"shard_{SHARD}.parquet"
    shard.parent.mkdir(parents=True, exist_ok=True)
    tmp = shard.with_suffix(".parquet.tmp")
    pq.write_table(pa.Table.from_pylist(rows, schema=SENTENCE_SCHEMA), tmp, compression="zstd", row_group_size=50_000)
    tmp.replace(shard)
    write_csv(out_dir / "extension" / "coalition_2024_2026.csv", COALITION_FIELDS, COALITION)
    write_csv(out_dir / "extension" / "new_factions.csv", NEW_FACTION_FIELDS, NEW_FACTIONS)
    print(f"wrote {len(rows):,} sentences to {shard}", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="knesset-ches extend", description="Add the sittings after April 2024.")
    parser.add_argument("--steps", nargs="+", choices=STEPS, default=STEPS)
    parser.add_argument("--refresh-listings", action="store_true", help="query the OData service again")
    parser.add_argument("--out-dir", type=Path, default=DATA_DIR,
                        help="where parse writes processed/_sentences/plenary/shard_90.parquet and extension/*.csv")
    args = parser.parse_args(argv)
    lists = listings(args.refresh_listings)
    sittings = plenary_sittings(lists)
    print(f"{len(sittings)} sittings, {sittings[0].date} to {sittings[-1].date}", flush=True)
    if "download" in args.steps:
        download(sittings)
    if "convert" in args.steps:
        convert(sittings)
    if "parse" in args.steps:
        parse(sittings, lists, args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
