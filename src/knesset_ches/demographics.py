"""Members' gender, religion and origin, women's share of speech, and heckles at women and men.
Run after classify has scored the heckle samples.

    python -m knesset_ches.demographics [--tables DIR]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from knesset_ches import speech_level as S
from knesset_ches import tables as T
from knesset_ches.questions import ROOT, load_settings

PROCESSED = ROOT / "data" / "corpus" / "processed"
UNITS = PROCESSED / "units.parquet"
SENTENCES = PROCESSED / "_sentences" / "plenary"
HECKLES = PROCESSED / "interjections_sample.parquet"
HECKLES_V2 = PROCESSED / "interjections_sample_v2.parquet"
MEMBERS = ROOT / "data" / "corpus" / "raw" / "all_knesset_members_jsons.jsonl"
# members first seated after April 2024, missing from the corpus member file
NEW_MEMBERS = ROOT / "data" / "corpus" / "extension" / "members_demographics_2024_2026.csv"
HANDCHECK = T.HANDCHECK

FSU = {"ברית המועצות", "רוסיה", "אוקראינה", "אוזבקיסטן", "גאורגיה", "בלארוס", "מולדובה", "לטביה", "ליטא", "אסטוניה",
       "קזחסטן", "אזרבייג'ן"}
MENA = {"מרוקו", "עיראק", "תימן", "תוניסיה", "אלג'יריה", "לוב", "מצרים", "סוריה", "לבנון", "איראן", "טורקיה",
        "אפגניסטן", "הודו"}
GENDER = {"male": "Men", "female": "Women"}
GROUP = {"יהודי": "Jewish", "ערבי": "Arab", "דרוזי": "Druze", "בדואי": "Bedouin"}
RELIGIOSITY = {"חילוני": "Secular", "דתי": "Religious", "חרדי": "Haredi"}
RATES = {"any_per_kw": "all", "by_members_per_kw": "member", "by_chair_per_kw": "chair",
         "anonymous_per_kw": "anonymous", "by_men_per_kw": "Men", "by_women_per_kw": "Women"}


def origin_of(place: str) -> str:
    if place in ("ישראל", "ארץ ישראל"):
        return "Israel"
    if place in FSU:
        return "Former Soviet Union"
    if place in MENA:
        return "Middle East and North Africa"
    if place == "אתיופיה":
        return "Ethiopia"
    return "Europe and the Americas" if place else ""


def load_members() -> pd.DataFrame:
    lines = MEMBERS.read_text(encoding="utf-8").splitlines()
    m = pd.DataFrame([json.loads(line) for line in lines if line.strip()])
    members = pd.DataFrame({
        "speaker_id": m["person_id"].astype(str),
        "gender": m["gender"].map(GENDER).fillna(""),
        "group": m["nationality"].map(GROUP).fillna(""),
        "religiosity": m["religious_orientation"].map(RELIGIOSITY).fillna(""),
        "origin": m["place_of_birth"].fillna("").map(origin_of),
    })
    new = pd.read_csv(NEW_MEMBERS, dtype=str).fillna("")[list(members.columns)]
    return pd.concat([members, new], ignore_index=True)


def speeches(keys: list[str]) -> pd.DataFrame:
    """Members' plenary speeches without chair units, with first and last turn and word count."""
    u = pd.read_parquet(UNITS, columns=[*keys, "first_turn", "last_turn", "n_words", "is_chairman"])
    u = u[~u["is_chairman"]]
    return u.groupby(keys, as_index=False).agg(first_turn=("first_turn", "min"), last_turn=("last_turn", "max"),
                                               n_words=("n_words", "sum"))


def presiding_keys() -> set[tuple[str, str]]:
    """(protocol, speaker) for members who presided in that sitting. All their turns that day count as chairing."""
    u = pd.read_parquet(UNITS, columns=["protocol_name", "speaker_id", "is_chairman"])
    chair = u[u["is_chairman"]]
    return set(zip(chair["protocol_name"], chair["speaker_id"]))


def real_heckles(d: pd.DataFrame, chair: set[tuple[str, str]]) -> pd.DataFrame:
    """Heckles at a man or woman by a named member not presiding that day."""
    d = d[d["heckler_named"] & d["target_gender"].isin(["Men", "Women"])]
    return d.loc[[(p, h) not in chair for p, h in zip(d["protocol_name"], d["heckler_id"])]]


def speech_share(members: pd.DataFrame) -> pd.DataFrame:
    """Per Knesset, women's share of the members who spoke and of the sentences."""
    u = pd.read_parquet(UNITS, columns=["speaker_id", "knesset", "n_sentences", "is_chairman"])
    u = u[~u["is_chairman"]].copy()
    u["gender"] = u["speaker_id"].map(members.set_index("speaker_id")["gender"]).fillna("")
    rows = []
    for k, d in u.groupby(S.periods(u["knesset"], load_settings())):
        women = d[d["gender"] == "Women"]
        speakers, n_women = d["speaker_id"].nunique(), women["speaker_id"].nunique()
        rows.append({"knesset": k, "speakers": speakers, "women": n_women, "share_of_speakers": n_women / speakers,
                     "share_of_sentences": women["n_sentences"].sum() / d["n_sentences"].sum()})
    return pd.DataFrame(rows)


def _inside(shard: Path, spans: pd.DataFrame) -> pd.DataFrame:
    """A shard's turns that fall inside someone else's speech."""
    t = pq.read_table(shard, columns=["protocol_name", "speaker_id", "is_valid_speaker", "is_chairman", "turn"])
    j = t.to_pandas().drop_duplicates(["protocol_name", "turn"]).merge(spans, on="protocol_name")
    return j[(j["turn"] > j["first_turn"]) & (j["turn"] < j["last_turn"]) & (j["speaker_id"] != j["speaker"])]


def _rates(d: pd.DataFrame) -> pd.Series:
    words = d["n_words"].sum()
    per_kw = {name: 1000 * d[col].sum() / words for name, col in RATES.items()}
    return pd.Series({"speeches": len(d), "words": int(words), **per_kw,
                      "share_interrupted": float((d["all"] > 0).mean())})


def interruptions(members: pd.DataFrame) -> pd.DataFrame:
    """Interruptions received per 1,000 words, by Knesset and the speaker's gender."""
    gender = members.set_index("speaker_id")["gender"]
    # coalition in the keys drops speeches with no side recorded, as in the published figures
    sp = speeches(["speech_id", "protocol_name", "speaker_id", "knesset", "coalition"])
    sp["speaker_gender"] = sp["speaker_id"].map(gender).fillna("")
    spans = sp[["speech_id", "protocol_name", "speaker_id", "first_turn", "last_turn"]]
    spans = spans.rename(columns={"speaker_id": "speaker"})
    inside = pd.concat([_inside(shard, spans) for shard in sorted(SENTENCES.glob("shard_*.parquet"))])
    member_or_anonymous = np.where(inside["is_valid_speaker"], "member", "anonymous")
    inside["kind"] = np.where(inside["is_chairman"], "chair", member_or_anonymous)
    inside["interrupter_gender"] = inside["speaker_id"].map(gender).fillna("")
    kinds = inside.groupby(["speech_id", "kind"]).size().unstack(fill_value=0)
    by_members = inside[inside["kind"] == "member"].groupby(["speech_id", "interrupter_gender"]).size()
    s = (sp.set_index("speech_id")
         .join(kinds.reindex(columns=["member", "chair", "anonymous"], fill_value=0))
         .join(by_members.unstack(fill_value=0).reindex(columns=["Men", "Women"], fill_value=0))
         .fillna(0))
    s = s[s["speaker_gender"].isin(["Men", "Women"]) & (s["n_words"] >= 100)].copy()
    s["all"] = s["member"] + s["chair"] + s["anonymous"]
    s["knesset"] = S.periods(s["knesset"], load_settings())
    return s.groupby(["knesset", "speaker_gender"]).apply(_rates, include_groups=False).reset_index()


def heckle_tables() -> pd.DataFrame:
    """Heckle shares by the target's gender, weighted back to the population."""
    flags = S.flags("heckle")
    items = list(flags.columns)
    columns = ["unit_id", "protocol_name", "heckler_id", "heckler_named", "target_gender", "heckler_gender", "knesset",
               "year", "sample_weight"]
    d = pd.read_parquet(HECKLES, columns=columns).merge(flags.reset_index(), on="unit_id")
    d = real_heckles(d, presiding_keys())
    d["era"] = np.where(d["knesset"] >= 20, "2015–24", "1992–2015")

    def rates(x: pd.DataFrame) -> pd.Series:
        shares = {i: float(np.average(x[i] >= 0.5, weights=x["sample_weight"])) for i in items}
        return pd.Series({**shares, "n_scored": len(x), "n_weighted": float(x["sample_weight"].sum())})

    named = d[d["heckler_gender"].isin(["Men", "Women"])]
    facets = (("target", d, ["target_gender"]), ("target_x_heckler", named, ["target_gender", "heckler_gender"]),
              ("era", d, ["era", "target_gender"]))
    parts = [x.groupby(by).apply(rates, include_groups=False).reset_index().assign(facet=facet)
             for facet, x, by in facets]
    per_knesset = d.groupby(["knesset", "target_gender"]).apply(rates, include_groups=False)
    knesset_weight = d.groupby("knesset")["sample_weight"].sum()
    # Same Knesset weights for both genders. Women spoke more after 2015, when heckling grew harsher.
    standardised = []
    for gender in ("Men", "Women"):
        x = per_knesset.xs(gender, level="target_gender")
        row = {i: float(np.average(x[i], weights=knesset_weight.loc[x.index])) for i in items}
        row.update({"target_gender": gender, "n_scored": int(x["n_scored"].sum()),
                    "n_weighted": float(x["n_weighted"].sum()), "facet": "standardised"})
        standardised.append(row)
    return pd.concat(parts + [pd.DataFrame(standardised)], ignore_index=True)


def gendered_heckles() -> pd.DataFrame:
    """The gendered question asked again, as Jev took Hebrew's grammatical gender for remarks about gender."""
    a = S.query("select unit_id, noul from answers where stage = 'heckle2'")
    d = pd.read_parquet(HECKLES_V2, columns=["unit_id", "knesset", "target_gender", "sample_weight"])
    d = d.merge(a, on="unit_id")
    d["flag"] = d["noul"] >= 0.5
    per_knesset = d.groupby(["knesset", "target_gender"]).apply(
        lambda x: np.average(x["flag"], weights=x["sample_weight"]), include_groups=False).unstack().dropna()
    knesset_weight = d.groupby("knesset")["sample_weight"].sum().loc[per_knesset.index]
    hand = pd.read_csv(HANDCHECK)
    rows = []
    for gender in ("Men", "Women"):
        label = hand.loc[hand["target_gender"] == gender, "hand_label"]
        strict = float(label.isin(["explicit", "insult"]).mean())
        broad = float(label.isin(["explicit", "insult", "trope", "family"]).mean())
        flagged = float(np.average(per_knesset[gender], weights=knesset_weight))
        scored = d[d["target_gender"] == gender]
        rows.append({"target_gender": gender, "n_scored": len(scored), "n_flagged": int(scored["flag"].sum()),
                     "flagged": flagged, "held_up_strict": strict, "held_up_broad": broad,
                     "confirmed_strict": flagged * strict, "confirmed_broad": flagged * broad})
    return pd.DataFrame(rows)


def build_tables(tables_dir: Path) -> list[Path]:
    members = load_members()
    spoke = pd.read_parquet(UNITS, columns=["speaker_id"])["speaker_id"].unique()
    return S.write_tables({"members_demographics": members[members["speaker_id"].isin(spoke)],
                           "women_share": speech_share(members), "interruptions": interruptions(members),
                           "heckles": heckle_tables(), "heckles_gendered": gendered_heckles()}, tables_dir)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m knesset_ches.demographics", description=__doc__.split("\n\n")[0])
    ap.add_argument("--tables", type=Path, default=S.TABLES)
    for path in build_tables(ap.parse_args(argv).tables):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
