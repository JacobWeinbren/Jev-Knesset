"""The extend parser, offline."""
from __future__ import annotations

import datetime as dt

from knesset_ches import extend as E

PROTOCOL = "\n".join([
    "תוכן העניינים",
    "היו\"ר אמיר אוחנה:\t9",
    " << נושא >> דברי יושב-ראש הכנסת << נושא >>   ",
    " << יור >> היו\"ר אמיר אוחנה: << יור >>   ",
    "חברות וחברי הכנסת, אני מתכבד לפתוח את ישיבת הכנסת.",
    "הצבעה מס' 1",
    "",
    "בעד ההצעה להעביר את הצעת החוק לוועדה – 49",
    "נגד – ",
    "ההצעה נתקבלה.",
    "",
    "בעד – 49, אין נמנעים.",                              # the chair reading the result out is speech
    " << הצח >> הצעת חוק האקלים, התשפ\"ד–2024 << הצח >> ",
    "[רשומות (הצעות חוק, חוב' מ/1735).]",
    "(קריאה ראשונה)",
    "1. כישלון הממשלה;",
    "חברי הכנסת, נעבור לנושא הבא על סדר-היום.",          # the chair goes on after a heading
    " דובר >> אחמד טיבי (חד\"ש-תע\"ל): << דובר >>   ",
    "(מחיאות כפיים)",
    "תודה.",
    " << הצח >> הצעת חוק שירות אזרחי, התשפ\"ד–2024 << הצח >> ",
    "אני מזמין את חבר הכנסת קריב.",                      # dropped: the turn before was not the chair's
    "<< קריאה >> קריאות: << קריאה >>",
    "בושה, בושה",
    " << קריאה >> טלי גוטליב (הליכוד): << קריאה >> ",
    "מיכאל, מיכאל.  << קריאה >> משה סעדה (הליכוד): << קריאה >> ",
    "מיכאל, אנחנו איתך - - -",
])


def test_turns_keep_speech_and_drop_headings_votes_and_stage_directions():
    assert [(t.kind, t.line, t.lines) for t in E.split_turns(PROTOCOL)] == [
        ("chair", 'היו"ר אמיר אוחנה:', ["חברות וחברי הכנסת, אני מתכבד לפתוח את ישיבת הכנסת.", "בעד – 49, אין נמנעים.",
                                         "חברי הכנסת, נעבור לנושא הבא על סדר-היום."]),
        ("speaker", 'אחמד טיבי (חד"ש-תע"ל):', ["תודה."]),
        ("interjection", "קריאות:", ["בושה, בושה"]),
        ("interjection", "טלי גוטליב (הליכוד):", ["מיכאל, מיכאל."]),
        ("interjection", "משה סעדה (הליכוד):", ["מיכאל, אנחנו איתך - - -"]),
    ]


def test_sentences_split_on_stops_and_keep_question_marks_quotes_and_decimals():
    line = 'אני שמחה – באמת קשה לומר "שמחה" בימים אלה. מה? התשפ"ד–2024 עבר!'
    assert E.sentences_of(line) == ['אני שמחה – באמת קשה לומר "שמחה" בימים אלה', "מה?", 'התשפ"ד–2024 עבר!']
    assert E.sentences_of('הוא אמר: "שלום." ואז הלך.') == ['הוא אמר: "שלום"', "ואז הלך"]
    assert E.sentences_of("עלה ב-2.5 אחוזים – – –") == ["עלה ב-2.5 אחוזים - - -"]


def person(pid: str, first: str, last: str, faction: int = 1096) -> E.Person:
    p = E.Person(pid, f"{first} {last}", True)
    p.variants, p.last_keys, p.first_keys = E._name_variants(first, last)
    p.spells = [(dt.date(2022, 11, 15), E.FAR_FUTURE)]
    p.memberships = [(dt.date(2022, 11, 15), E.FAR_FUTURE, faction)]
    return p


def test_speakers_are_matched_through_titles_nicknames_and_typos():
    people = [person("30300", "אמיר", "אוחנה"), person("30846", "יצחק", "גולדקנופף", 1101),
              person("30782", "טטיאנה (טניה)", "מזרסקי", 1102), person("30799", "מישל", "בוסקילה", 1108)]
    resolver = E.Resolver({p.person_id: p for p in people}, {})
    day = dt.date(2025, 1, 6)
    chair = resolver.resolve('היו"ר אמיר אוחנה:', "chair", day)
    assert (chair.speaker_id, chair.is_chair) == ("30300", True)
    assert resolver.resolve("שר הבינוי והשיכון יצחק גולדקנופ:", "speaker", day).speaker_id == "30846"
    assert resolver.resolve("טניה מזרסקי (יש עתיד):", "interjection", day).speaker_id == "30782"
    assert resolver.resolve('היו"ר שמעון בוסקילה:', "chair", day).speaker_id == "30799"
    heckle = resolver.resolve("קריאות:", "interjection", day)
    assert (heckle.is_valid, heckle.is_mk) == (False, False)
    assert heckle.speaker_id == resolver.resolve("קריאות:", "interjection", dt.date(2026, 1, 1)).speaker_id
