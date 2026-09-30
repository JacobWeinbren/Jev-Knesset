"""Heckle samples, on hand-made rows."""
from __future__ import annotations

import pandas as pd

from knesset_ches import demographics as D
from knesset_ches import samples


def test_pairs_keep_turns_inside_a_speech_by_someone_else() -> None:
    turns = pd.DataFrame({"turn": [1, 2, 3, 4], "speaker_id": ["a", "x", "a", "b"], "named": [True, False, True, True],
                          "text": ["t1", "t2", "t3", "t4"], "context": ["", "t1", "t2", "t3"]})
    speeches = pd.DataFrame({"speech_id": ["s1", "s2"], "speaker_id": ["a", "b"], "first_turn": [0, 2],
                             "last_turn": [4, 5], "target_gender": ["Women", "Men"], "knesset": 25, "year": 2024,
                             "n_words": [300, 200]})
    p = samples._pairs("p1", turns, speeches)
    assert list(zip(p["turn"], p["speech_id"])) == [(2, "s1"), (3, "s2")]
    assert list(p["target_gender"]) == ["Women", "Men"]


def test_heckle_sample_keeps_women_and_a_third_of_men_each_year() -> None:
    d = pd.DataFrame({"target_gender": ["Women"] * 2 + ["Men"] * 6, "year": [2020, 2021] + [2020] * 3 + [2021] * 3})
    s = samples.heckle_sample(d, seed=1)
    assert (s["target_gender"] == "Women").sum() == 2
    assert s.loc[s["target_gender"] == "Men", "year"].tolist() == [2020, 2021]
    assert s.groupby("target_gender")["sample_weight"].first().to_dict() == {"Men": 3.0, "Women": 1.0}


def test_gendered_sample_weights_back_to_the_population() -> None:
    d = pd.DataFrame({"target_gender": ["Men"] * 8 + ["Women"] * 4})
    s = samples.gendered_sample(d, pd.Series({"Men": 0.25, "Women": 0.5}))
    assert s["target_gender"].value_counts().to_dict() == {"Men": 2, "Women": 2}
    assert s.groupby("target_gender")["sample_weight"].first().to_dict() == {"Men": 4.0, "Women": 2.0}


def test_real_heckles_drop_anonymous_presiding_and_untargeted() -> None:
    d = pd.DataFrame({"protocol_name": "p1", "heckler_id": ["1", "2", "3", "4"],
                      "heckler_named": [True, False, True, True], "target_gender": ["Women", "Men", "Men", ""]})
    assert D.real_heckles(d, {("p1", "3")})["heckler_id"].tolist() == ["1"]
