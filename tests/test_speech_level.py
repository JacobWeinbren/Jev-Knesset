"""The Netanyahu and attack tables, on hand-made speeches."""
from __future__ import annotations

import pandas as pd
import pytest

from knesset_ches import speech_level as S


def test_netanyahu_tables_by_party_and_year() -> None:
    n = 50
    d = pd.DataFrame({"party": ["Labour"] * n + ["Likud"] * n + [""] * n,
                      "bloc": ["Left"] * n + ["Right"] * n + [""] * n,
                      "knesset": 25, "year": 2024, "n_words": 100,
                      "bibi_attack": [0.9] * n + [0.1] * 2 * n, "bibi_defend": [0.0] * n + [0.8] * n + [0.0] * n,
                      "bibi_must_go": 0.0})
    by_party, by_year = S.netanyahu_tables(d)
    assert list(by_party["party"]) == ["Labour", "Likud"]
    assert list(by_party["in_camp"]) == [False, True]
    assert list(by_party["share_attack"]) == [1.0, 0.0]
    assert by_year.iloc[0].to_dict() == pytest.approx({"year": 2024, "share_attack": 1 / 3, "share_defend": 1 / 3,
                                                       "share_must_go": 0.0, "n_speeches": 150})


def test_attacks_on_arab_citizens_by_quarter() -> None:
    df = pd.DataFrame({"protocol_type": "plenary", "date": ["2024-01-10", "2024-02-10", "2024-05-01"],
                       "bloc": ["Right", "Left", "Sectoral"], "attacks_arab_citizens": [0.9, 0.1, 0.7]})
    t = S.attacks_on_arab_citizens(df).set_index(["quarter", "group"])
    assert t.loc[("2024Q1", "All members"), "share"] == 0.5
    assert t.loc[("2024Q1", "Right and Orthodox"), "n_attacks"] == 1
    assert ("2024Q2", "Everyone else") not in t.index
