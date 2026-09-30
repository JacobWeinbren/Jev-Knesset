"""knesset_ches.tables on small hand-made tables."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from knesset_ches import tables as T


@pytest.fixture
def tables(tmp_path) -> T.Tables:
    pd.DataFrame({"period": ["25", "25"], "actor_id": ["31", "35"]}).to_csv(tmp_path / "group_scores.csv", index=False)
    mk = pd.DataFrame({"dimension": "lrgen", "period_type": "knesset", "period": "25", "period_start": "2022-11-15",
                       "speaker_id": [1, 2, 3, 4], "speaker_name": "x", "faction_id": [1, 1, 2, 2], "faction_name": "x",
                       "bloc": ["Right", "Orthodox", "Left", "Arab-Israeli"], "estimate": [8.0, 9.0, 2.0, 1.0],
                       "sufficient": [True, False, True, True], "n_speeches": 10})
    mk.to_csv(tmp_path / "mk_scores.csv", index=False)
    points = pd.DataFrame({"dimension": ["lrgen"] * 8 + ["galtan"] * 7, "bloc": "Right", "jev": 1.0, "ches": 2.0})
    points.to_csv(tmp_path / "validation_points.csv", index=False)
    pd.DataFrame({"dimension": ["lrgen"]}).to_csv(tmp_path / "validation_summary.csv", index=False)
    return T.load_tables(tmp_path)


def test_load_tables_reads_ids_as_text(tables) -> None:
    assert tables.group["actor_id"].tolist() == ["31", "35"] and tables.group["period"].tolist() == ["25", "25"]


def test_chamber_draws_every_member_at_their_own_score(tables) -> None:
    rows = T.prep_chamber(tables, "lrgen", "knesset", ["25"])
    assert list(rows["value"]) == [8.0, 9.0, 2.0, 1.0]
    assert list(rows["thin"]) == [False, True, False, False]
    med = T.chamber_medians(rows).iloc[0]
    assert (med["median_a"], med["median_b"], med["gap"]) == (8.5, 1.5, 7.0)


def test_validation_needs_eight_parties_a_scale(tables) -> None:
    assert set(T.prep_validation(tables)["dimension"]) == {"lrgen"}


def test_validation_r() -> None:
    points = pd.DataFrame({"dimension": "lrgen", "jev": [1.0, 2.0, 3.0], "ches": [9.0, 6.0, 3.0]})
    assert T.validation_r(points)["r"].iloc[0] == pytest.approx(-1.0)
    assert np.isnan(T.validation_r(points.head(2))["r"].iloc[0])


def test_names_and_numbers() -> None:
    assert T.display_name("Labor") == "Labour" and T.display_name("Likud") == "Likud"
    assert T.display_name("NU / RZ", 2024) == "Religious Zionism"
    assert T.display_name("NU / RZ", 2006) == "National Union"
    assert T.round_count(194_388) == "194,000" and T.round_count(1_234_567) == "1.2m" and T.round_count(843) == "840"
    assert T.share_words(0.25) == "a quarter" and T.share_words(0.44) == "nearly half"
    assert T.pct(0.254) == "25%" and T.number_word(7) == "seven"
