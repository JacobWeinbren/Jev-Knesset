"""blog_style: saving, the layout checks and the sources line."""
from __future__ import annotations

import pandas as pd
import pytest

from knesset_ches import blog_style as B


@pytest.fixture(autouse=True)
def _tmp_out(tmp_path, monkeypatch):
    monkeypatch.setattr(B, "OUT", tmp_path / "blog")


def page(headline: str = "A short headline") -> B.BlogPage:
    return B.BlogPage(headline, "", B.sources("A source"), body=1.5)


def test_a_page_saves_png_webp_and_alt() -> None:
    p = page()
    p.axes(p.M, p.top, p.inner, 1.2).plot([0, 1], [0, 1], color=B.INK)
    assert [path.suffix for path in p.save("unit_test_chart", alt="The finding.")] == [".png", ".webp"]
    assert (B.OUT / "unit_test_chart.alt.txt").read_text().strip() == "The finding."


def test_colliding_or_off_canvas_text_and_long_headlines_are_refused() -> None:
    p = page()
    p.text(1.0, p.top + 0.5, "one label")
    p.text(1.05, p.top + 0.5, "another label")
    with pytest.raises(ValueError, match="collides"):
        p.save("unit_test_collision")
    p = page()
    p.text(p.W - 0.1, p.top + 0.5, "runs off the right edge")
    with pytest.raises(ValueError, match="off the canvas"):
        p.save("unit_test_edge")
    with pytest.raises(ValueError, match="headline"):
        page("word " * 60)


def test_sources_and_sentence_counts(monkeypatch) -> None:
    assert B.sources("A", "B") == "Sources: A; B"
    assert B.sources("A", partial=True).endswith("*2026: January to July")
    assert B.from_sentences(63_142) == "from 63,000 sentences"
    assert B.from_sentences(2_400_000) == "from 2.4 million sentences"
    packs = pd.DataFrame({"unit_id": ["a", "b", "c", "d"], "speaker_id": ["1", "1", "2", "1"],
                          "knesset": [21, 23, 23, 24], "faction_id": ["5", "5", "6", "7"],
                          "n_sentences": [10, 20, 40, 80]})
    monkeypatch.setattr(B, "packs", lambda: packs)
    monkeypatch.setattr(B, "usable_units", lambda dimensions: {"a", "b", "c"})
    assert B.sentences_behind(["lrgen"], [("1", "21-22-23"), (1, "21-22-23")]) == 30   # each pack once
    assert B.sentences_behind(["lrgen"], [("1", "24", "7"), ("2", "23", "5")]) == 0
