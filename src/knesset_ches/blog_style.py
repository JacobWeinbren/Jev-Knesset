"""House style for the blog charts, and a page that refuses to save with colliding text.

    page = BlogPage("Headline", "Subtitle", sources("..."))
    ax = page.axes(page.M, page.top, page.inner, 3.0)      # inches from the top-left corner
    page.save("slug", alt="Alt text.")                     # png, webp and alt text
"""
from __future__ import annotations

import functools
import logging
import math
import re
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import matplotlib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager, patheffects
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
from matplotlib.text import Text
from PIL import Image

from knesset_ches.tables import ROOT, TABLES, round_count

OUT = ROOT / "outputs" / "charts" / "blog"
FONTS = ROOT / "assets" / "fonts"
DPI = 240

PAPER, INK, SEC, CAP, GRID, RULE = "#fbf6f1", "#1a1816", "#4d4842", "#6b645c", "#ebe3d9", "#d6ccbf"
# this order passes a colour-blindness check
BLOCS = [("Left", "#d81e3f"), ("Secular Centre", "#e0a417"), ("Arab-Israeli", "#238a68"), ("Right", "#3f46b0"),
         ("Orthodox", "#e06a1b")]
BLOC = dict(BLOCS)
ROW_ORDER = ["Right", "Orthodox", "Secular Centre", "Left", "Arab-Israeli", "Sectoral"]
OTHER = "#a39a8e"                      # Sectoral and anything else
CONTEXT = "#c9bfb2"
SHADE = "#f2eae0"
# non-bloc colours, one role each
TEAL, PLUM, SLATE, CLAY, MAUVE, OLIVE, ROSE, BRASS, SAGE = ("#1f7a8c", "#7b4b94", "#4f6d7a", "#b5836d", "#8a7aa8",
                                                            "#7f8c5a", "#b06a7a", "#a68a3a", "#6f9a7c")
GOV, OPP = TEAL, PLUM                  # government dashed with circles, opposition solid with squares
COURTS, CONFLICT, ECONOMY = CLAY, SLATE, SAGE
PLENUM, COMMITTEE = INK, BRASS         # plenum solid, committee dashed
WOMEN, MEN = ROSE, INK                 # women filled markers, men hollow
VOTERS = INK                           # dashed, hollow dots
SECULAR, RELIGIOUS, HAREDI = OLIVE, BRASS, MAUVE     # circle, square, diamond

# type scale in pt
T_HEAD, T_SUB, T_KEY, T_PANEL, T_LABEL = 18, 12, 10.5, 12.5, 11.5
T_VALUE, T_TICK, T_NOTE, T_COUNT, T_FOOT = 11, 10.5, 10.5, 10, 9
LW_DATA, LW_STORY, LW_CONTEXT, DOT_S = 2.2, 2.8, 1.1, 42
HALO = [patheffects.withStroke(linewidth=3, foreground=PAPER)]
KEY_LH = 0.24                          # inches
OVERLAP_PX = 1.0                       # overlap allowed before texts collide
MIN_GAP_IN = 0.04                      # smallest gap between texts side by side

SRC_CORPUS = "Knesset Corpus (Goldin et al. 2025), extended to July 2026 from the Knesset's protocols"
SRC_CORPUS_SHORT = "Knesset Corpus (Goldin et al. 2025), extended to July 2026"
SRC_CHES = "Chapel Hill Expert Survey Israel 2021–22 (Zur and Bakker)"
SRC_TONE = "Goldin, Rabinovich and Wintner (2025)"
SRC_DELEG = "Rivlin-Angert and Mor-Lan (2025)"
PARTIAL = "*2026: January to July"

KNESSET_SLOTS = ["13", "14", "15", "16", "17", "18", "19", "20", "21-22-23", "24", "25"]
KNESSET_YEARS = {"13": "1992–96", "14": "1996–99", "15": "1999–2003", "16": "2003–06", "17": "2006–09", "18": "2009–13",
                 "19": "2013–15", "20": "2015–19", "21-22-23": "2019–21", "24": "2021–22", "25": "2022–26*"}

matplotlib.use("Agg")
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)     # weights 500 and 600 fall back quietly
for _font in sorted(FONTS.rglob("*.ttf")):
    font_manager.fontManager.addfont(str(_font))
_INSTALLED = {f.name for f in font_manager.fontManager.ttflist}


def _installed(names: Sequence[str]) -> list[str]:
    return [n for n in names if n in _INSTALLED]


# system faces fill in missing glyphs, then Hebrew
_HEBREW = _installed(["Arial Hebrew", "Times New Roman", "Arial", "DejaVu Sans"])
SERIF = ["Newsreader", *_installed(["Iowan Old Style", "Georgia", "Palatino", "Times New Roman", "DejaVu Serif"])[:1],
         *_HEBREW]
SANS = ["Instrument Sans", *_installed(["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"])[:1], *_HEBREW]
plt.rcParams.update({"font.family": SANS, "text.color": INK, "axes.edgecolor": INK, "figure.facecolor": PAPER,
                     "axes.facecolor": PAPER, "savefig.facecolor": PAPER, "axes.unicode_minus": False,
                     "figure.max_open_warning": 0})

_HEB = re.compile(r"[֐-׿]")
_LTR_RUN = re.compile(r"[A-Za-z0-9][A-Za-z0-9.,:%/\-]*[A-Za-z0-9]|[A-Za-z0-9]")
_MIRROR = str.maketrans("()[]{}<>", ")(][}{><")
_MARKERS = {"dot": "o", "hollow": "o", "square": "s", "diamond": "D"}


def visual(text: Any) -> str:
    """Hebrew in visual order."""
    s = "" if text is None or (isinstance(text, float) and math.isnan(text)) else str(text)
    if not _HEB.search(s):
        return s
    return _LTR_RUN.sub(lambda m: m.group(0)[::-1], s[::-1].translate(_MIRROR))


def sources(*parts: str, partial: bool = False) -> str:
    """'Sources: A; B; C', with the 2026 note if `partial`."""
    s = "Sources: " + "; ".join(p for p in parts if p)
    return s + (". " + PARTIAL if partial else "")


def _luminance(colour: Any) -> float:
    ch = [v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4 for v in mcolors.to_rgb(colour)]
    return 0.2126 * ch[0] + 0.7152 * ch[1] + 0.0722 * ch[2]


def contrast(a: Any, b: Any) -> float:
    """WCAG contrast ratio."""
    dark, light = sorted((_luminance(a), _luminance(b)))
    return (light + 0.05) / (dark + 0.05)


class BlogPage:
    """One blog image. Positions are in inches from the top-left corner."""

    def __init__(self, headline: str, subtitle: str, sources: str, width: float = 6.0,
                 key: Sequence[tuple] | None = None, body: float = 3.4, max_sub_lines: int = 2,
                 key_lead: str | None = None, key_spacing: tuple[float, float] = (0.17, 0.28)):
        self.W, self.M = float(width), 0.30
        self.fig = plt.figure(figsize=(self.W, 8.0), facecolor=PAPER)
        self.fig.set_dpi(DPI)
        self.bg = self.fig.add_axes([0, 0, 1, 1], facecolor="none")
        self.bg.axis("off")
        self._axes: list[tuple[Any, tuple[float, float, float, float]]] = []
        self._renderer = self.fig.canvas.get_renderer()
        self.head_lines = self._lines("headline", headline, T_HEAD, 2, serif=True, weight=600)
        self.sub_lines = self._lines("subtitle", subtitle.rstrip(". "), T_SUB, max_sub_lines, serif=True)
        self.foot_lines = self._lines("sources footer", sources, T_FOOT, 2)
        self.key_lead = key_lead
        self.mark_w, self.key_gap = key_spacing
        self.key_rows = self._key_rows(key) if key else []

        head_lh, sub_lh, self.foot_lh = T_HEAD * 1.15 / 72, T_SUB * 1.3 / 72, T_FOOT * 1.3 / 72
        y = 0.28
        self.head_y = [y + i * head_lh for i in range(len(self.head_lines))]
        y += len(self.head_lines) * head_lh + 0.10
        self.sub_y = [y + i * sub_lh for i in range(len(self.sub_lines))]
        y += len(self.sub_lines) * sub_lh
        self.key_y = None
        if self.key_rows:
            y += 0.12 + T_KEY * 0.8 / 72
            self.key_y = y
            y += T_KEY * 0.3 / 72
        self.top = y + 0.22 + KEY_LH * max(0, len(self.key_rows) - 1)
        self.foot_h = 0.22 + 0.07 + len(self.foot_lines) * self.foot_lh + 0.14
        self.set_body(body)

    @property
    def inner(self) -> float:
        return self.W - 2 * self.M

    def _lines(self, what: str, s: str, size: float, most: int, **kw: Any) -> list[str]:
        lines = self.wrap(s, size, **kw)
        if len(lines) > most:
            raise ValueError(f"{what} runs to {len(lines)} lines (max {most}) at {self.W} in: {s!r}")
        return lines

    def set_body(self, body: float) -> None:
        """Body height in inches."""
        self.body = float(body)
        self.bottom = self.top + self.body
        self.H = self.bottom + self.foot_h
        self.fig.set_size_inches(self.W, self.H)
        self.bg.set_xlim(0, self.W)
        self.bg.set_ylim(self.H, 0)
        for ax, (left, top, width, height) in self._axes:
            ax.set_position([left / self.W, 1 - (top + height) / self.H, width / self.W, height / self.H])

    def fit(self, lowest: float) -> None:
        """End the body at `lowest` inches from the top."""
        self.set_body(max(0.5, lowest - self.top))

    def axes(self, left: float, top: float, width: float, height: float, **kw: Any):
        """Axes placed in inches. They stay put when the page grows."""
        ax = self.fig.add_axes([left / self.W, 1 - (top + height) / self.H, width / self.W, height / self.H], **kw)
        ax.set_facecolor("none")
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(length=0, labelsize=T_TICK, labelcolor=CAP)
        for lab in ax.get_xticklabels() + ax.get_yticklabels():
            lab.set_fontfamily(SANS)
        self._axes.append((ax, (left, top, width, height)))
        return ax

    def text(self, x: float, y: float, s: str, size: float = T_LABEL, colour: str = INK, ha: str = "left",
             va: str = "baseline", weight: int = 400, serif: bool = False, italic: bool = False, halo: bool = False,
             **kw: Any) -> Text:
        if halo:
            kw["path_effects"] = HALO
        return self.bg.text(x, y, visual(s), fontsize=size, color=colour, ha=ha, va=va, weight=weight,
                            family=SERIF if serif else SANS, style="italic" if italic else "normal", **kw)

    def note(self, x: float, y: float, s: str, **kw: Any) -> Text:
        return self.text(x, y, s, size=T_NOTE, colour=SEC, serif=True, italic=True, **kw)

    def width_of(self, s: str | Text, size: float = T_LABEL, serif: bool = False, weight: int = 400,
                 italic: bool = False) -> float:
        """Rendered width in inches."""
        if isinstance(s, Text):
            return s.get_window_extent(self._renderer).width / self.fig.dpi
        t = self.bg.text(0, 0, visual(s), fontsize=size, family=SERIF if serif else SANS, weight=weight,
                         style="italic" if italic else "normal")
        w = t.get_window_extent(self._renderer).width / self.fig.dpi
        t.remove()
        return w

    def wrap(self, s: str, size: float, serif: bool = False, weight: int = 400, width: float | None = None,
             italic: bool = False) -> list[str]:
        """Greedy wrap by measured width. A no-break space keeps two words together."""
        width = width or self.inner
        lines, cur = [], ""
        for w in s.split(" "):
            if not w.strip():
                continue
            trial = f"{cur} {w}".strip()
            if cur and self.width_of(trial, size, serif, weight, italic) > width:
                lines.append(cur)
                cur = w
            else:
                cur = trial
        if cur:
            lines.append(cur)
        return lines

    def _mark_width(self, shape: str) -> float:
        return {"line": 0.33, "dash": 0.33, "left": 0.19, "right": 0.19}.get(shape, self.mark_w)

    def _key_rows(self, items: Sequence[tuple]) -> list[list[tuple]]:
        rows: list[list[tuple]] = [[]]
        x = self.width_of(self.key_lead, T_KEY) + 0.12 if self.key_lead else 0.0
        for item in items:
            w = self._mark_width(item[2] if len(item) > 2 else "dot") + self.width_of(item[0], T_KEY)
            if rows[-1] and x + w > self.inner:
                rows.append([])
                x = 0.0
            rows[-1].append(item)
            x += w + self.key_gap
        return rows

    def _key_mark(self, x: float, cy: float, colour: Any, shape: str) -> None:
        if shape in ("line", "dash"):
            self.bg.add_line(Line2D([x, x + 0.26], [cy, cy], color=colour, lw=LW_DATA,
                                    ls="--" if shape == "dash" else "-", solid_capstyle="round"))
        elif shape in ("left", "right"):
            tip, back = (x, x + 0.11) if shape == "left" else (x + 0.11, x)
            self.bg.add_patch(Polygon([(tip, cy), (back, cy - 0.055), (back, cy + 0.055)], closed=True, fc=colour,
                                      ec="none", zorder=3))
        elif shape == "triangle":
            self.bg.scatter([x + 0.06], [cy + 0.01], s=48, marker="^", facecolor=colour, edgecolor=colour,
                            linewidths=0, zorder=3)
        else:
            self.bg.scatter([x + 0.06], [cy], s=DOT_S * 0.9, marker=_MARKERS.get(shape, "o"),
                            facecolor=PAPER if shape == "hollow" else colour, edgecolor=colour, linewidths=1.4,
                            zorder=3)

    def _draw_key(self) -> None:
        for r, row in enumerate(self.key_rows):
            x, y = self.M, self.key_y + r * KEY_LH
            if r == 0 and self.key_lead:
                t = self.text(x, y, self.key_lead, size=T_KEY, colour=SEC)
                x += self.width_of(t) + 0.12
            for label, colour, *shape in row:
                shape = shape[0] if shape else "dot"
                self._key_mark(x, y - T_KEY * 0.33 / 72, colour, shape)
                x += self._mark_width(shape)
                t = self.text(x, y, label, size=T_KEY)
                x += self.width_of(t) + self.key_gap

    def _draw_frame(self) -> None:
        for s, y in zip(self.head_lines, self.head_y):
            self.text(self.M, y, s, size=T_HEAD, weight=600, serif=True, va="top")
        for s, y in zip(self.sub_lines, self.sub_y):
            self.text(self.M, y, s, size=T_SUB, colour=SEC, serif=True, va="top")
        self._draw_key()
        rule_y = self.bottom + 0.22
        self.bg.plot([self.M, self.W - self.M], [rule_y, rule_y], color=RULE, lw=0.9, solid_capstyle="butt")
        for i, s in enumerate(self.foot_lines):
            self.text(self.M, rule_y + 0.07 + i * self.foot_lh, s, size=T_FOOT, colour=CAP, va="top")

    def _texts(self) -> list[Text]:
        """Visible texts, including tick labels inside the axis limits."""
        out: list[Text] = list(self.bg.texts)
        for ax in self.fig.axes:
            if ax is self.bg:
                continue
            out += ax.texts
            lo_x, hi_x = sorted(ax.get_xlim())
            lo_y, hi_y = sorted(ax.get_ylim())
            out += [t for t, v in zip(ax.get_xticklabels(), ax.get_xticks()) if lo_x - 1e-9 <= v <= hi_x + 1e-9]
            out += [t for t, v in zip(ax.get_yticklabels(), ax.get_yticks()) if lo_y - 1e-9 <= v <= hi_y + 1e-9]
            if ax.get_legend() is not None:
                out += ax.get_legend().get_texts()
        out += self.fig.texts
        return [t for t in out if t.get_visible() and str(t.get_text()).strip()]

    def collisions(self) -> list[str]:
        """Texts that overlap, crowd each other, half-cover a bar or leave the canvas."""
        self.fig.canvas.draw()
        r = self.fig.canvas.get_renderer()
        W, H = self.fig.bbox.width, self.fig.bbox.height
        boxes = []
        for t in self._texts():
            b = t.get_window_extent(r)
            pad = 0.18 * b.height          # trim empty space above and below
            boxes.append((t, type(b).from_extents(b.x0, b.y0 + pad, b.x1, b.y1 - pad)))
        problems = []
        for t, b in boxes:
            if b.x0 < -OVERLAP_PX or b.y0 < -OVERLAP_PX or b.x1 > W + OVERLAP_PX or b.y1 > H + OVERLAP_PX:
                problems.append(f"off the canvas: {t.get_text()!r}")
        # text inside a bar is fine, half over one is not. Pale bands don't count.
        shapes = [p.get_window_extent(r) for ax in self.fig.axes for p in ax.patches
                  if p.get_visible() and p.get_facecolor()[3] > 0.2 and contrast(p.get_facecolor(), PAPER) >= 1.35]
        for t, b in boxes:
            cx, cy = (b.x0 + b.x1) / 2, (b.y0 + b.y1) / 2
            for sb in shapes:
                ox = min(b.x1, sb.x1) - max(b.x0, sb.x0)
                oy = min(b.y1, sb.y1) - max(b.y0, sb.y0)
                inside = sb.x0 <= cx <= sb.x1 and sb.y0 <= cy <= sb.y1
                if ox > OVERLAP_PX and oy > OVERLAP_PX and not inside:
                    problems.append(f"text over a bar or shape: {t.get_text()!r}")
                    break
        for i, (ti, bi) in enumerate(boxes):
            for tj, bj in boxes[i + 1:]:
                ox = min(bi.x1, bj.x1) - max(bi.x0, bj.x0)
                oy = min(bi.y1, bj.y1) - max(bi.y0, bj.y0)
                if ox > OVERLAP_PX and oy > OVERLAP_PX:
                    problems.append(f"overlap: {ti.get_text()!r} and {tj.get_text()!r}")
                elif oy > 0.5 * min(bi.height, bj.height) and -MIN_GAP_IN * self.fig.dpi < ox <= OVERLAP_PX:
                    problems.append(f"too close (under {MIN_GAP_IN} in apart): {ti.get_text()!r} and {tj.get_text()!r}")
        return problems

    def save(self, slug: str, variant: str = "", alt: str | None = None) -> list[Path]:
        """Check for collisions, then write the PNG, WebP and alt text."""
        self._draw_frame()
        problems = self.collisions()
        if problems:
            plt.close(self.fig)
            raise ValueError(f"{slug}{variant}: text collides\n  " + "\n  ".join(problems))
        OUT.mkdir(parents=True, exist_ok=True)
        png, webp = OUT / f"{slug}{variant}.png", OUT / f"{slug}{variant}.webp"
        self.fig.savefig(png, dpi=DPI, facecolor=PAPER)
        plt.close(self.fig)
        Image.open(png).convert("RGB").save(webp, "WEBP", quality=90, method=6)
        if alt:
            (OUT / f"{slug}.alt.txt").write_text(alt.strip() + "\n")
        return [png, webp]


def ax_label(ax: Any, x: float, y: float, s: str, size: float = T_LABEL, colour: Any = INK, **kw: Any) -> Text:
    return ax.text(x, y, s, fontsize=size, color=colour, family=SANS, path_effects=HALO, **kw)


def ax_note(ax: Any, x: float, y: float, s: str, colour: Any = SEC, halo: bool = True, **kw: Any) -> Text:
    return ax.text(x, y, s, fontsize=T_NOTE, color=colour, family=SERIF, style="italic",
                   path_effects=HALO if halo else None, **kw)


def spread(ys: Sequence[float], gap: float, lo: float, hi: float) -> list[float]:
    """Push label positions `gap` apart, within lo and hi."""
    order = np.argsort(ys)
    out = [float(ys[i]) for i in order]
    for j in range(1, len(out)):
        out[j] = max(out[j], out[j - 1] + gap)
    shift = max(0.0, out[-1] - hi) if out else 0.0
    out = [v - shift for v in out]
    for j in range(len(out) - 2, -1, -1):
        out[j] = min(out[j], out[j + 1] - gap)
    if out:
        out[0] = max(out[0], lo)
    for j in range(1, len(out)):
        out[j] = max(out[j], out[j - 1] + gap)
    res = [0.0] * len(out)
    for k, i in enumerate(order):
        res[i] = out[k]
    return res


@functools.cache
def packs() -> pd.DataFrame:
    """Plenary and committee packs, without the text."""
    cols = ["unit_id", "speaker_id", "knesset", "faction_id", "n_sentences"]
    p = pd.concat([pd.read_parquet(ROOT / "data" / "corpus" / sub / "packs.parquet", columns=cols)
                   for sub in ("processed", "processed_committee")], ignore_index=True)
    return p.astype({"speaker_id": str, "knesset": int, "faction_id": str})


@functools.cache
def _usable() -> pd.DataFrame:
    us = pd.read_parquet(TABLES / "unit_scores.parquet", columns=["unit_id", "dimension", "usable"])
    return us[us["usable"].fillna(False).astype(bool)]


def usable_units(dimensions: Iterable[str]) -> set:
    us = _usable()
    return set(us.loc[us["dimension"].isin(list(dimensions)), "unit_id"])


def sentences_behind(dimensions: Iterable[str], terms: Iterable[tuple]) -> int:
    """Sentences behind usable scores for (speaker_id, period[, faction_id]) terms. "21-22-23" counts as three."""
    rows = [(str(t[0]), int(k), *(str(x) for x in t[2:])) for t in terms for k in str(t[1]).split("-")]
    if not rows:
        return 0
    columns = ["speaker_id", "knesset", "faction_id"][:len(rows[0])]
    p = packs()
    p = p[p["unit_id"].isin(usable_units(dimensions))]
    return int(p.merge(pd.DataFrame(rows, columns=columns).drop_duplicates(), on=columns)["n_sentences"].sum())


def from_sentences(n: int) -> str:
    """'from 63,000 sentences', 'from 1.2 million sentences'."""
    if n >= 1_000_000:
        return f"from {n / 1e6:.1f} million sentences".replace(".0 million", " million")
    return f"from {round_count(n)} sentences"


def knesset_axis(ax: Any, slots: Sequence[str] = KNESSET_SLOTS) -> dict[str, int]:
    """Knessets as equal slots on x. Returns slot -> x."""
    ax.set_xlim(-0.5, len(slots) - 0.5)
    ticks = [i for i, k in enumerate(slots) if i % 2 == 0 or k == slots[-1]]
    ax.set_xticks(ticks)
    ax.set_xticklabels([KNESSET_YEARS.get(slots[i], slots[i]) for i in ticks], fontsize=T_TICK, color=CAP, family=SANS)
    return {k: i for i, k in enumerate(slots)}


def year_axis(ax: Any) -> None:
    """Years 1992 to 2026 on x, the partial year starred."""
    ax.set_xlim(1992 - 0.6, 2026 + 0.6)
    ticks = [1992, 2000, 2010, 2020, 2026]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{v}*" if v == 2026 else str(v) for v in ticks], fontsize=T_TICK, color=CAP, family=SANS)


def hgrid(ax: Any, values: Sequence[float], fmt: Callable[[float], str] = lambda v: f"{v:g}",
          baseline: float | None = None, label_side: str = "left") -> None:
    """Gridlines with value labels and an optional baseline."""
    for v in values:
        ax.axhline(v, color=GRID, lw=0.6, zorder=0)
    ax.set_yticks(list(values))
    ax.set_yticklabels([fmt(v) for v in values], fontsize=T_TICK, color=CAP, family=SANS)
    if label_side == "right":
        ax.yaxis.tick_right()
    if baseline is not None:
        ax.axhline(baseline, color=INK, lw=0.9, zorder=2)
