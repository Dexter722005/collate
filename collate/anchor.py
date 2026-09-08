"""The anchoring gate.

A claim that cannot be found in the page it allegedly came from is not a claim,
it is a plausible sentence. This module is the only thing standing between the
two, and it is deliberately mechanical: no model is consulted about whether a
model told the truth.

Three passes, weakest evidence last:

    exact       the quote is in the page text verbatim
    whitespace  it matches once newlines and runs of spaces are flattened, which
                is nearly always what "the model reflowed a table row" looks like
    fuzzy       rapidfuzz clears a high bar; accepted, but recorded as fuzzy so
                the UI can mark it and the stats can count it

Anything below that is quarantined with its payload intact. The quarantine
table is not a shameful corner of the schema - its size is a headline number.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from rapidfuzz import fuzz

FUZZY_FLOOR = 88.0
MIN_QUOTE_CHARS = 12


@dataclass
class Anchor:
    page_no: int
    char_start: int
    char_end: int
    bbox: tuple[float, float, float, float]
    match_kind: str


def _flatten(text: str) -> tuple[str, list[int]]:
    """Collapse whitespace, keeping a map from each output char to its source index."""
    out: list[str] = []
    idx: list[int] = []
    prev_space = False
    for i, ch in enumerate(text):
        if ch.isspace():
            if prev_space:
                continue
            out.append(" ")
            idx.append(i)
            prev_space = True
        else:
            out.append(ch)
            idx.append(i)
            prev_space = False
    return "".join(out), idx


def _bbox_for(page, start: int, end: int) -> tuple[float, float, float, float]:
    """Union of every block the span touches. Block-level rather than glyph-level
    because a highlight that is slightly generous reads fine, whereas one that is
    slightly wrong destroys trust in the evidence pane."""
    hits = [b for b in page.blocks if b.char_start < end and b.char_end > start]
    if not hits:
        return (0.0, 0.0, page.width, 0.0)
    return (
        min(b.bbox[0] for b in hits),
        min(b.bbox[1] for b in hits),
        max(b.bbox[2] for b in hits),
        max(b.bbox[3] for b in hits),
    )


def locate(quote: str, pages: list, hint_page: int | None = None) -> Anchor | None:
    """Find a quote among candidate pages. hint_page is tried first but never trusted -
    models are reliable about content and careless about which page marker it sat under."""
    quote = quote.strip()
    if len(quote) < MIN_QUOTE_CHARS:
        return None

    ordered = list(pages)
    if hint_page is not None:
        ordered.sort(key=lambda p: abs(p.page_no - hint_page))

    # Pass 1 and 2: exact, then whitespace-insensitive.
    flat_quote, _ = _flatten(quote)
    for page in ordered:
        at = page.text.find(quote)
        if at >= 0:
            return Anchor(page.page_no, at, at + len(quote), _bbox_for(page, at, at + len(quote)), "exact")

    for page in ordered:
        flat_page, back = _flatten(page.text)
        at = flat_page.find(flat_quote)
        if at >= 0:
            start = back[at]
            end = back[min(at + len(flat_quote), len(back)) - 1] + 1
            return Anchor(page.page_no, start, end, _bbox_for(page, start, end), "whitespace")

    # Pass 3: fuzzy, windowed so a long page cannot drown a short quote.
    best: tuple[float, object, int, int] | None = None
    for page in ordered:
        flat_page, back = _flatten(page.text)
        if len(flat_page) < len(flat_quote):
            continue
        m = fuzz.partial_ratio_alignment(flat_quote, flat_page, score_cutoff=FUZZY_FLOOR)
        if m is None:
            continue
        if best is None or m.score > best[0]:
            best = (m.score, page, m.dest_start, m.dest_end)

    if best:
        _, page, a, b = best
        flat_page, back = _flatten(page.text)
        start = back[min(a, len(back) - 1)]
        end = back[min(b, len(back)) - 1] + 1
        return Anchor(page.page_no, start, end, _bbox_for(page, start, end), "fuzzy")

    return None


_NUM = re.compile(r"-?[\d,][\d,.]*")


def value_present(value_raw: str, quote: str) -> bool:
    """Does the value the model reported actually appear in the quote it cited?

    Catches the failure that matters most: a real quote paired with a number
    lifted from somewhere else on the page. An anchor proves the quote exists;
    this proves the quote is the right one.
    """
    if not value_raw:
        return True
    v = value_raw.strip().strip("()%").replace(",", "").lstrip("-")
    if not v:
        return True
    if v.lower() in quote.lower().replace(",", ""):
        return True
    # Compare numerically, so 8,142 in the quote satisfies a reported 8142.
    try:
        target = float(v)
    except ValueError:
        return v.lower() in quote.lower()
    for token in _NUM.findall(quote.replace(",", "")):
        try:
            if abs(float(token.strip(".,")) - target) < 1e-9:
                return True
        except ValueError:
            continue
    return False
