"""Turning written periods into intervals.

This is the single highest-leverage file in Collate, because almost every
apparent contradiction in the starter corpus is a period-labelling difference
wearing a disguise. Three documents describe the identical twelve months as:

    "FY24"          Delhivery, ending-year convention
    "2024-25"       Economic Survey, spanning-year convention  <- a DIFFERENT year
    "FY2024/25"     IMF, spanning-year convention

Get the convention wrong and the system either invents contradictions between
documents that agree, or silently merges years that do not. Both failures are
invisible without tests, which is why this module has more of them than any
other.

Conventions implemented, all Indian-fiscal (April to March):

    FY24, FY2024          -> the year ENDING March 2024   [2023-04-01, 2024-04-01)
    FY2023-24, FY23-24    -> the same twelve months
    FY2024-25, FY2024/25  -> the year ENDING March 2025   [2024-04-01, 2025-04-01)
    2024-25               -> spanning form, same as FY2024-25
    Q4 FY24               -> Jan-Mar 2024 (Q1 is Apr-Jun)
    H1 FY24, 9M FY24      -> Apr-Sep 2023, Apr-Dec 2023
    first eight months of FY25 -> Apr-Nov 2024, and NOT the whole year
    CY2024                -> calendar [2024-01-01, 2025-01-01)
    year ended 31 Mar 24  -> resolved to the fiscal year it closes
    as at 31 March 2024   -> an instant, not a span

Intervals are half-open: [start, end). An instant is start == end.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date

from dateutil import parser as dateparser

FY_START_MONTH = 4  # April. The one convention this module is opinionated about.

QUARTER_MONTHS = {1: (4, 6), 2: (7, 9), 3: (10, 12), 4: (1, 3)}
PART_YEAR = {"h1": (0, 6), "h2": (6, 12), "9m": (0, 9), "1h": (0, 6), "2h": (6, 12)}


@dataclass(frozen=True)
class Period:
    start: date
    end: date  # exclusive
    grain: str  # year | half | quarter | month | instant | unknown

    @property
    def is_instant(self) -> bool:
        return self.start == self.end

    def as_tuple(self) -> tuple[str, str, str]:
        return self.start.isoformat(), self.end.isoformat(), self.grain

    def overlaps(self, other: "Period") -> bool:
        if self.is_instant or other.is_instant:
            return self.start == other.start
        return self.start < other.end and other.start < self.end

    def contains(self, other: "Period") -> bool:
        return self.start <= other.start and other.end <= self.end

    def __str__(self) -> str:
        if self.is_instant:
            return f"on {self.start.isoformat()}"
        return f"[{self.start.isoformat()}, {self.end.isoformat()})"


def fiscal_year(ending_year: int) -> Period:
    """The Indian fiscal year that ENDS in March of `ending_year`."""
    return Period(
        date(ending_year - 1, FY_START_MONTH, 1),
        date(ending_year, FY_START_MONTH, 1),
        "year",
    )


def _expand(yy: int) -> int:
    """23 -> 2023. Two-digit years in these documents are never last century."""
    return yy + 2000 if yy < 100 else yy


_INSTANT = re.compile(
    r"\b(?:as\s+(?:at|of|on)|as\s+at\s+the\s+end\s+of|position\s+as\s+at)\b", re.I
)
_ENDED = re.compile(
    r"\b(?:year|period|twelve\s+months?|12\s+months?)\s+ended\b", re.I
)
_FY_SPAN = re.compile(
    r"\b(?:FY|F\.Y\.|fiscal(?:\s+year)?|financial\s+year)?\s*"
    r"(\d{4}|\d{2})\s*[-/–]\s*(\d{4}|\d{2})\b",
    re.I,
)
_FY_SINGLE = re.compile(r"\b(?:FY|F\.Y\.|fiscal(?:\s+year)?|financial\s+year)\s*[' ]?(\d{4}|\d{2})\b", re.I)
_CY = re.compile(r"\b(?:CY|calendar\s+year)\s*(\d{4}|\d{2})\b", re.I)
_QUARTER = re.compile(r"\bQ([1-4])\b", re.I)
_PART = re.compile(r"\b(H1|H2|1H|2H|9M)\b", re.I)
# "the first eight months of FY25" is eight months, not a year. Left unhandled,
# it resolved to the whole of FY25 and an eight-month FDI figure was compared
# against a twelve-month one, which the cascade then reported as a 98%
# contradiction between two institutions that were not disagreeing.
_WORD_MONTHS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
}
_FIRST_N_MONTHS = re.compile(
    r"\bfirst\s+(" + "|".join(_WORD_MONTHS) + r"|\d{1,2})\s+months?\b", re.I
)
_BARE_YEAR = re.compile(r"\b(19|20)\d{2}\b")


def parse(text: str | None) -> Period | None:
    """Best-effort period parse. Returns None when nothing is stated, which is a
    legitimate outcome: many claims are timeless and forcing a period onto them
    would manufacture false frames."""
    if not text:
        return None
    s = " ".join(str(text).split())
    if not s:
        return None
    low = s.lower()

    # "as at 31 March 2024" - a balance-sheet instant, never a span.
    if _INSTANT.search(low):
        d = _try_date(s)
        if d:
            return Period(d, d, "instant")

    # "year ended March 31, 2024" resolves through its closing date.
    if _ENDED.search(low):
        d = _try_date(s)
        if d:
            end_year = d.year if d.month >= FY_START_MONTH else d.year
            if d.month == 3:  # closes the Indian fiscal year
                return fiscal_year(d.year)
            return Period(_minus_year(d), d, "year")

    part = _PART.search(low)
    quarter = _QUARTER.search(low)
    first_n = _FIRST_N_MONTHS.search(low)

    base = _fiscal_from(low)
    if base is None and (part or quarter):
        base = None

    if base is not None:
        if first_n:
            token = first_n.group(1).lower()
            n = _WORD_MONTHS.get(token) or int(token)
            n = max(1, min(n, 12))
            return Period(base.start, _add_months(base.start, n),
                          "year" if n == 12 else "part")
        if quarter:
            q = int(quarter.group(1))
            m0, m1 = QUARTER_MONTHS[q]
            # Q1-Q3 sit in the opening calendar year, Q4 in the closing one.
            year = base.end.year if q == 4 else base.start.year
            start = date(year, m0, 1)
            end = _add_month(date(year, m1, 1))
            return Period(start, end, "quarter")
        if part:
            off, span = PART_YEAR[part.group(1).lower()]
            start = _add_months(base.start, off)
            return Period(start, _add_months(base.start, span), "half" if span == 6 else "period")
        return base

    cy = _CY.search(s)
    if cy:
        y = _expand(int(cy.group(1)))
        return Period(date(y, 1, 1), date(y + 1, 1, 1), "year")

    # A bare date with no framing word.
    d = _try_date(s)
    if d and not _BARE_YEAR.fullmatch(s.strip()):
        return Period(d, d, "instant")

    m = _BARE_YEAR.search(s)
    if m and len(s) <= 6:
        y = int(m.group(0))
        return Period(date(y, 1, 1), date(y + 1, 1, 1), "year")

    return None


def _fiscal_from(low: str) -> Period | None:
    """FY2023-24 / 2024-25 / FY24, in that order of specificity."""
    span = _FY_SPAN.search(low)
    if span:
        a, b = _expand(int(span.group(1))), _expand(int(span.group(2)))
        if b == a + 1:
            return Period(date(a, FY_START_MONTH, 1), date(b, FY_START_MONTH, 1), "year")
        if b > a:  # e.g. "2019-2024", a multi-year span
            return Period(date(a, FY_START_MONTH, 1), date(b, FY_START_MONTH, 1), "multi")

    single = _FY_SINGLE.search(low)
    if single:
        return fiscal_year(_expand(int(single.group(1))))
    return None


def _try_date(s: str) -> date | None:
    try:
        parsed = dateparser.parse(s, fuzzy=True, dayfirst=True, default=None)
    except (ValueError, OverflowError, TypeError):
        return None
    return parsed.date() if parsed else None


def _add_month(d: date) -> date:
    return date(d.year + 1, 1, 1) if d.month == 12 else date(d.year, d.month + 1, 1)


def _add_months(d: date, n: int) -> date:
    total = (d.year * 12 + d.month - 1) + n
    return date(total // 12, total % 12 + 1, 1)


def _minus_year(d: date) -> date:
    try:
        return d.replace(year=d.year - 1)
    except ValueError:  # 29 Feb
        return d.replace(year=d.year - 1, day=28)


def describe_difference(a: Period, b: Period) -> str | None:
    """A human sentence for why two periods differ, or None when they match."""
    if a == b:
        return None
    if a.contains(b) or b.contains(a):
        inner, outer = (b, a) if a.contains(b) else (a, b)
        return f"{inner} falls inside {outer}: a part being compared with a whole"
    if not a.overlaps(b):
        return f"{a} and {b} do not overlap at all"
    return f"{a} and {b} overlap only partly"
