"""Values and units.

Two jobs that have to happen together, because neither is meaningful alone:
"8,142" is not a number until you know it means crore, and "crore" is not a
scale until you know what it is scaling.

The parsing rules here are the ones that actually bite in Indian filings:

    (452)       accounting negatives, which a naive float() reads as positive
                and which then flip the sign of every comparison downstream
    1,23,456    Indian digit grouping - lakh-crore, not thousand-million
    ~6.5, 6.5%  approximation marks and trailing units
    6.4-6.8     a stated range; kept as a range, because collapsing it to a
                midpoint would let us report a precision the document did not
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Scale words -> multiplier. Indian and international side by side, because
# these documents mix them freely and sometimes within one table.
SCALES: dict[str, float] = {
    "hundred": 1e2, "hundreds": 1e2,
    "thousand": 1e3, "thousands": 1e3, "k": 1e3,
    "lakh": 1e5, "lakhs": 1e5, "lac": 1e5, "lacs": 1e5,
    "million": 1e6, "millions": 1e6, "mn": 1e6, "mio": 1e6, "m": 1e6,
    "crore": 1e7, "crores": 1e7, "cr": 1e7, "cr.": 1e7,
    "billion": 1e9, "billions": 1e9, "bn": 1e9, "b": 1e9,
    "trillion": 1e12, "trillions": 1e12, "tn": 1e12, "tr": 1e12,
}

CURRENCIES: dict[str, str] = {
    "₹": "INR", "rs": "INR", "rs.": "INR", "inr": "INR", "rupee": "INR", "rupees": "INR",
    "$": "USD", "us$": "USD", "usd": "USD", "us dollar": "USD", "dollars": "USD",
    "€": "EUR", "eur": "EUR", "£": "GBP", "gbp": "GBP", "¥": "JPY", "jpy": "JPY",
}

PERCENT_WORDS = {"%", "percent", "per cent", "pct", "percentage", "percentage points", "bps", "basis points"}

_NUMBER = re.compile(r"[-+]?\d[\d,\s]*(?:\.\d+)?")
_PARENS_NEG = re.compile(r"^\s*\(\s*(.+?)\s*\)\s*$")
_RANGE = re.compile(
    r"^\s*([-+]?\d[\d,]*(?:\.\d+)?)\s*(?:-|–|—|to|and)\s*([-+]?\d[\d,]*(?:\.\d+)?)\s*$", re.I
)


@dataclass(frozen=True)
class Quantity:
    value: float
    dimension: str           # INR | USD | percent | count | ratio | text | ...
    scale: float             # multiplier already folded into `value`
    low: float | None = None  # populated only for stated ranges
    high: float | None = None

    @property
    def is_range(self) -> bool:
        return self.low is not None

    def __str__(self) -> str:
        if self.is_range:
            return f"{self.low:,.4g}-{self.high:,.4g} {self.dimension}"
        return f"{self.value:,.6g} {self.dimension}"


def parse_number(raw: str | None) -> tuple[float | None, float | None, float | None]:
    """Return (value, range_low, range_high). Ranges report their midpoint as value
    so ordinary comparisons still work, while keeping the bounds for display."""
    if raw is None:
        return None, None, None
    s = str(raw).strip()
    if not s:
        return None, None, None

    negative = False
    m = _PARENS_NEG.match(s)
    if m:                       # (452) is minus 452 in every filing that uses it
        negative, s = True, m.group(1)

    s = s.replace("−", "-").replace("–", "-").strip()
    s = re.sub(r"^[~≈<>≤≥+]+\s*", "", s)          # approximation and inequality marks
    s = re.sub(r"(?i)\b(?:approx\.?|about|around|over|under|nearly)\b", "", s).strip()

    rng = _RANGE.match(s)
    if rng:
        lo = _to_float(rng.group(1))
        hi = _to_float(rng.group(2))
        if lo is not None and hi is not None:
            if lo > hi:
                lo, hi = hi, lo
            return (lo + hi) / 2.0, lo, hi

    hit = _NUMBER.search(s)
    if not hit:
        return None, None, None
    val = _to_float(hit.group(0))
    if val is None:
        return None, None, None
    return (-val if negative else val), None, None


def _to_float(token: str) -> float | None:
    """Commas are group separators regardless of grouping style, so stripping them
    handles 1,234,567 and 12,34,567 identically without having to detect which."""
    try:
        return float(token.replace(",", "").replace(" ", ""))
    except ValueError:
        return None


def parse_unit(raw: str | None, fallback_currency: str | None = None) -> tuple[str, float]:
    """Return (dimension, scale multiplier)."""
    s = " ".join(str(raw or "").lower().split())
    if not s:
        # No unit stated means a count, never the document's default currency.
        # Inheriting INR here turned "Processing centers: 160" into 160 rupees,
        # which then sat in the same dimension as revenue and compared against it.
        return "count", 1.0

    if any(w in s for w in PERCENT_WORDS):
        # Basis points are percent/100 and get folded in here rather than
        # becoming their own dimension, so 50 bps compares to 0.5%.
        if "bps" in s or "basis point" in s:
            return "percent", 0.01
        return "percent", 1.0

    dimension = None
    for token, code in CURRENCIES.items():
        if token in s:
            dimension = code
            break
    if dimension is None:
        # A unit was stated but names no currency ("Mn shipments", "tonnes").
        # Only inherit the document's currency when the unit is a bare scale
        # word, which is the one case where the currency really was elided.
        bare_scale = s.strip() in SCALES
        dimension = (fallback_currency or "count") if bare_scale else "count"

    scale = 1.0
    for word, mult in sorted(SCALES.items(), key=lambda kv: -len(kv[0])):
        if re.search(rf"(?<![a-z]){re.escape(word)}(?![a-z])", s):
            scale = mult
            break

    return dimension, scale


def parse(value_raw: str | None, unit_raw: str | None,
          fallback_currency: str | None = None) -> Quantity | None:
    val, lo, hi = parse_number(value_raw)
    if val is None:
        return None
    dimension, scale = parse_unit(unit_raw, fallback_currency)
    return Quantity(
        value=val * scale,
        dimension=dimension,
        scale=scale,
        low=lo * scale if lo is not None else None,
        high=hi * scale if hi is not None else None,
    )


def relative_gap(a: float, b: float) -> float:
    """Symmetric relative difference, safe around zero."""
    denom = max(abs(a), abs(b))
    return 0.0 if denom == 0 else abs(a - b) / denom


def same_to_sig_figs(a: float, b: float, figs: int = 3) -> bool:
    """Do two values agree once rounded to the coarser one's precision?

    This is how a deck reporting Rs 8,142 Cr and statements reporting
    Rs 81,424 million are recognised as the same number rather than a 0.005%
    discrepancy that some threshold has to be tuned to forgive.
    """
    if a == 0 or b == 0:
        return a == b
    from math import floor, log10

    def rounded(x: float) -> float:
        digits = figs - 1 - floor(log10(abs(x)))
        return round(x, digits)

    return rounded(a) == rounded(b)
