"""Value and unit parsing.

Most of these are cases where a plausible-looking implementation is wrong in a
way no one notices until a comparison silently flips sign or moves by a factor
of a hundred.
"""

import pytest

from collate.normalize.quantity import (
    parse,
    parse_number,
    parse_unit,
    relative_gap,
    same_to_sig_figs,
)


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("8,142", 8142.0),
        ("1,23,456", 123456.0),      # Indian grouping, not thousands
        ("1,234,567", 1234567.0),    # international grouping, same answer
        ("6.4", 6.4),
        ("~6.5", 6.5),
        ("approx. 12.3", 12.3),
        ("-3.2", -3.2),
    ],
)
def test_plain_numbers(raw, expected):
    assert parse_number(raw)[0] == pytest.approx(expected)


@pytest.mark.parametrize("raw,expected", [("(452)", -452.0), ("(1,234.5)", -1234.5)])
def test_accounting_negatives(raw, expected):
    """Parentheses mean negative. float('(452)') raises, but a regex that merely
    strips punctuation returns +452 and inverts every comparison that follows."""
    assert parse_number(raw)[0] == pytest.approx(expected)


def test_stated_ranges_keep_their_bounds():
    value, low, high = parse_number("6.3-6.8")
    assert (low, high) == (pytest.approx(6.3), pytest.approx(6.8))
    assert value == pytest.approx(6.55)


def test_unparseable_values_are_none_not_zero():
    """Zero is a real figure. Returning it for 'not applicable' would put a
    fabricated number into the apparatus."""
    assert parse_number("not applicable")[0] is None
    assert parse_number("")[0] is None
    assert parse_number("NA")[0] is None


@pytest.mark.parametrize(
    "unit,dimension,scale",
    [
        ("Rs Cr", "INR", 1e7),
        ("₹ crore", "INR", 1e7),
        ("INR million", "INR", 1e6),
        ("Rs. in lakhs", "INR", 1e5),
        ("US$ bn", "USD", 1e9),
        ("$ billion", "USD", 1e9),
        ("per cent", "percent", 1.0),
        ("%", "percent", 1.0),
        ("bps", "percent", 0.01),
    ],
)
def test_units(unit, dimension, scale):
    assert parse_unit(unit) == (dimension, scale)


def test_crore_and_million_reconcile():
    """The corroboration case: a deck in crore and statements in million.

    Delhivery's FY24 revenue is Rs 8,142 Cr. The same figure in the annual
    report's tables is Rs 81,420 million. Nothing about the two strings
    matches; the numbers must.
    """
    deck = parse("8,142", "Rs Cr")
    statements = parse("81,420", "₹ in Million")
    assert deck.dimension == statements.dimension == "INR"
    assert relative_gap(deck.value, statements.value) < 0.001


def test_lakh_versus_million_is_a_hundredfold_error():
    """The failure the table lane exists to prevent: reading a figure under
    'Rs in lakhs' as though the header had said millions."""
    correct = parse("5,000", "Rs in lakhs")
    misread = parse("5,000", "Rs in million")
    assert misread.value / correct.value == pytest.approx(10.0)


def test_sig_figs_absorbs_rounding_between_scales():
    assert same_to_sig_figs(8.142e10, 8.1424e10, figs=4)
    assert not same_to_sig_figs(8.142e10, 8.9e10, figs=4)


def test_relative_gap_is_symmetric_and_zero_safe():
    assert relative_gap(100, 110) == pytest.approx(relative_gap(110, 100))
    assert relative_gap(0, 0) == 0.0


def test_percent_never_takes_a_currency_scale():
    """'6.4 per cent' must not inherit a document's default crore scaling."""
    q = parse("6.4", "per cent", fallback_currency="INR")
    assert q.dimension == "percent"
    assert q.value == pytest.approx(6.4)
