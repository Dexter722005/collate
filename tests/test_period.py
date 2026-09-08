"""Period parsing.

The cases that matter are the ones where two documents describe the same twelve
months differently, and the ones where they describe *different* twelve months
using labels that look the same. Both failures are invisible in the UI: the
first invents contradictions, the second hides them.
"""

from datetime import date

import pytest

from collate.normalize.period import Period, parse


def d(y, m, day=1):
    return date(y, m, day)


@pytest.mark.parametrize(
    "text,start,end",
    [
        # Ending-year convention: FY24 closes in March 2024.
        ("FY24", d(2023, 4), d(2024, 4)),
        ("FY 24", d(2023, 4), d(2024, 4)),
        ("FY2024", d(2023, 4), d(2024, 4)),
        ("fiscal 2024", d(2023, 4), d(2024, 4)),
        # Spanning form of the very same year.
        ("FY2023-24", d(2023, 4), d(2024, 4)),
        ("FY23-24", d(2023, 4), d(2024, 4)),
        ("2023-24", d(2023, 4), d(2024, 4)),
        # ...and the year after it, which the short form would call FY25.
        ("FY2024-25", d(2024, 4), d(2025, 4)),
        ("FY2024/25", d(2024, 4), d(2025, 4)),
        ("2024-25", d(2024, 4), d(2025, 4)),
    ],
)
def test_fiscal_year_conventions(text, start, end):
    p = parse(text)
    assert p is not None, text
    assert (p.start, p.end) == (start, end), text


def test_the_trap_that_would_silently_merge_two_years():
    """The Economic Survey's '2024-25' and Delhivery's 'FY24' are a year apart.

    A parser that reads the leading number of each and calls it a year would
    map both onto 2024 and quietly reconcile figures twelve months apart.
    """
    survey = parse("2024-25")
    filing = parse("FY24")
    assert survey != filing
    assert not survey.overlaps(filing)


def test_imf_and_survey_agree_on_the_same_year():
    """Different publishers, different notation, identical interval."""
    assert parse("FY2024/25") == parse("2024-25")


@pytest.mark.parametrize(
    "text,start,end",
    [
        ("Q1 FY24", d(2023, 4), d(2023, 7)),
        ("Q2 FY24", d(2023, 7), d(2023, 10)),
        ("Q3 FY24", d(2023, 10), d(2024, 1)),
        ("Q4 FY24", d(2024, 1), d(2024, 4)),  # Q4 falls in the closing year
    ],
)
def test_indian_fiscal_quarters(text, start, end):
    p = parse(text)
    assert (p.start, p.end) == (start, end), text
    assert p.grain == "quarter"


def test_quarter_sits_inside_its_year():
    year, q4 = parse("FY24"), parse("Q4 FY24")
    assert year.contains(q4)
    assert not q4.contains(year)


@pytest.mark.parametrize(
    "text,start,end",
    [("H1 FY24", d(2023, 4), d(2023, 10)), ("9M FY24", d(2023, 4), d(2024, 1))],
)
def test_part_years(text, start, end):
    p = parse(text)
    assert (p.start, p.end) == (start, end), text


def test_calendar_year_is_not_a_fiscal_year():
    cy = parse("CY2024")
    assert (cy.start, cy.end) == (d(2024, 1), d(2025, 1))
    assert cy != parse("FY2024")


def test_year_ended_resolves_through_its_closing_date():
    assert parse("year ended March 31, 2024") == parse("FY24")


def test_as_at_is_an_instant_not_a_span():
    p = parse("as at March 31, 2024")
    assert p.is_instant
    assert p.start == d(2024, 3, 31)


def test_missing_period_stays_missing():
    """Inventing a period would manufacture frames that do not exist."""
    assert parse("") is None
    assert parse(None) is None
    assert parse("in the ordinary course of business") is None


def test_disjoint_and_nested_are_distinguishable():
    fy22, fy24 = parse("FY22"), parse("FY24")
    assert not fy22.overlaps(fy24)
    assert parse("FY24").contains(parse("Q3 FY24"))
