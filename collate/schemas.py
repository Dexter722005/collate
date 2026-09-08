"""Contracts between Collate and the model.

These doubles as the structured-output schema handed to Gemini and as the
validation layer for what comes back. Field descriptions are load-bearing -
they are the prompt as much as the prompt is.

Note what is deliberately absent: any list of expected measures, entities,
periods or units. The model reports what the document says, in the document's
own words. Every act of interpretation happens downstream in normalize/, in
Python, where it can be unit-tested and argued with.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

ClaimType = Literal["quantity", "state", "event", "identity"]


class RawClaim(BaseModel):
    """One assertion, exactly as the document words it."""

    claim_type: ClaimType = Field(
        description=(
            "quantity: a measured number. state: a condition holding over time "
            "(a person holds an office, a facility is operational). event: something "
            "that happened on a date. identity: two surface forms naming one thing."
        )
    )
    quote: str = Field(
        description=(
            "Copied character-for-character from the passage, long enough to contain "
            "the value and its label. Never paraphrased, never reflowed, never repaired."
        )
    )
    page: int = Field(description="The [page N] marker the quote sits under.")
    entity: str = Field(
        description=(
            "The organisation, country, person or place the claim is about, as written. "
            "A business segment, product line or line item is NOT an entity: for "
            "'Express Parcel revenue' at Acme Ltd, the entity is 'Acme Ltd' and the "
            "measure is 'Express Parcel revenue'. Use the document's own name for its "
            "subject; 'the Company' is acceptable and will be resolved."
        )
    )
    measure: str = Field(
        description="What is being asserted, as written: 'Revenue from operations', "
        "'real GDP growth', 'Director'. Keep any segment or product qualifier here. "
        "Do not translate into a standard vocabulary."
    )
    value: str = Field(description="The value exactly as printed, digits and all: "
                                   "'8,142', '(452)', '6.4', 'resigned'.")
    unit: str = Field(description="Unit as written, including scale: 'Rs Cr', "
                                  "'per cent', 'INR million'. Empty if none.")
    period: str = Field(description="The time the claim covers, as written: 'FY24', "
                                    "'Q4 FY2023-24', 'as at March 31, 2024'. Empty if none.")
    basis: str = Field(description="Scope or accounting qualifiers as written: "
                                   "'consolidated', 'standalone', 'provisional', "
                                   "'restated', 'seasonally adjusted'. Empty if none.")
    qualifiers: list[str] = Field(
        default_factory=list,
        description="Any other conditions that change how the value should be read, "
        "including footnote text if it is visible in the passage.",
    )
    confidence: float = Field(
        description="0-1. Lower it when the label is far from the value, when a scale "
        "header had to be assumed, or when the table structure was ambiguous."
    )


class Extraction(BaseModel):
    claims: list[RawClaim]


class FrontMatter(BaseModel):
    """Document-level properties, established once and inherited by every claim."""

    title: str = Field(description="Document title as printed.")
    publisher: str = Field(description="Issuing organisation.")
    doc_type: str = Field(description="e.g. prospectus, annual report, earnings "
                                      "presentation, staff report, statistical release.")
    vintage_date: str = Field(
        description="Publication date as ISO yyyy-mm-dd, or yyyy-mm / yyyy if only that "
        "is knowable. This is when the document was PUBLISHED, never the period it "
        "reports on. Empty string if genuinely absent."
    )
    default_currency: str = Field(description="Predominant currency code, e.g. INR, USD. "
                                              "Empty if not a financial document.")
    default_scale: str = Field(description="Predominant numeric scale, e.g. million, "
                                           "crore, billion. Empty if unscaled.")
    primary_entity: str = Field(description="The organisation, country or subject the "
                                            "document is principally about.")


class Adjudication(BaseModel):
    """The escalation verdict. Advisory only - it annotates, never overrides.

    The rule cascade has already decided by the time this is asked for. We want
    the model's read on context the rules cannot see, not its arithmetic.
    """

    relationship: Literal["corroborates", "contradicts", "reconciled", "distinct"]
    reason_code: str = Field(
        description="A short SCREAMING_SNAKE label for the mechanism, invented to fit "
        "what you actually found: DIFFERENT_SEGMENT, RESTATED_AFTER_ACQUISITION, "
        "EXCLUDES_ONE_OFFS. Do not force it into a preset list."
    )
    explanation: str = Field(
        description="One or two sentences a careful analyst would accept, citing only "
        "what is visible in the two quotes. If the quotes do not support a "
        "reconciliation, say the contradiction stands."
    )
    sufficient_context: bool = Field(
        description="False if the quotes alone cannot settle it. Say so rather than "
        "inventing a reconciliation."
    )
