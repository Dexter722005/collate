"""The rule cascade.

Rules decide; the model explains and escalates. That ordering is the central
engineering choice in Collate, and it is the reverse of the obvious design.

The obvious design hands two claims to an LLM and asks "do these conflict?".
It works, demonstrates nothing, and cannot be audited: the answer changes
between runs, cites no mechanism, and is wrong in ways that look exactly like
being right. Worse, it cannot produce the interesting case - saying *why* two
figures differ requires comparing their coordinates, which is arithmetic and
date logic, not language.

So every pair goes through an ordered cascade of deterministic rules. Each rule
inspects one coordinate of the frame, and the first to fire wins and names
itself in the output. A pair reaches the model only when the rules have already
concluded CONTRADICTS - and even then the model's opinion is recorded beside
the verdict, never in place of it. If it finds context the rules could not see,
that is a finding worth showing. If it disagrees on the arithmetic, the
arithmetic wins.

Reading the rules top to bottom is meant to read like an argument.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date

from .normalize import period as period_mod
from .normalize import stage_rank
from .normalize.quantity import relative_gap, same_to_sig_figs

# Below this, two figures are the same number reported at different precision.
ROUNDING_TOLERANCE = 0.005
SIG_FIGS = 3

# Families whose disagreement explains a difference rather than being one.
CONSOLIDATION = {"consolidated", "standalone", "segment"}
ADJUSTMENT = {"real", "nominal", "seasonally_adjusted", "annualised", "per_capita", "gross", "net"}
ASSURANCE = {"audited", "unaudited"}


@dataclass
class Ruling:
    verdict: str          # corroborates | contradicts | reconciled | distinct
    rule_id: str
    explanation: str
    divergence: float | None = None
    severity: float | None = None


@dataclass
class Side:
    """One claim, with everything a rule needs, flattened."""

    id: int
    witness_id: int
    witness_title: str
    vintage: date | None
    claim_type: str
    measure: str
    value: float | None
    value_text: str | None
    unit_dim: str | None
    unit_raw: str | None
    value_raw: str | None
    period: period_mod.Period | None
    period_raw: str | None
    basis: list[str]
    quote: str
    confidence: float | None

    @classmethod
    def build(cls, row, witness) -> "Side":
        per = None
        if row["period_start"] and row["period_end"]:
            per = period_mod.Period(
                date.fromisoformat(row["period_start"]),
                date.fromisoformat(row["period_end"]),
                row["period_grain"] or "unknown",
            )
        vintage = None
        if witness and witness["vintage_date"]:
            try:
                vintage = date.fromisoformat(witness["vintage_date"][:10])
            except ValueError:
                vintage = None
        return cls(
            id=row["id"],
            witness_id=row["witness_id"],
            witness_title=(witness["title"] if witness else "?") or "?",
            vintage=vintage,
            claim_type=row["claim_type"],
            measure=row["measure_raw"] or "",
            value=row["value_num"],
            value_text=row["value_text"],
            unit_dim=row["unit_dim"],
            unit_raw=row["unit_raw"],
            value_raw=row["value_raw"],
            period=per,
            period_raw=row["period_raw"],
            basis=json.loads(row["basis"]) if row["basis"] else [],
            quote=row["quote"] or "",
            confidence=row["confidence"],
        )

    def where(self) -> str:
        return f"{self.witness_title} ({self.vintage.isoformat() if self.vintage else 'undated'})"


def _family(basis: list[str], family: set[str]) -> set[str]:
    return set(basis) & family


# -- the cascade ------------------------------------------------------------
# Each rule returns a Ruling to claim the pair, or None to pass it along.


def r_dimension(a: Side, b: Side) -> Ruling | None:
    """A level and a rate are not rival readings of anything."""
    if a.unit_dim and b.unit_dim and a.unit_dim != b.unit_dim:
        return Ruling(
            "distinct",
            "DIMENSION_MISMATCH",
            f"Not comparable: one is measured in {a.unit_dim}, the other in {b.unit_dim}.",
        )
    return None


def r_period_missing(a: Side, b: Side) -> Ruling | None:
    """Refuse to compare undated figures rather than assume they share a year."""
    if a.period is None or b.period is None:
        return Ruling(
            "distinct",
            "PERIOD_UNSTATED",
            "At least one claim states no period, so the two cannot be placed in the "
            "same frame. Comparing them would assume a period neither document gives.",
        )
    return None


def r_period_disjoint(a: Side, b: Side) -> Ruling | None:
    if not a.period.overlaps(b.period):
        return Ruling(
            "reconciled",
            "PERIOD_DISJOINT",
            f"Different reporting periods, so no disagreement: "
            f"{a.period_raw or a.period} in {a.where()} against "
            f"{b.period_raw or b.period} in {b.where()}.",
        )
    return None


def r_period_nested(a: Side, b: Side) -> Ruling | None:
    # Identical periods are not nested, whatever set theory says. contains() is
    # inclusive on both ends, so an equal pair satisfies it and this rule was
    # claiming every same-period comparison before r_agreement could see it -
    # 92 spurious PERIOD_NESTED verdicts and not one corroboration in the whole
    # corpus. The cascade's power is that the first match wins, which is exactly
    # what makes a rule that matches too eagerly so quiet a failure.
    if a.period == b.period:
        return None
    if a.period.contains(b.period) or b.period.contains(a.period):
        inner, outer = (b, a) if a.period.contains(b.period) else (a, b)
        return Ruling(
            "reconciled",
            "PERIOD_NESTED",
            f"A part is being compared with a whole: {inner.period_raw or inner.period} "
            f"falls inside {outer.period_raw or outer.period}. The smaller figure is a "
            f"component of the larger, not a rival estimate of it.",
        )
    return None


def r_period_partial(a: Side, b: Side) -> Ruling | None:
    if a.period != b.period:
        return Ruling(
            "reconciled",
            "PERIOD_PARTIAL",
            f"Periods overlap only in part ({a.period} against {b.period}), so the two "
            f"figures cover different spans of time.",
        )
    return None


def r_basis_consolidation(a: Side, b: Side) -> Ruling | None:
    fa, fb = _family(a.basis, CONSOLIDATION), _family(b.basis, CONSOLIDATION)
    if fa and fb and fa != fb:
        return Ruling(
            "reconciled",
            "BASIS_MISMATCH",
            f"Different reporting basis: {'/'.join(sorted(fa))} against "
            f"{'/'.join(sorted(fb))}. These are different figures by construction, not "
            f"competing measurements of one figure.",
        )
    return None


def r_basis_adjustment(a: Side, b: Side) -> Ruling | None:
    fa, fb = _family(a.basis, ADJUSTMENT), _family(b.basis, ADJUSTMENT)
    if fa and fb and fa != fb:
        return Ruling(
            "reconciled",
            "ADJUSTMENT_MISMATCH",
            f"Different adjustment: {'/'.join(sorted(fa))} against {'/'.join(sorted(fb))}.",
        )
    return None


def _values_agree(a: Side, b: Side) -> tuple[bool, float]:
    gap = relative_gap(a.value, b.value)
    return (gap <= ROUNDING_TOLERANCE or same_to_sig_figs(a.value, b.value, SIG_FIGS)), gap


def r_agreement(a: Side, b: Side) -> Ruling | None:
    """Same frame, same number. The only question left is how differently it was written."""
    if a.value is None or b.value is None:
        return None
    agree, gap = _values_agree(a, b)
    if not agree:
        return None

    if (a.unit_raw or "").strip().lower() != (b.unit_raw or "").strip().lower():
        return Ruling(
            "corroborates",
            "UNIT_SCALE",
            f"The same figure written two ways: {a.value_raw} {a.unit_raw} in "
            f"{a.where()} and {b.value_raw} {b.unit_raw} in {b.where()} both normalise "
            f"to {a.value:,.0f} {a.unit_dim}.",
            divergence=gap,
        )
    if gap > 0:
        return Ruling(
            "corroborates",
            "ROUNDING",
            f"Agree to {SIG_FIGS} significant figures; the {gap:.2%} gap is rounding "
            f"({a.value_raw} against {b.value_raw}).",
            divergence=gap,
        )
    return Ruling(
        "corroborates",
        "EXACT",
        f"Identical: both report {a.value_raw} {a.unit_raw or ''}".strip() + ".",
        divergence=0.0,
    )


def r_sign_convention(a: Side, b: Side) -> Ruling | None:
    """Same magnitude, opposite sign.

    Financial statements write a loss as "(2,491.86)" in a table and as
    "decreased to Rs 2,491.86 million" in the prose two pages later. Both mean
    the same thing; only one carries the sign. Without this rule the pair looks
    like a 200% disagreement, which is the most confident kind of wrong - the
    system reporting a contradiction between a document and itself over a
    typographic convention.

    Deliberately narrow: magnitudes must match to rounding, and one side must
    actually be parenthesised in its raw form. It reconciles rather than
    corroborates, because which sign is intended is a real ambiguity the
    document has left open.
    """
    if a.value is None or b.value is None or a.value == 0 or b.value == 0:
        return None
    if (a.value > 0) == (b.value > 0):
        return None
    if relative_gap(abs(a.value), abs(b.value)) > ROUNDING_TOLERANCE:
        return None
    parenthesised = [s for s in (a, b) if "(" in (s.value_raw or "")]
    if not parenthesised:
        return None
    return Ruling(
        "reconciled",
        "SIGN_CONVENTION",
        f"Identical magnitude reported with opposite signs: {a.value_raw} in {a.where()} "
        f"against {b.value_raw} in {b.where()}. One follows the accounting convention of "
        f"parenthesising a negative, the other states the magnitude in prose. Not a "
        f"disagreement about the number.",
        divergence=0.0,
    )


def r_vintage_revision(a: Side, b: Side) -> Ruling | None:
    """Same frame, different values, and one document is simply older.

    This is the case the brief cares most about. A January advance estimate and
    a November actual are not in conflict; the second replaced the first. What
    makes it decidable is holding the publication date separately from the
    period, so "older" is a fact rather than an inference.
    """
    if a.value is None or b.value is None:
        return None
    ra, rb = stage_rank(a.basis), stage_rank(b.basis)
    gap = relative_gap(a.value, b.value)

    if ra is not None and rb is not None and ra != rb:
        early, late = (a, b) if ra < rb else (b, a)
        return Ruling(
            "reconciled",
            "VINTAGE_REVISION",
            f"Successive vintages of one figure, not a disagreement: "
            f"{early.where()} reports a {_stage_word(early.basis)} of {early.value_raw}, "
            f"and {late.where()} a {_stage_word(late.basis)} of {late.value_raw}. "
            f"The later reading supersedes the earlier ({gap:.1%} revision).",
            divergence=gap,
        )

    # No stage word, but one document predates the other and carries the
    # weaker claim to finality. Only accepted when the gap is small enough to
    # look like a revision rather than a different quantity entirely.
    if a.vintage and b.vintage and a.vintage != b.vintage and gap <= 0.15:
        early, late = (a, b) if a.vintage < b.vintage else (b, a)
        if "restated" in late.basis or "revised" in late.basis:
            return Ruling(
                "reconciled",
                "RESTATEMENT",
                f"{late.where()} explicitly restates the figure "
                f"({early.value_raw} to {late.value_raw}, {gap:.1%}).",
                divergence=gap,
            )
    return None


def _stage_word(basis: list[str]) -> str:
    from .normalize import stage_of

    return (stage_of(basis) or "figure").replace("_", " ")


def r_state_supersession(a: Side, b: Side) -> Ruling | None:
    """A director active in one document and resigned in a later one."""
    if a.claim_type not in ("state", "event") or b.claim_type not in ("state", "event"):
        return None
    left = (a.value_text or a.value_raw or "").strip().lower()
    right = (b.value_text or b.value_raw or "").strip().lower()
    if not left or not right or left == right:
        return None
    if a.vintage and b.vintage and a.vintage != b.vintage:
        early, late = (a, b) if a.vintage < b.vintage else (b, a)
        return Ruling(
            "reconciled",
            "STATE_SUPERSEDED",
            f"A condition that changed rather than a disagreement: {early.where()} "
            f"records \"{early.value_raw}\" and the later {late.where()} records "
            f"\"{late.value_raw}\".",
        )
    return Ruling(
        "contradicts",
        "STATE_CONFLICT",
        f"Two documents of the same vintage record different states: "
        f"\"{a.value_raw}\" against \"{b.value_raw}\".",
        severity=0.6,
    )


def r_contradiction(a: Side, b: Side) -> Ruling | None:
    """Nothing explains the gap. Same entity, same measure, same period, same
    basis, same units - and different numbers."""
    if a.value is None or b.value is None:
        return None
    gap = relative_gap(a.value, b.value)
    conf = min(a.confidence or 0.5, b.confidence or 0.5)
    severity = min(1.0, gap * 2) * (0.5 + 0.5 * conf)
    if a.witness_id == b.witness_id:
        severity *= 0.6  # internal inconsistency; likelier an extraction artefact
    return Ruling(
        "contradicts",
        "CONTRADICTS",
        f"Same entity, measure, period ({a.period_raw or a.period}) and basis, but "
        f"{a.where()} reports {a.value_raw} {a.unit_raw or ''} and {b.where()} reports "
        f"{b.value_raw} {b.unit_raw or ''} - a {gap:.1%} difference with nothing in "
        f"either frame to account for it.",
        divergence=gap,
        severity=round(severity, 3),
    )


# Order is the argument. Cheap structural disqualifications first, then the
# coordinates that explain differences, then agreement, then - only if nothing
# above has claimed the pair - conflict.
CASCADE = [
    r_dimension,
    r_state_supersession,
    r_period_missing,
    r_period_disjoint,
    r_period_nested,
    r_period_partial,
    r_basis_consolidation,
    r_basis_adjustment,
    r_agreement,
    r_sign_convention,
    r_vintage_revision,
    r_contradiction,
]


def adjudicate(a: Side, b: Side) -> Ruling:
    for rule in CASCADE:
        ruling = rule(a, b)
        if ruling is not None:
            return ruling
    return Ruling("distinct", "NO_RULE", "No rule claimed this pair.")
