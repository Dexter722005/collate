"""Building a frame from a raw claim.

A frame is the full set of coordinates a claim sits at:

    lemma  = (entity, measure)          what is being talked about
    frame  = lemma + period + basis + unit dimension

Two claims in the same frame are making the same assertion and must agree.
Two claims in different frames are not in conflict at all - the coordinate that
differs is the explanation, and handing that coordinate to the rule cascade is
the entire purpose of this module.

On the basis vocabulary below: it is a *linguistic* vocabulary, not a domain
one. "consolidated", "provisional", "seasonally adjusted" are words documents
use to qualify any figure, in any field. Unrecognised qualifiers are not
discarded - they pass through as free tokens, so a document introducing a
qualifier nobody anticipated still gets a distinct frame for it.
"""

from __future__ import annotations

import json
import re

from . import period as period_mod
from . import quantity as quantity_mod
from . import registry

# Families of qualifier, canonicalised so "Consolidated" and "consolidated
# financial statements" land on the same token. Order matters only within a
# family: the first pattern that matches wins.
BASIS_FAMILIES: dict[str, list[tuple[str, str]]] = {
    "consolidation": [
        (r"\bconsolidat", "consolidated"),
        (r"\bstandalone|\bseparate financial", "standalone"),
        (r"\bsegment\b", "segment"),
    ],
    "assurance": [
        (r"\bunaudited|\blimited review", "unaudited"),
        (r"\baudited", "audited"),
    ],
    "vintage_stage": [
        (r"\bfirst advance|\badvance estimate", "advance_estimate"),
        (r"\bsecond advance", "second_advance_estimate"),
        (r"\bprovisional", "provisional"),
        (r"\brevised", "revised"),
        (r"\brestated|\bre-?stated", "restated"),
        (r"\bproforma|\bpro\s?forma", "proforma"),
        (r"\bproject|\bforecast|\bprojected", "projected"),
        (r"\bestimate|\bestimated|\be\b(?!\w)", "estimate"),
        (r"\bactual", "actual"),
    ],
    "adjustment": [
        (r"\bseasonally adjusted", "seasonally_adjusted"),
        (r"\bconstant price|\bat constant|\breal terms|\breal\b", "real"),
        (r"\bcurrent price|\bnominal\b", "nominal"),
        (r"\bannualis|\bannualiz", "annualised"),
    ],
    "aggregation": [
        (r"\bgross\b", "gross"),
        (r"\bnet\b", "net"),
        (r"\bper capita", "per_capita"),
    ],
}

# Ordered strongest-to-weakest. A revision supersedes an estimate; this is the
# ladder VINTAGE_REVISION climbs.
STAGE_ORDER = [
    "advance_estimate", "second_advance_estimate", "estimate", "projected",
    "provisional", "revised", "restated", "actual",
]

_TOKEN = re.compile(r"[a-z][a-z_]{2,}")


# Words that turn up in free-text qualifiers and mean nothing as a basis. Not a
# domain vocabulary - just prose that survived tokenising.
_BASIS_NOISE = {
    "the", "and", "for", "with", "from", "that", "this", "which", "were", "was",
    "has", "have", "been", "are", "its", "their", "there", "value", "values",
    "figure", "figures", "number", "numbers", "data", "based", "including",
    "percent", "cent", "year", "years", "period", "memo", "term", "total",
    "amount", "level", "rate", "growth", "over", "under", "above", "below",
    "assumption", "assumptions", "baseline", "projected", "medium", "ion",
}
_MIN_FREE_TOKEN = 5
_MAX_FREE_TOKENS = 2


def basis_tokens(basis_raw: str | None, qualifiers: list[str] | None) -> list[str]:
    """Canonical qualifier tokens for a claim, sorted so the frame key is stable.

    Canonical families are matched against the basis AND the free-text
    qualifiers, because a document will say "restated" in either place.

    Unrecognised words are only taken from `basis_raw`. `qualifiers` is where
    the model puts whole clauses, and tokenising those was pulling "assumption",
    "baseline", "tariffs" and even "ion" into the frame - noise that looked
    absurd in the UI and, far worse, split frames that should have matched, so
    two readings of one figure stopped being comparable. The vocabulary stays
    open, but only where the document was actually naming a basis.
    """
    quals = list(qualifiers or [])
    blob = " ".join(filter(None, [basis_raw or ""] + quals)).lower()
    if not blob.strip():
        return []

    found: set[str] = set()
    for patterns in BASIS_FAMILIES.values():
        for pattern, token in patterns:
            if re.search(pattern, blob):
                found.add(token)
                break

    stated = (basis_raw or "").lower()
    for pattern, _ in [pt for fam in BASIS_FAMILIES.values() for pt in fam]:
        stated = re.sub(pattern, " ", stated)
    free = [
        w for w in _TOKEN.findall(stated)
        if len(w) >= _MIN_FREE_TOKEN and w not in _BASIS_NOISE
    ]
    found.update(sorted(set(free))[:_MAX_FREE_TOKENS])

    return sorted(found)


def stage_of(tokens: list[str]) -> str | None:
    for stage in reversed(STAGE_ORDER):
        if stage in tokens:
            return stage
    return None


def stage_rank(tokens: list[str]) -> int | None:
    stage = stage_of(tokens)
    return STAGE_ORDER.index(stage) if stage else None


def normalize_claim(
    conn, claim: dict, witness: dict,
    index: "registry.MeasureIndex | None" = None,
    vec=None,
) -> dict:
    """Resolve one claim's frame. Returns the fields to write back onto the row."""
    qualifiers = claim.get("qualifiers")
    if isinstance(qualifiers, str):
        try:
            qualifiers = json.loads(qualifiers)
        except ValueError:
            qualifiers = [qualifiers]

    front = {}
    if witness.get("front_matter"):
        try:
            front = json.loads(witness["front_matter"])
        except ValueError:
            front = {}

    entity_id = registry.resolve_entity(
        conn,
        claim.get("entity_raw") or "",
        front.get("primary_entity") or witness.get("publisher"),
        witness.get("id"),
    )

    qty = quantity_mod.parse(
        claim.get("value_raw"),
        claim.get("unit_raw"),
        fallback_currency=witness.get("default_currency") or None,
    )
    dimension = qty.dimension if qty else None

    # A qualifier written into the measure's name ("Revenue from Operations
    # (Consolidated)") is a basis, and has to be moved before the measure is
    # resolved - otherwise it both splits the registry and hides itself from
    # the basis rules.
    measure_clean, inline_basis = registry.split_qualifiers(claim.get("measure_raw") or "")

    idx = index or registry.MeasureIndex(conn)
    measure_id = idx.resolve(measure_clean, dimension, witness.get("id"), vec=vec)

    per = period_mod.parse(claim.get("period_raw"))
    tokens = basis_tokens(claim.get("basis_raw"), list(qualifiers or []) + inline_basis)

    lemma_key = f"{entity_id}:{measure_id}" if entity_id and measure_id else None
    period_part = f"{per.start.isoformat()}..{per.end.isoformat()}" if per else "none"
    frame_key = (
        f"{lemma_key}|{period_part}|{','.join(tokens) or 'none'}|{dimension or 'none'}"
        if lemma_key
        else None
    )

    return {
        "entity_id": entity_id,
        "measure_id": measure_id,
        "value_num": qty.value if qty else None,
        "value_text": (claim.get("value_raw") if qty is None else None),
        "unit_dim": dimension,
        "unit_scale": qty.scale if qty else None,
        "period_start": per.start.isoformat() if per else None,
        "period_end": per.end.isoformat() if per else None,
        "period_grain": per.grain if per else None,
        "basis": json.dumps(tokens),
        "lemma_key": lemma_key,
        "frame_key": frame_key,
    }


def normalize_witness(conn, witness_id: int) -> int:
    """Normalise every claim of one witness. Returns how many were framed."""
    witness = dict(conn.execute("SELECT * FROM witness WHERE id=?", (witness_id,)).fetchone())
    claims = conn.execute(
        "SELECT * FROM claim WHERE witness_id=? AND frame_key IS NULL", (witness_id,)
    ).fetchall()
    if not claims:
        return 0

    # Encode every distinct measure phrase in one pass. Batching here rather
    # than per-claim is the difference between seconds and an hour on a full
    # corpus, and the model is far more efficient on a batch than on singles.
    index = registry.MeasureIndex(conn)
    phrases = sorted(
        {registry.split_qualifiers(c["measure_raw"] or "")[0] for c in claims} - {""}
    )
    vectors = dict(zip(phrases, registry.embed(phrases))) if phrases else {}

    framed = 0
    for row in claims:
        phrase = registry.split_qualifiers(row["measure_raw"] or "")[0]
        fields = normalize_claim(
            conn, dict(row), witness, index=index, vec=vectors.get(phrase)
        )
        conn.execute(
            "UPDATE claim SET entity_id=?, measure_id=?, value_num=?, value_text=?, unit_dim=?,"
            " unit_scale=?, period_start=?, period_end=?, period_grain=?, basis=?, lemma_key=?,"
            " frame_key=? WHERE id=?",
            (
                fields["entity_id"], fields["measure_id"], fields["value_num"],
                fields["value_text"], fields["unit_dim"], fields["unit_scale"],
                fields["period_start"], fields["period_end"], fields["period_grain"],
                fields["basis"], fields["lemma_key"], fields["frame_key"], row["id"],
            ),
        )
        framed += 1 if fields["frame_key"] else 0
    conn.commit()
    return framed
