"""Getting claims out of a passage, and refusing the ones that will not anchor.

The prompt below is the only place in Collate where domain knowledge could have
leaked in, so it is written to be strictly about *reading*, not about finance or
macroeconomics. It never names a metric, a company, a currency or a fiscal
convention. Point it at a lease agreement or a clinical trial and it behaves the
same way: find assertions, quote them exactly, and report the words the document
used rather than the words we wish it had used.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from . import anchor as anchor_mod
from .schemas import Extraction, FrontMatter

SYSTEM = """\
You extract checkable assertions from one passage of a document.

An assertion is checkable when a careful reader, given only this passage, could
confirm or refute it. Numbers with labels, offices people hold, dates things
happened, statements that two names refer to one thing. Not: marketing language,
aspirations, section headings on their own, page furniture, or anything whose
truth needs a document you cannot see.

Rules that matter more than completeness:

1. QUOTE EXACTLY. The quote field is copied character-for-character from the
   passage. Do not reflow table rows, fix spacing, expand abbreviations, or
   repair anything. A quote that has been tidied is worse than no quote,
   because it will be silently discarded downstream.

2. USE THE DOCUMENT'S WORDS. Report the measure, period, unit and basis as the
   passage words them. 'FY24' stays 'FY24'. 'Rs Cr' stays 'Rs Cr'. Do not
   convert, standardise, or translate into any vocabulary of your own. Something
   else handles that, and it can only do so if you preserve the original.

3. REPORT, DO NOT KNOW. If the passage says a figure, extract that figure, even
   where it disagrees with what you believe to be true. Disagreement between
   documents is the point of this system, not a defect to be smoothed over.

4. CARRY THE SCOPING HEADER. When a header is supplied, it governs the numbers
   beneath it. Put its scale into `unit` and its scope into `basis`. A bare
   '82,568' under '(Rs in Million)' has unit 'Rs Million', not ''.

5. PREFER PRECISION TO VOLUME. Twelve claims you would defend beat forty you
   would not. Lower `confidence` when the label sits far from the value, when
   you had to assume a scale, or when the table's shape was ambiguous.

Extract every table row that carries a distinct labelled value; those rows are
usually the densest and most comparable facts in the document.
"""


@dataclass
class ElicitStats:
    passages: int = 0
    proposed: int = 0
    anchored: int = 0
    quarantined: int = 0
    by_reason: dict | None = None

    def as_dict(self) -> dict:
        return {
            "passages": self.passages,
            "proposed": self.proposed,
            "anchored": self.anchored,
            "quarantined": self.quarantined,
            "anchor_rate": round(self.anchored / self.proposed, 4) if self.proposed else None,
            "by_reason": self.by_reason or {},
        }


def read_front_matter(client, pages: list) -> dict:
    """One pass over the opening pages to establish document-level facts.

    Done once, up front, because vintage and default scale are properties of the
    whole document that no individual passage can see. Every claim inherits them.
    """
    head = "\n\n".join(f"[page {p.page_no + 1}]\n{p.text[:3000]}" for p in pages[:6])
    tail = "\n\n".join(f"[page {p.page_no + 1}]\n{p.text[:1200]}" for p in pages[-2:])
    return client.json(
        purpose="front_matter",
        system=(
            "Identify what this document is from its opening pages. Report only what "
            "is printed. The publication date is when the document was issued, never "
            "the period it reports on - a report about 2024 published in 2025 has "
            "vintage 2025."
        ),
        user=f"{head}\n\n[end matter]\n{tail}",
        schema=FrontMatter,
    )


def elicit_passage(client, passage, header_default: str | None) -> list[dict]:
    header = passage.context_header or header_default
    prefix = f"SCOPING HEADER IN FORCE: {header}\n\n" if header else ""
    payload = client.json(
        purpose="elicit",
        system=SYSTEM,
        user=f"{prefix}PASSAGE:\n{passage.text}",
        schema=Extraction,
    )
    return payload.get("claims", []) if isinstance(payload, dict) else []


def elicit_witness(
    client,
    conn,
    witness_id: int,
    pages: list,
    passages: list,
    header_default: str | None = None,
    workers: int = 4,
) -> ElicitStats:
    """Extract, anchor, and persist. Concurrency is bounded by the rate limiter
    inside the client, so `workers` only controls how many requests are queued
    behind it, not how fast we hit the API."""
    stats = ElicitStats(passages=len(passages))
    reasons: dict[str, int] = {}

    page_ids = {p.page_no: p for p in pages}
    passage_ids: dict[str, int] = {}
    for p in passages:
        from .db import insert

        passage_ids[p.hash] = insert(
            conn,
            "passage",
            witness_id=witness_id,
            page_start=p.page_start,
            page_end=p.page_end,
            text=p.text,
            context_header=p.context_header,
            hash=p.hash,
            char_count=len(p.text),
        )
    conn.commit()

    def work(p):
        try:
            return p, elicit_passage(client, p, header_default), None
        except Exception as exc:  # noqa: BLE001 - recorded per passage, never fatal
            return p, [], exc

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for passage, raw_claims, err in pool.map(work, passages):
            pid = passage_ids.get(passage.hash)
            if err is not None:
                _quarantine(conn, witness_id, pid, "passage_failed", {"error": str(err)[:500]})
                reasons["passage_failed"] = reasons.get("passage_failed", 0) + 1
                stats.quarantined += 1
                continue

            candidates = [page_ids[n] for n in passage.pages if n in page_ids]
            for rc in raw_claims:
                stats.proposed += 1
                quote = (rc.get("quote") or "").strip()
                hint = (rc.get("page") or 1) - 1

                loc = anchor_mod.locate(quote, candidates, hint_page=hint)
                if loc is None:
                    _quarantine(conn, witness_id, pid, "unanchored", rc)
                    reasons["unanchored"] = reasons.get("unanchored", 0) + 1
                    stats.quarantined += 1
                    continue

                if rc.get("claim_type") == "quantity" and not anchor_mod.value_present(
                    rc.get("value", ""), quote
                ):
                    _quarantine(conn, witness_id, pid, "value_not_in_quote", rc)
                    reasons["value_not_in_quote"] = reasons.get("value_not_in_quote", 0) + 1
                    stats.quarantined += 1
                    continue

                _persist(conn, witness_id, pid, rc, loc)
                stats.anchored += 1

    conn.commit()
    stats.by_reason = reasons
    return stats


def _quarantine(conn, witness_id: int, passage_id, reason: str, payload: dict) -> None:
    from .db import insert

    insert(
        conn,
        "quarantine",
        witness_id=witness_id,
        passage_id=passage_id,
        reason=reason,
        payload=json.dumps(payload)[:4000],
    )


def _persist(conn, witness_id: int, passage_id, rc: dict, loc) -> int:
    from .db import insert

    claim_id = insert(
        conn,
        "claim",
        witness_id=witness_id,
        passage_id=passage_id,
        claim_type=rc.get("claim_type") or "quantity",
        quote=rc.get("quote", ""),
        entity_raw=rc.get("entity"),
        measure_raw=rc.get("measure"),
        value_raw=rc.get("value"),
        unit_raw=rc.get("unit"),
        period_raw=rc.get("period"),
        basis_raw=rc.get("basis"),
        qualifiers=rc.get("qualifiers") or [],
        confidence=rc.get("confidence"),
    )
    insert(
        conn,
        "anchor",
        claim_id=claim_id,
        witness_id=witness_id,
        page_no=loc.page_no,
        char_start=loc.char_start,
        char_end=loc.char_end,
        x0=loc.bbox[0],
        y0=loc.bbox[1],
        x1=loc.bbox[2],
        y1=loc.bbox[3],
        match_kind=loc.match_kind,
    )
    return claim_id
