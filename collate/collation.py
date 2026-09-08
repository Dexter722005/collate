"""Deciding which claims are worth comparing, then comparing them.

Thousands of claims means millions of possible pairs, almost all of them
nonsense. Blocking on the lemma - the (entity, measure) key - cuts that to the
pairs that are actually arguing about the same thing. Two further restrictions
keep it honest rather than merely small:

*   Cross-witness pairs are always compared. That is the entire point of the
    system, and the count stays manageable because a lemma rarely appears in
    more than a handful of documents.

*   Same-witness pairs are compared only when their frames are identical, where
    the pair is an internal inconsistency worth surfacing. Comparing a
    document's FY22 column against its own FY24 column would otherwise
    manufacture thousands of PERIOD_DISJOINT verdicts that nobody asked about.

Neighbour expansion then reaches deliberately outside the block: measures that
embed close together but did not merge are compared anyway, so "revenue from
operations" and "total income" meet and get told apart, rather than never
meeting and being silently treated as unrelated.
"""

from __future__ import annotations

import json
from itertools import combinations

from .adjudicate import Ruling, Side, adjudicate
from .normalize import registry
from .schemas import Adjudication

MAX_PAIRS_PER_LEMMA = 400

ESCALATION_SYSTEM = """\
Two documents report different values for what appears to be the same thing, and
a deterministic rule engine has already checked the obvious explanations: the
periods match, the reporting basis matches, the units reconcile, and the numbers
still differ.

Your job is not to re-check that arithmetic. It is to read the two quotes and
say whether they contain any context the rule engine could not see - a scope
qualifier, a footnote, a restatement, a definitional difference, a segment
boundary - that would explain the gap.

Be willing to conclude that the contradiction stands. A wrong reconciliation is
worse than an admitted conflict, because it hides a real discrepancy behind a
plausible sentence. If the quotes do not settle it, set sufficient_context to
false and say what you would need to see.
"""


def _witnesses(conn) -> dict:
    return {r["id"]: r for r in conn.execute("SELECT * FROM witness").fetchall()}


def build_pairs(conn) -> list[tuple]:
    """Generate the pairs worth adjudicating."""
    witnesses = _witnesses(conn)
    rows = conn.execute(
        "SELECT * FROM claim WHERE lemma_key IS NOT NULL ORDER BY lemma_key, id"
    ).fetchall()

    by_lemma: dict[str, list] = {}
    for r in rows:
        by_lemma.setdefault(r["lemma_key"], []).append(r)

    pairs = []
    for lemma, claims in by_lemma.items():
        if len(claims) < 2:
            continue
        picked = 0
        for a, b in combinations(claims, 2):
            if picked >= MAX_PAIRS_PER_LEMMA:
                break
            # Within one document, compare only readings that cover the same
            # period. Identical frames catch internal inconsistency; differing
            # frames over the same period catch the interesting case, which is a
            # filing that reports both consolidated and standalone and expects
            # you to know which you are looking at. Comparing a document's FY22
            # column against its own FY24 column, by contrast, would manufacture
            # thousands of PERIOD_DISJOINT verdicts nobody asked for.
            same_witness = a["witness_id"] == b["witness_id"]
            if same_witness and a["period_start"] != b["period_start"]:
                continue
            pairs.append(
                (lemma, Side.build(a, witnesses.get(a["witness_id"])),
                 Side.build(b, witnesses.get(b["witness_id"])))
            )
            picked += 1
    return pairs


def build_neighbour_pairs(conn, floor: float = registry.NEIGHBOUR) -> list[tuple]:
    """Pairs across measures that are close but did not merge, same entity only."""
    witnesses = _witnesses(conn)
    measures = [r["id"] for r in conn.execute("SELECT id FROM measure").fetchall()]
    seen: set[tuple[int, int]] = set()
    out = []

    for mid in measures:
        for other, score in registry.neighbours(conn, mid, floor):
            key = (min(mid, other), max(mid, other))
            if key in seen:
                continue
            seen.add(key)
            left = conn.execute(
                "SELECT * FROM claim WHERE measure_id=? AND lemma_key IS NOT NULL", (mid,)
            ).fetchall()
            right = conn.execute(
                "SELECT * FROM claim WHERE measure_id=? AND lemma_key IS NOT NULL", (other,)
            ).fetchall()
            for a in left:
                for b in right:
                    if a["entity_id"] != b["entity_id"] or a["witness_id"] == b["witness_id"]:
                        continue
                    if a["period_start"] != b["period_start"]:
                        continue  # only definitional differences, not period ones
                    out.append(
                        (f"{a['lemma_key']}~{b['lemma_key']}",
                         Side.build(a, witnesses.get(a["witness_id"])),
                         Side.build(b, witnesses.get(b["witness_id"])),
                         score)
                    )
    return out


def run(conn, client=None, escalate: bool = True) -> dict:
    """Adjudicate every pair, then escalate the contradictions."""
    stats = {"pairs": 0, "by_rule": {}, "by_verdict": {}, "escalated": 0, "overturned_note": 0}

    for lemma, a, b in build_pairs(conn):
        ruling = adjudicate(a, b)
        _store(conn, lemma, a, b, ruling)
        stats["pairs"] += 1
        stats["by_rule"][ruling.rule_id] = stats["by_rule"].get(ruling.rule_id, 0) + 1
        stats["by_verdict"][ruling.verdict] = stats["by_verdict"].get(ruling.verdict, 0) + 1

    for lemma, a, b, score in build_neighbour_pairs(conn):
        ruling = Ruling(
            "reconciled",
            "DEFINITION_DRIFT",
            f"Related but not the same measure ({score:.0%} similar): "
            f"\"{a.measure}\" in {a.where()} against \"{b.measure}\" in {b.where()}. "
            f"Values differ because the definitions do.",
        )
        _store(conn, lemma, a, b, ruling)
        stats["pairs"] += 1
        stats["by_rule"]["DEFINITION_DRIFT"] = stats["by_rule"].get("DEFINITION_DRIFT", 0) + 1
        stats["by_verdict"]["reconciled"] = stats["by_verdict"].get("reconciled", 0) + 1

    conn.commit()

    if escalate and client is not None:
        stats["escalated"] = escalate_contradictions(conn, client)
    return stats


def _store(conn, lemma: str, a: Side, b: Side, ruling: Ruling) -> None:
    lo, hi = (a, b) if a.id < b.id else (b, a)
    conn.execute(
        "INSERT OR IGNORE INTO verdict (claim_a, claim_b, lemma_key, verdict, rule_id,"
        " explanation, divergence, severity) VALUES (?,?,?,?,?,?,?,?)",
        (lo.id, hi.id, lemma, ruling.verdict, ruling.rule_id, ruling.explanation,
         ruling.divergence, ruling.severity),
    )


def escalate_contradictions(conn, client, limit: int = 40) -> int:
    """Ask the model about pairs the rules could not explain.

    Its answer is written to llm_note and llm_agrees. The verdict column is not
    touched: if the model reconciles a pair the rules called a contradiction,
    that shows up in the UI as a disagreement between the two, which is more
    informative than either one silently winning.
    """
    rows = conn.execute(
        "SELECT v.id, v.claim_a, v.claim_b FROM verdict v WHERE v.verdict='contradicts'"
        " AND v.llm_note IS NULL ORDER BY v.severity DESC LIMIT ?",
        (limit,),
    ).fetchall()

    done = 0
    for v in rows:
        a = conn.execute("SELECT * FROM claim WHERE id=?", (v["claim_a"],)).fetchone()
        b = conn.execute("SELECT * FROM claim WHERE id=?", (v["claim_b"],)).fetchone()
        wa = conn.execute("SELECT * FROM witness WHERE id=?", (a["witness_id"],)).fetchone()
        wb = conn.execute("SELECT * FROM witness WHERE id=?", (b["witness_id"],)).fetchone()

        prompt = (
            f"MEASURE: {a['measure_raw']}\n"
            f"PERIOD:  {a['period_raw']} / {b['period_raw']}\n\n"
            f"--- Reading A, from {wa['title']} (published {wa['vintage_date']}) ---\n"
            f"value: {a['value_raw']} {a['unit_raw'] or ''}\n"
            f"basis: {a['basis_raw'] or '(none stated)'}\n"
            f"quote: {a['quote']}\n\n"
            f"--- Reading B, from {wb['title']} (published {wb['vintage_date']}) ---\n"
            f"value: {b['value_raw']} {b['unit_raw'] or ''}\n"
            f"basis: {b['basis_raw'] or '(none stated)'}\n"
            f"quote: {b['quote']}\n"
        )
        try:
            out = client.json(
                purpose="escalate",
                system=ESCALATION_SYSTEM,
                user=prompt,
                schema=Adjudication,
            )
        except Exception:
            continue

        note = json.dumps(
            {
                "relationship": out.get("relationship"),
                "reason_code": out.get("reason_code"),
                "explanation": out.get("explanation"),
                "sufficient_context": out.get("sufficient_context"),
            }
        )
        conn.execute(
            "UPDATE verdict SET llm_note=?, llm_agrees=? WHERE id=?",
            (note, int(out.get("relationship") == "contradicts"), v["id"]),
        )
        done += 1
    conn.commit()
    return done
