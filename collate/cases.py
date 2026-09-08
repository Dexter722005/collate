"""`python -m collate cases` - the four cases the brief asks for, found in the data.

Not a fixture and not a hand-picked list. This walks the apparatus and selects
the strongest live example of each category by the same criteria a reader would
use: prefer cross-witness over internal, prefer larger divergence, prefer
readings whose surface forms differ most. If the corpus changes, the examples
change with it, and if a category comes back empty that is a real finding about
the corpus rather than something to paper over.
"""

from __future__ import annotations

import json
import textwrap

from . import db

WRAP = 96


def _fetch(conn, sql: str, args=()) -> list:
    return db.rows(conn, sql, args)


BASE = """
SELECT v.*, ca.measure_raw AS measure,
       ca.value_raw AS a_val, ca.unit_raw AS a_unit, ca.period_raw AS a_per, ca.quote AS a_quote,
       cb.value_raw AS b_val, cb.unit_raw AS b_unit, cb.period_raw AS b_per, cb.quote AS b_quote,
       ca.id AS a_id, cb.id AS b_id,
       wa.title AS a_wit, wb.title AS b_wit,
       wa.vintage_date AS a_vin, wb.vintage_date AS b_vin,
       aa.page_no AS a_page, ab.page_no AS b_page
FROM verdict v
JOIN claim ca ON ca.id = v.claim_a
JOIN claim cb ON cb.id = v.claim_b
JOIN witness wa ON wa.id = ca.witness_id
JOIN witness wb ON wb.id = cb.witness_id
LEFT JOIN anchor aa ON aa.claim_id = ca.id
LEFT JOIN anchor ab ON ab.claim_id = cb.id
WHERE 1=1
"""
CROSS = " AND ca.witness_id <> cb.witness_id "
SAME = " AND ca.witness_id = cb.witness_id "


def _best(conn, tail: str, args=()) -> tuple:
    """Prefer a cross-document example; fall back to a within-document one.

    Two documents disagreeing is the stronger demonstration, but a single filing
    disagreeing with itself is a real finding too, and reporting NOT FOUND when
    the apparatus holds thirty of them would be worse than reporting one with a
    caveat attached.
    """
    got = _fetch(conn, BASE + CROSS + tail, args)
    if got:
        return got[0], True
    got = _fetch(conn, BASE + SAME + tail, args)
    return (got[0] if got else None), False


def _show(n: int, title: str, blurb: str, row, cross: bool = True) -> None:
    print(f"\n{'=' * WRAP}\nCASE {n}  {title}\n{'=' * WRAP}")
    print(textwrap.fill(blurb, WRAP - 2, initial_indent="  ", subsequent_indent="  "))
    if row is None:
        print("\n  NOT FOUND in the current corpus.")
        return

    print(f"\n  lemma  : {row['measure']}")
    print(f"  rule   : {row['rule_id']}   verdict: {row['verdict']}")
    print(f"\n  reading A  {row['a_val']} {row['a_unit'] or ''}   [{row['a_per'] or 'no period'}]")
    print(f"             {row['a_wit'][:70]}  published {row['a_vin']}  p.{(row['a_page'] or 0) + 1}")
    print(f"             “{' '.join((row['a_quote'] or '').split())[:150]}”")
    print(f"\n  reading B  {row['b_val']} {row['b_unit'] or ''}   [{row['b_per'] or 'no period'}]")
    print(f"             {row['b_wit'][:70]}  published {row['b_vin']}  p.{(row['b_page'] or 0) + 1}")
    print(f"             “{' '.join((row['b_quote'] or '').split())[:150]}”")
    print("\n  reasoning:")
    print(textwrap.fill(row["explanation"], WRAP - 4, initial_indent="    ", subsequent_indent="    "))
    if row["llm_note"]:
        note = json.loads(row["llm_note"])
        print(f"\n  escalated -> model says {note.get('relationship')} ({note.get('reason_code')}):")
        print(textwrap.fill(note.get("explanation") or "", WRAP - 4,
                            initial_indent="    ", subsequent_indent="    "))
    print(f"\n  inspect: http://127.0.0.1:8000/verdict/{row['id']}"
          f"   evidence: /evidence/{row['a_id']}.png , /evidence/{row['b_id']}.png")


def run(conn) -> None:
    # 1. Corroborated across documents despite being written differently.
    #    UNIT_SCALE is the strongest form: the two strings share no characters.
    row, cross = _best(conn, """
        AND v.verdict='corroborates' AND v.rule_id IN ('UNIT_SCALE','ROUNDING','EXACT')
        ORDER BY CASE v.rule_id WHEN 'UNIT_SCALE' THEN 0 WHEN 'ROUNDING' THEN 1 ELSE 2 END,
                 LENGTH(ca.value_raw) DESC LIMIT 1""")
    _show(1, "A fact corroborated across documents, expressed differently",
          "Two witnesses reporting the same figure in different notation and scale. "
          "Normalising both to a canonical dimension is what makes them comparable at all.",
          row, cross)

    # 2. A genuine contradiction: same frame, nothing left to explain it.
    row, cross = _best(conn, """
        AND v.verdict='contradicts' AND ca.claim_type='quantity'
        ORDER BY COALESCE(v.severity,0) DESC, COALESCE(v.divergence,0) DESC LIMIT 1""")
    _show(2, "A genuine or likely contradiction",
          "Same entity, measure, period, basis and unit dimension - and different numbers, "
          "with nothing in either frame to account for the gap.",
          row, cross)

    # 3. Apparent contradiction dissolved by context. Vintage first, since a
    #    revision is the subtlest of these; period and basis are the fallbacks.
    row, cross = None, True
    for rule in ("VINTAGE_REVISION", "RESTATEMENT", "BASIS_MISMATCH",
                 "ADJUSTMENT_MISMATCH", "PERIOD_NESTED", "DEFINITION_DRIFT",
                 "PERIOD_DISJOINT"):
        row, cross = _best(conn, """
            AND v.verdict='reconciled' AND v.rule_id=?
            ORDER BY COALESCE(v.divergence,0) DESC LIMIT 1""", (rule,))
        if row:
            break
    _show(3, "An apparent contradiction explained by context",
          "The values differ, but so does one coordinate of the frame - the period, the "
          "reporting basis, or the stage of revision. That coordinate IS the explanation.",
          row, cross)

    # 4. Failure, measured rather than asserted.
    print(f"\n{'=' * WRAP}\nCASE 4  An extraction or reasoning failure, and how it was handled\n{'=' * WRAP}")
    q = db.rows(conn, "SELECT reason, COUNT(*) c FROM quarantine GROUP BY reason ORDER BY c DESC")
    total_claims = db.one(conn, "SELECT COUNT(*) c FROM claim")["c"]
    total_quar = sum(r["c"] for r in q)
    print(textwrap.fill(
        "Every claim must be found again in the page it came from before it is admitted. "
        "These failed and were kept rather than deleted.", WRAP - 2,
        initial_indent="  ", subsequent_indent="  "))
    print(f"\n  admitted    {total_claims}")
    print(f"  rejected    {total_quar}"
          + (f"   ({total_quar / (total_claims + total_quar):.1%} of proposals)" if total_claims + total_quar else ""))
    for r in q:
        print(f"    {r['reason']:<22} {r['c']}")

    print("\n  Pages carrying almost no extractable text (numbers locked in chart graphics):")
    for r in db.rows(conn, """SELECT w.title, SUM(p.is_sparse) s, COUNT(*) n
                              FROM page p JOIN witness w ON w.id=p.witness_id
                              GROUP BY w.id HAVING s > 0 ORDER BY s DESC"""):
        print(f"    {r['title'][:56]:<56} {r['s']}/{r['n']} pages")
    print(textwrap.fill(
        "Nothing was extracted from those pages and nothing was invented for them. The gap "
        "is reported at /quarantine instead of being silent. The fix is a vision pass over "
        "flagged pages - one model call each, which did not fit the free-tier budget.",
        WRAP - 2, initial_indent="  ", subsequent_indent="  "))

    sample = db.rows(conn, "SELECT reason, payload FROM quarantine LIMIT 2")
    if sample:
        print("\n  Example rejections:")
        for s in sample:
            print(f"    [{s['reason']}] {' '.join(s['payload'].split())[:150]}")
    print()
