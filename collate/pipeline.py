"""Ingest one document into the apparatus.

Idempotent on file content: re-running against bytes already in the corpus is a
no-op that costs nothing and calls nothing. That property is what makes adding
the seventh document cheap, and it falls out of hashing rather than out of any
clever incremental machinery.
"""

from __future__ import annotations

import json
from pathlib import Path

from . import elicit, segment
from .db import insert, one


def ingest(conn, client, pdf_path: str | Path, *, force: bool = False) -> dict:
    path = Path(pdf_path)
    digest = segment.sha256_file(path)

    existing = one(conn, "SELECT id, status FROM witness WHERE sha256 = ?", (digest,))
    if existing and not force:
        return {"witness_id": existing["id"], "skipped": True, "reason": "already ingested"}

    pages = segment.read_pages(path)
    if not pages:
        raise ValueError(f"{path.name}: no readable pages")

    front = {}
    try:
        front = elicit.read_front_matter(client, pages)
    except Exception as exc:  # noqa: BLE001 - a document without front matter still works
        front = {"title": path.stem, "_error": str(exc)[:300]}

    witness_id = insert(
        conn,
        "witness",
        sha256=digest,
        filename=path.name,
        title=front.get("title") or path.stem,
        publisher=front.get("publisher"),
        doc_type=front.get("doc_type"),
        vintage_date=(front.get("vintage_date") or None),
        default_currency=front.get("default_currency"),
        default_scale=front.get("default_scale"),
        page_count=len(pages),
        status="segmented",
        front_matter=json.dumps(front),
    )

    for p in pages:
        insert(
            conn,
            "page",
            witness_id=witness_id,
            page_no=p.page_no,
            printed_label=p.printed_label,
            text=p.text,
            char_count=len(p.text),
            text_density=round(p.density, 6),
            is_sparse=int(p.is_sparse),
        )
        for b in p.blocks:
            insert(
                conn,
                "block",
                witness_id=witness_id,
                page_no=p.page_no,
                idx=b.idx,
                kind=b.kind,
                text=b.text,
                char_start=b.char_start,
                char_end=b.char_end,
                x0=b.bbox[0], y0=b.bbox[1], x1=b.bbox[2], y1=b.bbox[3],
            )
    conn.commit()

    passages = segment.build_passages(pages)
    header_default = None
    if front.get("default_currency") or front.get("default_scale"):
        header_default = " ".join(
            x for x in [front.get("default_currency"), front.get("default_scale")] if x
        )

    stats = elicit.elicit_witness(
        client, conn, witness_id, pages, passages, header_default=header_default
    )

    sparse = sum(1 for p in pages if p.is_sparse)
    payload = {
        "witness_id": witness_id,
        "skipped": False,
        "title": front.get("title"),
        "vintage": front.get("vintage_date"),
        "pages": len(pages),
        "sparse_pages": sparse,
        **stats.as_dict(),
    }
    insert(conn, "run_log", witness_id=witness_id, stage="elicit", stats=json.dumps(payload))
    conn.execute("UPDATE witness SET status='elicited' WHERE id=?", (witness_id,))
    conn.commit()
    return payload
