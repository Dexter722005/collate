"""HTTP surface: upload documents, read the apparatus, see the evidence.

Server-rendered Jinja with htmx for the interactive parts. No bundler, no build
step, no node_modules - `uvicorn collate.api:app` and it runs. For a three-pane
reading tool that is not a compromise: the panes are documents, and documents
are what server rendering is good at.

The evidence endpoint is the one that matters. It renders the actual PDF page
with the anchor rectangle drawn on it, so a reader can see that a claim came
from where the system says it did. Every other view is a claim about the data;
that one is the data.
"""

from __future__ import annotations

import io
import json
import shutil
from datetime import date
from pathlib import Path

import fitz
from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from . import collation, db, llm, normalize, pipeline

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
UPLOADS = DATA / "uploads"
WEB = Path(__file__).resolve().parent / "web"

app = FastAPI(title="Collate")
app.mount("/static", StaticFiles(directory=str(WEB)), name="static")
templates = Jinja2Templates(directory=str(WEB))
conn = db.connect()

VERDICT_ORDER = ["contradicts", "reconciled", "corroborates", "distinct"]


def _find_pdf(filename: str) -> Path | None:
    for candidate in DATA.rglob(filename):
        return candidate
    return None


# -- views ------------------------------------------------------------------


@app.get("/", response_class=HTMLResponse)
def apparatus(request: Request, verdict: str = "contradicts", rule: str = "", q: str = ""):
    where, args = ["1=1"], []
    if verdict and verdict != "all":
        where.append("v.verdict = ?")
        args.append(verdict)
    if rule:
        where.append("v.rule_id = ?")
        args.append(rule)
    if q:
        where.append("(ca.measure_raw LIKE ? OR e.canonical_name LIKE ?)")
        args += [f"%{q}%", f"%{q}%"]

    rows = db.rows(
        conn,
        f"""
        SELECT v.*, ca.measure_raw AS measure, e.canonical_name AS entity,
               ca.value_raw AS a_value, ca.unit_raw AS a_unit, ca.period_raw AS a_period,
               cb.value_raw AS b_value, cb.unit_raw AS b_unit, cb.period_raw AS b_period,
               wa.title AS a_witness, wb.title AS b_witness,
               wa.vintage_date AS a_vintage, wb.vintage_date AS b_vintage
        FROM verdict v
        JOIN claim ca ON ca.id = v.claim_a
        JOIN claim cb ON cb.id = v.claim_b
        JOIN witness wa ON wa.id = ca.witness_id
        JOIN witness wb ON wb.id = cb.witness_id
        LEFT JOIN entity e ON e.id = ca.entity_id
        WHERE {' AND '.join(where)}
        ORDER BY COALESCE(v.severity, 0) DESC, ABS(COALESCE(v.divergence, 0)) DESC, v.id
        LIMIT 300
        """,
        args,
    )

    return templates.TemplateResponse(
        "apparatus.html",
        {
            "request": request,
            "rows": rows,
            "verdict": verdict,
            "rule": rule,
            "q": q,
            "witnesses": db.rows(conn, "SELECT * FROM witness ORDER BY vintage_date, id"),
            "counts": {
                r["verdict"]: r["c"]
                for r in db.rows(conn, "SELECT verdict, COUNT(*) c FROM verdict GROUP BY verdict")
            },
            "rules": db.rows(
                conn,
                "SELECT rule_id, COUNT(*) c FROM verdict GROUP BY rule_id ORDER BY c DESC",
            ),
            "stats": _stats(),
            "order": VERDICT_ORDER,
        },
    )


@app.get("/verdict/{vid}", response_class=HTMLResponse)
def verdict_detail(request: Request, vid: int):
    v = db.one(conn, "SELECT * FROM verdict WHERE id = ?", (vid,))
    if not v:
        raise HTTPException(404, "no such verdict")
    a = _claim(v["claim_a"])
    b = _claim(v["claim_b"])
    return templates.TemplateResponse(
        "verdict.html",
        {
            "request": request,
            "v": v,
            "a": a,
            "b": b,
            "llm": json.loads(v["llm_note"]) if v["llm_note"] else None,
            "diff": _frame_diff(a, b),
        },
    )


def _claim(cid: int) -> dict:
    row = db.one(
        conn,
        """SELECT c.*, w.title AS witness_title, w.filename, w.vintage_date, w.publisher,
                  a.page_no, a.match_kind, a.x0, a.y0, a.x1, a.y1,
                  m.canonical_label AS measure_label, e.canonical_name AS entity_name,
                  p.printed_label
           FROM claim c
           JOIN witness w ON w.id = c.witness_id
           LEFT JOIN anchor a ON a.claim_id = c.id
           LEFT JOIN measure m ON m.id = c.measure_id
           LEFT JOIN entity e ON e.id = c.entity_id
           LEFT JOIN page p ON p.witness_id = c.witness_id AND p.page_no = a.page_no
           WHERE c.id = ?""",
        (cid,),
    )
    if not row:
        raise HTTPException(404, "no such claim")
    d = dict(row)
    d["basis_list"] = json.loads(d["basis"]) if d["basis"] else []
    return d


def _frame_diff(a: dict, b: dict) -> list[dict]:
    """Which coordinate actually differs. This is the reasoning, made visible."""
    fields = [
        ("entity", a.get("entity_name"), b.get("entity_name")),
        ("measure", a.get("measure_raw"), b.get("measure_raw")),
        ("period", a.get("period_raw") or "-", b.get("period_raw") or "-"),
        ("period (resolved)",
         f"{a.get('period_start')} to {a.get('period_end')}",
         f"{b.get('period_start')} to {b.get('period_end')}"),
        ("basis", ", ".join(a["basis_list"]) or "-", ", ".join(b["basis_list"]) or "-"),
        ("unit", f"{a.get('unit_raw') or '-'} ({a.get('unit_dim')})",
                 f"{b.get('unit_raw') or '-'} ({b.get('unit_dim')})"),
        ("value", a.get("value_raw"), b.get("value_raw")),
        ("value (normalised)",
         f"{a['value_num']:,.0f}" if a.get("value_num") is not None else "-",
         f"{b['value_num']:,.0f}" if b.get("value_num") is not None else "-"),
        ("published", a.get("vintage_date") or "-", b.get("vintage_date") or "-"),
    ]
    return [
        {"field": f, "a": x, "b": y, "same": str(x) == str(y)}
        for f, x, y in fields
    ]


@app.get("/evidence/{claim_id}.png")
def evidence(claim_id: int, zoom: float = 2.0):
    """The rendered source page with the anchor drawn on it."""
    c = _claim(claim_id)
    if c["page_no"] is None:
        raise HTTPException(404, "claim has no anchor")
    path = _find_pdf(c["filename"])
    if path is None:
        raise HTTPException(404, f"source file {c['filename']} not found under data/")

    doc = fitz.open(str(path))
    try:
        page = doc[c["page_no"]]
        if c["x1"] and c["y1"]:
            pad = 2.0
            rect = fitz.Rect(c["x0"] - pad, c["y0"] - pad, c["x1"] + pad, c["y1"] + pad)
            page.draw_rect(
                rect, color=(0.63, 0.20, 0.13), fill=(0.99, 0.87, 0.42),
                fill_opacity=0.30, width=1.4,
            )
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
        return Response(pix.tobytes("png"), media_type="image/png")
    finally:
        doc.close()


@app.get("/quarantine", response_class=HTMLResponse)
def quarantine(request: Request):
    """The claims that did not survive anchoring. Deliberately a first-class page."""
    return templates.TemplateResponse(
        "quarantine.html",
        {
            "request": request,
            "rows": db.rows(
                conn,
                """SELECT q.*, w.title FROM quarantine q JOIN witness w ON w.id=q.witness_id
                   ORDER BY q.id DESC LIMIT 200""",
            ),
            "by_reason": db.rows(
                conn, "SELECT reason, COUNT(*) c FROM quarantine GROUP BY reason ORDER BY c DESC"
            ),
            "sparse": db.rows(
                conn,
                """SELECT w.title, COUNT(*) c FROM page p JOIN witness w ON w.id=p.witness_id
                   WHERE p.is_sparse=1 GROUP BY w.id ORDER BY c DESC""",
            ),
            "stats": _stats(),
        },
    )


# -- machine-readable -------------------------------------------------------


def _stats() -> dict:
    g = lambda sql: (db.one(conn, sql) or {"c": 0})["c"]  # noqa: E731
    return {
        "witnesses": g("SELECT COUNT(*) c FROM witness"),
        "claims": g("SELECT COUNT(*) c FROM claim"),
        "framed": g("SELECT COUNT(*) c FROM claim WHERE frame_key IS NOT NULL"),
        "quarantined": g("SELECT COUNT(*) c FROM quarantine"),
        "measures": g("SELECT COUNT(*) c FROM measure"),
        "verdicts": g("SELECT COUNT(*) c FROM verdict"),
    }


@app.get("/api/export")
def export():
    """Everything, as JSON. Committed to samples/ so the apparatus can be
    inspected without running the pipeline or holding an API key."""
    out = {"generated": date.today().isoformat(), "stats": _stats(), "verdicts": []}
    for v in db.rows(conn, "SELECT * FROM verdict ORDER BY COALESCE(severity,0) DESC LIMIT 2000"):
        a, b = _claim(v["claim_a"]), _claim(v["claim_b"])
        out["verdicts"].append(
            {
                "verdict": v["verdict"],
                "rule": v["rule_id"],
                "explanation": v["explanation"],
                "divergence": v["divergence"],
                "llm_note": json.loads(v["llm_note"]) if v["llm_note"] else None,
                "readings": [
                    {
                        "witness": s["witness_title"],
                        "published": s["vintage_date"],
                        "page": (s["page_no"] or 0) + 1,
                        "measure": s["measure_raw"],
                        "value": s["value_raw"],
                        "unit": s["unit_raw"],
                        "period": s["period_raw"],
                        "basis": s["basis_list"],
                        "quote": s["quote"],
                        "anchor": s["match_kind"],
                    }
                    for s in (a, b)
                ],
            }
        )
    return JSONResponse(out)


@app.post("/upload")
async def upload(file: UploadFile):
    """Add a document to the live apparatus. Ingest, frame, adjudicate, in that
    order - and only the new claims are compared, so this stays cheap as the
    corpus grows."""
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "expected a .pdf")
    UPLOADS.mkdir(parents=True, exist_ok=True)
    dest = UPLOADS / file.filename
    with open(dest, "wb") as fh:
        shutil.copyfileobj(file.file, fh)

    client = llm.Gemini(conn=conn)
    result = pipeline.ingest(conn, client, dest)
    if not result.get("skipped"):
        normalize.normalize_witness(conn, result["witness_id"])
        result["collation"] = collation.run(conn, client=client, escalate=True)
    return JSONResponse(result)
