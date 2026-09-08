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

import json
import shutil
from datetime import date
from pathlib import Path

import fitz
from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from . import db, jobs

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
UPLOADS = DATA / "uploads"
WEB = Path(__file__).resolve().parent / "web"

app = FastAPI(title="Collate")
app.mount("/static", StaticFiles(directory=str(WEB)), name="static")
templates = Jinja2Templates(directory=str(WEB))
conn = db.connect()

VERDICT_ORDER = ["contradicts", "reconciled", "corroborates", "distinct"]


def short_label(title: str | None, filename: str = "") -> str:
    """A witness name that fits in a table cell.

    Document titles run to "INDIA 2025 ARTICLE IV CONSULTATION-PRESS RELEASE;
    STAFF REPORT; STAFF STATEMENT; AND STATEMENT BY THE EXECUTIVE DIRECTOR FOR
    INDIA", which wraps to six lines in a sidebar and three in a source column.
    Cut at a word boundary rather than mid-word, and keep the head, which is
    where the identifying words are.
    """
    text = " ".join((title or filename or "document").split())
    if len(text) <= 30:
        return text
    out: list[str] = []
    for word in text.replace(";", " ").replace("—", " ").split():
        if sum(len(w) + 1 for w in out) + len(word) > 30:
            break
        out.append(word)
    return " ".join(out) or text[:30]


templates.env.filters["short"] = short_label


def _find_pdf(filename: str) -> Path | None:
    for candidate in DATA.rglob(filename):
        return candidate
    return None


# -- views ------------------------------------------------------------------


# The four cases the brief asks for, surfaced as a starting point rather than
# left for a reader to find among 704 verdicts. Each is a rule that produces a
# genuinely different kind of finding, so this doubles as a tour of the cascade.
HIGHLIGHTS = [
    ("UNIT_SCALE", "Same number, written two ways",
     "Prose in one filing, a table cell in another. No characters in common."),
    ("BASIS_MISMATCH", "Explained by reporting basis",
     "Standalone against consolidated - different figures by construction."),
    ("SIGN_CONVENTION", "Explained by accounting notation",
     "A table's (2,491.86) against the same document's prose 2,491.86."),
    ("CONTRADICTS", "A real disagreement",
     "Same subject, period and units; different numbers, nothing to explain it."),
]


def _lemma_rows(verdict: str, rule: str, q: str, limit: int = 400) -> list:
    """Group the apparatus by lemma - the subject two readings are arguing about.

    The first version of this listed verdicts, which are PAIRS. A subject with
    four readings produces six pairs, so the same measure appeared six times and
    read as duplication. A critical apparatus is organised by lemma, with the
    variant readings gathered underneath; that is the whole metaphor and the
    list should match it.
    """
    where, args = ["c.lemma_key IS NOT NULL"], []
    if verdict and verdict != "all":
        where.append(
            "EXISTS (SELECT 1 FROM verdict v WHERE v.lemma_key = c.lemma_key AND v.verdict = ?)"
        )
        args.append(verdict)
    if rule:
        where.append(
            "EXISTS (SELECT 1 FROM verdict v WHERE v.lemma_key = c.lemma_key AND v.rule_id = ?)"
        )
        args.append(rule)
    if q:
        where.append("(c.measure_raw LIKE ? OR e.canonical_name LIKE ?)")
        args += [f"%{q}%", f"%{q}%"]

    return db.rows(
        conn,
        f"""
        SELECT c.lemma_key,
               MIN(c.measure_raw)          AS measure,
               MAX(e.canonical_name)       AS entity,
               COUNT(*)                    AS readings,
               COUNT(DISTINCT c.witness_id) AS witnesses,
               (SELECT COUNT(*) FROM verdict v WHERE v.lemma_key = c.lemma_key) AS verdicts,
               (SELECT GROUP_CONCAT(DISTINCT v.verdict) FROM verdict v
                 WHERE v.lemma_key = c.lemma_key) AS kinds
        FROM claim c
        LEFT JOIN entity e ON e.id = c.entity_id
        WHERE {' AND '.join(where)}
        GROUP BY c.lemma_key
        HAVING verdicts > 0
        ORDER BY witnesses DESC, verdicts DESC, readings DESC
        LIMIT ?
        """,
        args + [limit],
    )


@app.get("/", response_class=HTMLResponse)
def apparatus(request: Request, verdict: str = "all", rule: str = "", q: str = ""):
    return templates.TemplateResponse(
        "apparatus.html",
        {
            "request": request,
            "lemmas": _lemma_rows(verdict, rule, q),
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
            "highlights": _highlights(),
        },
    )


def _highlights() -> list[dict]:
    """One representative verdict per showcase rule, preferring cross-document."""
    out = []
    for rule_id, title, blurb in HIGHLIGHTS:
        row = db.one(
            conn,
            """SELECT v.id, ca.measure_raw AS measure, ca.witness_id AS wa,
                      cb.witness_id AS wb
               FROM verdict v
               JOIN claim ca ON ca.id = v.claim_a
               JOIN claim cb ON cb.id = v.claim_b
               WHERE v.rule_id = ?
               ORDER BY
                        -- Two documents disagreeing beats one disagreeing with
                        -- itself.
                        (ca.witness_id <> cb.witness_id) DESC,
                        -- A pair whose printed values look nothing alike is the
                        -- demonstration; two documents both writing "10.1" agree
                        -- trivially and show nothing.
                        (COALESCE(ca.value_raw,'') <> COALESCE(cb.value_raw,'')) DESC,
                        (LOWER(COALESCE(ca.unit_raw,'')) <> LOWER(COALESCE(cb.unit_raw,''))) DESC,
                        -- A corroboration convinces when the figures match most
                        -- closely, a contradiction when they diverge most. One
                        -- column, opposite directions.
                        CASE WHEN v.verdict = 'corroborates'
                             THEN  ABS(COALESCE(v.divergence, 0))
                             ELSE -ABS(COALESCE(v.divergence, 0)) END ASC,
                        ABS(COALESCE(ca.value_num, 0)) DESC
               LIMIT 1""",
            (rule_id,),
        )
        if row:
            out.append({
                "rule": rule_id, "title": title, "blurb": blurb,
                "vid": row["id"], "measure": row["measure"],
                "cross": row["wa"] != row["wb"],
            })
    return out


@app.get("/lemma/{lemma_key}", response_class=HTMLResponse)
def lemma_detail(request: Request, lemma_key: str):
    """Every reading of one subject, and every relationship between them."""
    readings = db.rows(
        conn,
        """SELECT c.*, w.title AS witness_title, w.vintage_date,
                  a.page_no, a.match_kind, p.printed_label
           FROM claim c
           JOIN witness w ON w.id = c.witness_id
           LEFT JOIN anchor a ON a.claim_id = c.id
           LEFT JOIN page p ON p.witness_id = c.witness_id AND p.page_no = a.page_no
           WHERE c.lemma_key = ?
           ORDER BY c.period_start, w.vintage_date, c.id""",
        (lemma_key,),
    )
    if not readings:
        raise HTTPException(404, "no such lemma")

    verdicts = db.rows(
        conn,
        """SELECT v.*, ca.value_raw AS a_value, ca.unit_raw AS a_unit,
                  ca.period_raw AS a_period, cb.value_raw AS b_value,
                  cb.unit_raw AS b_unit, cb.period_raw AS b_period,
                  wa.title AS a_witness, wb.title AS b_witness
           FROM verdict v
           JOIN claim ca ON ca.id = v.claim_a
           JOIN claim cb ON cb.id = v.claim_b
           JOIN witness wa ON wa.id = ca.witness_id
           JOIN witness wb ON wb.id = cb.witness_id
           WHERE v.lemma_key = ?
           ORDER BY CASE v.verdict WHEN 'contradicts' THEN 0 WHEN 'corroborates' THEN 1
                                   WHEN 'reconciled' THEN 2 ELSE 3 END,
                    COALESCE(v.severity, 0) DESC""",
        (lemma_key,),
    )

    # Fold readings that say the identical thing. One document phrasing the same
    # figure as "Fiscal 2019" in a table and "for the year ended March 31, 2019"
    # in prose yields two claims, and showing both as separate rows makes the
    # subject look inconsistent when it is simply repeating itself. Both are
    # kept in the database; only the display is collapsed, with a count.
    prepared: list[dict] = []
    seen: dict[tuple, dict] = {}
    for r in readings:
        d = dict(r)
        d["basis_list"] = json.loads(d["basis"]) if d["basis"] else []
        # Basis is deliberately NOT in the key. The same figure is often
        # extracted once with its qualifier and once without, and those are the
        # same reading. Standalone and consolidated must stay apart, so a fold
        # only happens when one side states no basis at all or the two agree.
        key = (d["witness_id"], d["period_start"], d["period_end"],
               d["value_num"], d["unit_dim"])
        prior = seen.get(key)
        mergeable = (
            prior is not None
            and d["value_num"] is not None
            and (not d["basis_list"] or not prior["basis_list"]
                 or d["basis_list"] == prior["basis_list"])
        )
        if mergeable:
            prior["repeats"] += 1
            # Keep whichever reading actually states its basis.
            if d["basis_list"] and not prior["basis_list"]:
                prior["basis_list"] = d["basis_list"]
            continue
        d["repeats"] = 1
        if prior is None:
            seen[key] = d
        prepared.append(d)

    grouped: dict[str, list] = {}
    for v in verdicts:
        grouped.setdefault(v["verdict"], []).append(v)

    return templates.TemplateResponse(
        "lemma.html",
        {
            "request": request,
            "lemma_key": lemma_key,
            "measure": readings[0]["measure_raw"],
            "readings": prepared,
            "collapsed": len(readings) - len(prepared),
            "grouped": grouped,
            "n_verdicts": len(verdicts),
            "n_witnesses": len({r["witness_id"] for r in readings}),
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


@app.get("/evidence-block/{claim_id}", response_class=HTMLResponse)
def evidence_block(request: Request, claim_id: int):
    """One reading's quote plus its rendered page, fetched on demand.

    Page renders are a quarter of a megabyte each and a subject can carry a
    dozen readings, so the subject view lists them and loads the picture only
    when a reader asks for that one.
    """
    return templates.TemplateResponse(
        "evidence_block.html", {"request": request, "claim": _claim(claim_id)}
    )


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


@app.post("/upload", response_class=HTMLResponse)
async def upload(request: Request, file: UploadFile):
    """Accept a PDF and start ingesting it in the background.

    Returns the job card immediately rather than the finished result: reading a
    100-page document is a couple of minutes of model calls, and a request held
    open for that long is a request that times out.
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        return templates.TemplateResponse(
            "job.html",
            {"request": request, "job": None,
             "reject": f"{file.filename or 'that file'} is not a PDF."},
            status_code=400,
        )
    UPLOADS.mkdir(parents=True, exist_ok=True)
    dest = UPLOADS / Path(file.filename).name
    with open(dest, "wb") as fh:
        shutil.copyfileobj(file.file, fh)

    job = jobs.start(dest, db_path=str(db.DB_PATH))
    return templates.TemplateResponse("job.html", {"request": request, "job": job})


@app.get("/jobs/{job_id}", response_class=HTMLResponse)
def job_status(request: Request, job_id: str):
    """Polled by the job card until the work finishes."""
    job = jobs.get(job_id)
    if job is None:
        return HTMLResponse('<div class="job job--gone">That job is no longer tracked.</div>')
    return templates.TemplateResponse("job.html", {"request": request, "job": job})


@app.post("/upload-json")
async def upload_json(file: UploadFile):
    """Same thing for scripts. Returns a job id to poll at /api/jobs/{id}."""
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "expected a .pdf")
    UPLOADS.mkdir(parents=True, exist_ok=True)
    dest = UPLOADS / Path(file.filename).name
    with open(dest, "wb") as fh:
        shutil.copyfileobj(file.file, fh)
    job = jobs.start(dest, db_path=str(db.DB_PATH))
    return JSONResponse({"job": job.id, "poll": f"/api/jobs/{job.id}"}, status_code=202)


@app.get("/api/jobs/{job_id}")
def job_json(job_id: str):
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "no such job")
    return JSONResponse({
        "id": job.id, "filename": job.filename, "stage": job.stage,
        "label": job.label, "percent": job.percent, "detail": job.detail,
        "finished": job.finished, "error": job.error, "hint": job.hint,
        "result": job.result, "elapsed": job.elapsed,
    })
