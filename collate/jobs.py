"""Background ingestion, so the browser is not asked to hold a three-minute request.

Adding a document is slow in a way that cannot be optimised away: it is a
handful of model calls, each taking tens of seconds. A synchronous upload would
either time out or look hung, and a spinner with no stages behind it is
indistinguishable from a crash.

So an upload starts a job here and returns immediately. The page polls for
status and shows which stage is running. Jobs live in memory for the life of
the process, which is the right lifetime: they describe work in flight, and the
result of that work is in the database, which is the thing that persists.

The worker opens its OWN sqlite connection. Sharing the request connection
across threads is what previously collapsed throughput to zero calls a minute -
every worker's write queued behind the main thread's - and it is a much nastier
bug from inside a web server than from inside a CLI.
"""

from __future__ import annotations

import threading
import traceback
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

# The stages a document passes through, in order, with the share of the wait
# each one typically accounts for. Used only to drive the progress bar.
STAGES = [
    ("queued", "queued", 0),
    ("reading", "reading the PDF and finding its tables", 10),
    ("front_matter", "identifying the document", 20),
    ("extracting", "extracting claims (the slow part)", 35),
    ("anchoring", "checking every quote against its page", 75),
    ("framing", "resolving units, periods and measures", 85),
    ("comparing", "comparing against everything already known", 92),
    ("done", "done", 100),
]
_LABEL = {k: (label, pct) for k, label, pct in STAGES}


@dataclass
class Job:
    id: str
    filename: str
    stage: str = "queued"
    detail: str = ""
    result: dict[str, Any] | None = None
    error: str | None = None
    hint: str | None = None
    started: datetime = field(default_factory=datetime.now)

    @property
    def label(self) -> str:
        return _LABEL.get(self.stage, (self.stage, 0))[0]

    @property
    def percent(self) -> int:
        return _LABEL.get(self.stage, (self.stage, 0))[1]

    @property
    def finished(self) -> bool:
        return self.stage == "done" or self.error is not None

    @property
    def elapsed(self) -> str:
        s = int((datetime.now() - self.started).total_seconds())
        return f"{s // 60}m {s % 60}s" if s >= 60 else f"{s}s"


_jobs: dict[str, Job] = {}
_lock = threading.Lock()


def get(job_id: str) -> Job | None:
    with _lock:
        return _jobs.get(job_id)


def recent(limit: int = 6) -> list[Job]:
    with _lock:
        return sorted(_jobs.values(), key=lambda j: j.started, reverse=True)[:limit]


def _friendly(exc: Exception) -> tuple[str, str | None]:
    """Turn an exception into something a reader can act on.

    Quota exhaustion is the expected failure here, not an exotic one, and a raw
    500 makes a working system look broken. It deserves a sentence that says
    what happened and what to do about it.
    """
    msg = str(exc)
    low = msg.lower()
    if "429" in msg or "resource_exhausted" in low or "quota" in low:
        return (
            "The free tier's daily request allowance is spent.",
            "Gemini allows 20 requests per day per model. Collate rotates through "
            "nine of them, so this means all nine are used up for today; the "
            "allowance resets at midnight US Pacific. Everything already in the "
            "apparatus still works - only adding new documents is blocked.",
        )
    if "api key" in low or "GEMINI_API_KEY" in msg:
        return (
            "No usable API key.",
            "Copy .env.example to .env and paste a key from "
            "https://aistudio.google.com/apikey",
        )
    if "no readable pages" in low:
        return ("That PDF has no extractable text.",
                "It is most likely a scan. Collate reads text layers, not images.")
    return (f"{type(exc).__name__}: {msg[:300]}", None)


def start(path: Path, db_path: str | None = None) -> Job:
    job = Job(id=uuid.uuid4().hex[:12], filename=path.name)
    with _lock:
        _jobs[job.id] = job

    def run() -> None:
        from . import collation, db, llm, normalize, pipeline, segment

        conn = db.connect(db_path)  # this thread's own connection, deliberately
        try:
            # Check for a duplicate before building a model client. Negotiating
            # one costs a live request, and spending a request from a 20-a-day
            # allowance to discover we already have the file is a bad trade.
            digest = segment.sha256_file(path)
            seen = db.one(conn, "SELECT id FROM witness WHERE sha256=?", (digest,))
            if seen:
                job.result = {"witness_id": seen["id"], "skipped": True}
                job.stage = "done"
                job.detail = "already in the corpus - identical file contents"
                return

            job.stage = "front_matter"
            client = llm.Gemini(conn=conn)
            job.detail = f"using {client.model}"

            def progress(stage: str, detail: str = "") -> None:
                job.stage = stage
                if detail:
                    job.detail = detail

            result = pipeline.ingest(conn, client, path, progress=progress)

            if result.get("skipped"):
                job.result = result
                job.stage = "done"
                job.detail = "already in the corpus - identical file contents"
                return

            job.stage = "framing"
            normalize.normalize_witness(conn, result["witness_id"])

            job.stage = "comparing"
            result["collation"] = collation.run(conn, client=client, escalate=False)
            client.flush_log()
            conn.commit()

            job.result = result
            job.stage = "done"
            job.detail = ""
        except Exception as exc:  # noqa: BLE001 - reported to the page, not raised
            job.error, job.hint = _friendly(exc)
            job.detail = traceback.format_exc(limit=2)[-400:]
        finally:
            conn.close()

    threading.Thread(target=run, daemon=True, name=f"ingest-{job.id}").start()
    return job
