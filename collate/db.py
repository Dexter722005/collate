"""Storage for Collate.

Plain sqlite3, no ORM. The schema is small enough to hold in your head, and the
queries the apparatus view needs are joins an ORM would only get in the way of.

Vocabulary (DECISIONS.md #1):
    witness   a source document; it testifies, it does not define truth
    claim     one grounded assertion a witness makes
    anchor    the machine-verified location of a claim inside its witness
    lemma     (entity, measure) - the subject two claims are arguing about
    frame     lemma + period + basis + unit dimension; the exact coordinates
    reading   what one witness says at one frame
    verdict   the adjudicated relationship between two readings
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Iterable

DB_PATH = Path(__file__).resolve().parent.parent / "collate.db"

SCHEMA = """
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

-- A source document. sha256 is the idempotency key: re-ingesting the same
-- bytes is a no-op, which is what makes growing the corpus incremental.
CREATE TABLE IF NOT EXISTS witness (
    id               INTEGER PRIMARY KEY,
    sha256           TEXT UNIQUE NOT NULL,
    filename         TEXT NOT NULL,
    title            TEXT,
    publisher        TEXT,
    doc_type         TEXT,
    -- vintage is when the document was PUBLISHED, deliberately distinct from
    -- the period a claim covers. An FY24 number published in 2022 is a
    -- forecast; the same number published in 2025 is history. The
    -- VINTAGE_REVISION rule is unimplementable without this column.
    vintage_date     TEXT,
    default_currency TEXT,
    default_scale    TEXT,
    page_count       INTEGER,
    status           TEXT NOT NULL DEFAULT 'new',
    front_matter     TEXT,
    ingested_at      TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Page text is stored verbatim so anchors stay resolvable long after the
-- passage that produced a claim has been forgotten.
CREATE TABLE IF NOT EXISTS page (
    id            INTEGER PRIMARY KEY,
    witness_id    INTEGER NOT NULL REFERENCES witness(id) ON DELETE CASCADE,
    page_no       INTEGER NOT NULL,      -- 0-based physical index
    printed_label TEXT,                  -- the number printed ON the page; the
                                         -- curated excerpts skip pages, so
                                         -- physical 5 may print "26"
    text          TEXT NOT NULL,
    char_count    INTEGER NOT NULL,
    -- Text characters per unit of page area. Slide decks whose numbers are
    -- locked inside chart graphics score near zero here. That is failure
    -- case 4, measured rather than guessed at.
    text_density  REAL,
    is_sparse     INTEGER NOT NULL DEFAULT 0,
    UNIQUE (witness_id, page_no)
);

CREATE TABLE IF NOT EXISTS block (
    id         INTEGER PRIMARY KEY,
    witness_id INTEGER NOT NULL REFERENCES witness(id) ON DELETE CASCADE,
    page_no    INTEGER NOT NULL,
    idx        INTEGER NOT NULL,
    kind       TEXT NOT NULL,            -- prose | table | caption
    text       TEXT NOT NULL,
    char_start INTEGER,                  -- offsets into page.text
    char_end   INTEGER,
    x0 REAL, y0 REAL, x1 REAL, y1 REAL
);
CREATE INDEX IF NOT EXISTS block_page ON block(witness_id, page_no);

-- A passage is what we actually send to the model: several pages of prose and
-- serialised tables, plus any scoping header inherited from above. Batching
-- whole pages rather than sliding chunks is what keeps a table header and its
-- rows inside the same request.
CREATE TABLE IF NOT EXISTS passage (
    id             INTEGER PRIMARY KEY,
    witness_id     INTEGER NOT NULL REFERENCES witness(id) ON DELETE CASCADE,
    page_start     INTEGER NOT NULL,
    page_end       INTEGER NOT NULL,
    text           TEXT NOT NULL,
    context_header TEXT,
    hash           TEXT NOT NULL,
    char_count     INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS passage_witness ON passage(witness_id);

-- The dynamic schema. Nothing here is enumerated in advance: a measure is
-- minted the first time a document says something new, and merged into an
-- existing measure when the embedding says they are the same idea.
CREATE TABLE IF NOT EXISTS measure (
    id              INTEGER PRIMARY KEY,
    slug            TEXT UNIQUE NOT NULL,
    canonical_label TEXT NOT NULL,
    dimension       TEXT,                -- currency | percent | count | ratio | text
    embedding       BLOB,
    n_claims        INTEGER NOT NULL DEFAULT 0,
    -- SET NULL, not the default NO ACTION: a measure outlives the document
    -- that introduced it, and re-ingesting that document must not be blocked
    -- by a provenance column.
    first_seen_in   INTEGER REFERENCES witness(id) ON DELETE SET NULL,
    created_at      TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS measure_alias (
    id         INTEGER PRIMARY KEY,
    measure_id INTEGER NOT NULL REFERENCES measure(id) ON DELETE CASCADE,
    phrase     TEXT NOT NULL,
    similarity REAL,
    UNIQUE (measure_id, phrase)
);

CREATE TABLE IF NOT EXISTS entity (
    id             INTEGER PRIMARY KEY,
    slug           TEXT UNIQUE NOT NULL,
    canonical_name TEXT NOT NULL,
    kind           TEXT,                 -- company | country | person | place
    n_claims       INTEGER NOT NULL DEFAULT 0,
    created_at     TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS entity_alias (
    id        INTEGER PRIMARY KEY,
    entity_id INTEGER NOT NULL REFERENCES entity(id) ON DELETE CASCADE,
    phrase    TEXT NOT NULL,
    UNIQUE (entity_id, phrase)
);

CREATE TABLE IF NOT EXISTS claim (
    id          INTEGER PRIMARY KEY,
    witness_id  INTEGER NOT NULL REFERENCES witness(id) ON DELETE CASCADE,
    passage_id  INTEGER REFERENCES passage(id) ON DELETE SET NULL,
    claim_type  TEXT NOT NULL,           -- quantity | state | event | identity
    quote       TEXT NOT NULL,

    -- As the document words it. Kept for display, and for auditing the model
    -- against the page when a normalisation looks wrong.
    entity_raw  TEXT,
    measure_raw TEXT,
    value_raw   TEXT,
    unit_raw    TEXT,
    period_raw  TEXT,
    basis_raw   TEXT,
    qualifiers  TEXT,                    -- json list

    -- The frame, resolved.
    entity_id    INTEGER REFERENCES entity(id),
    measure_id   INTEGER REFERENCES measure(id),
    value_num    REAL,
    value_text   TEXT,                   -- for state / identity claims
    unit_dim     TEXT,                   -- INR | USD | percent | count | ...
    unit_scale   REAL,                   -- multiplier already applied to value_num
    period_start TEXT,
    period_end   TEXT,                   -- half-open [start, end)
    period_grain TEXT,                   -- year | quarter | month | instant
    basis        TEXT,                   -- json sorted list e.g. ["audited","consolidated"]
    lemma_key    TEXT,                   -- entity:measure, the blocking key
    frame_key    TEXT,                   -- lemma + period + basis + unit_dim

    confidence   REAL,
    created_at   TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS claim_lemma   ON claim(lemma_key);
CREATE INDEX IF NOT EXISTS claim_frame   ON claim(frame_key);
CREATE INDEX IF NOT EXISTS claim_witness ON claim(witness_id);

-- One row per claim that survived the anchoring gate. No anchor, no claim.
CREATE TABLE IF NOT EXISTS anchor (
    id         INTEGER PRIMARY KEY,
    claim_id   INTEGER NOT NULL REFERENCES claim(id) ON DELETE CASCADE,
    witness_id INTEGER NOT NULL REFERENCES witness(id) ON DELETE CASCADE,
    page_no    INTEGER NOT NULL,
    char_start INTEGER NOT NULL,
    char_end   INTEGER NOT NULL,
    x0 REAL, y0 REAL, x1 REAL, y1 REAL,
    match_kind TEXT NOT NULL             -- exact | whitespace | fuzzy
);
CREATE INDEX IF NOT EXISTS anchor_claim ON anchor(claim_id);

-- Claims the model produced that could not be tied back to real page text.
-- Kept rather than discarded: the size of this table is a headline number in
-- the README, and pretending it were empty would be a lie.
CREATE TABLE IF NOT EXISTS quarantine (
    id         INTEGER PRIMARY KEY,
    witness_id INTEGER NOT NULL REFERENCES witness(id) ON DELETE CASCADE,
    passage_id INTEGER,
    reason     TEXT NOT NULL,
    payload    TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS verdict (
    id          INTEGER PRIMARY KEY,
    claim_a     INTEGER NOT NULL REFERENCES claim(id) ON DELETE CASCADE,
    claim_b     INTEGER NOT NULL REFERENCES claim(id) ON DELETE CASCADE,
    lemma_key   TEXT NOT NULL,
    verdict     TEXT NOT NULL,           -- corroborates | contradicts | reconciled | distinct
    rule_id     TEXT NOT NULL,
    explanation TEXT NOT NULL,
    divergence  REAL,                    -- relative gap between the two values
    severity    REAL,
    llm_note    TEXT,                    -- escalation only; advisory, never overrides
    llm_agrees  INTEGER,
    created_at  TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE (claim_a, claim_b)
);
CREATE INDEX IF NOT EXISTS verdict_lemma ON verdict(lemma_key);
CREATE INDEX IF NOT EXISTS verdict_kind  ON verdict(verdict);

-- Observability. Cheap to write, and it turns "is the free tier enough?" from
-- an argument into a query.
CREATE TABLE IF NOT EXISTS llm_call (
    id         INTEGER PRIMARY KEY,
    purpose    TEXT NOT NULL,
    model      TEXT,
    cache_hit  INTEGER NOT NULL DEFAULT 0,
    prompt_key TEXT,
    in_tokens  INTEGER,
    out_tokens INTEGER,
    latency_ms INTEGER,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS run_log (
    id         INTEGER PRIMARY KEY,
    witness_id INTEGER REFERENCES witness(id) ON DELETE CASCADE,
    stage      TEXT NOT NULL,
    stats      TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
"""


def connect(path: Path | str | None = None) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path or DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn


def one(conn: sqlite3.Connection, sql: str, args: Iterable[Any] = ()) -> sqlite3.Row | None:
    return conn.execute(sql, tuple(args)).fetchone()


def rows(conn: sqlite3.Connection, sql: str, args: Iterable[Any] = ()) -> list[sqlite3.Row]:
    return conn.execute(sql, tuple(args)).fetchall()


def insert(conn: sqlite3.Connection, table: str, **fields: Any) -> int:
    """Insert a row, JSON-encoding any list or dict values on the way in."""
    clean = {
        k: (json.dumps(v, sort_keys=True) if isinstance(v, (list, dict)) else v)
        for k, v in fields.items()
    }
    cols = ", ".join(clean)
    marks = ", ".join("?" for _ in clean)
    cur = conn.execute(f"INSERT INTO {table} ({cols}) VALUES ({marks})", tuple(clean.values()))
    return int(cur.lastrowid)


def loads(value: str | None, default: Any = None) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except (ValueError, TypeError):
        return default
