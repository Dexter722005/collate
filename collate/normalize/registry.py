"""The dynamic schema: measures and entities, discovered rather than declared.

Collate ships with an empty vocabulary. The first document to say "Revenue from
operations" mints that measure; the second document saying "revenue from
operations (net)" merges into it; a document saying "Total income" does not,
and the gap between them is later reported as a definition difference rather
than a contradiction. Nothing here is seeded, so pointing the system at a
domain nobody anticipated costs no code.

Two thresholds do all the work:

    MERGE      above this, two phrases are the same measure
    NEIGHBOUR  between the two, they are related but distinct - close enough to
               be worth comparing and reporting as a definitional difference,
               which is exactly the DEFINITION_DRIFT case

Embeddings are local (MiniLM on CPU). That is a deliberate trade: a hosted
embedding model would be marginally better, but it would put the dynamic schema
behind a rate limit and break the system whenever the network does.
"""

from __future__ import annotations

import re
import threading

import numpy as np
from rapidfuzz import fuzz

MERGE = 0.86
NEIGHBOUR = 0.72

_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_model = None
_model_lock = threading.Lock()

# Pronouns a document uses for itself. Resolved against the witness's primary
# entity, which came from the front-matter pass rather than from a constant.
SELF_REFERENCE = {
    "the company", "the group", "the bank", "the issuer", "we", "our company",
    "the corporation", "the firm", "company", "group", "it", "the organisation",
    "the organization",
}

_LEGAL_SUFFIX = re.compile(
    r"\b(limited|ltd\.?|private|pvt\.?|inc\.?|incorporated|plc|llp|llc|corp\.?|"
    r"corporation|company|co\.?|s\.a\.|n\.v\.|gmbh)\b",
    re.I,
)


def embedder():
    global _model
    with _model_lock:
        if _model is None:
            from sentence_transformers import SentenceTransformer

            _model = SentenceTransformer(_MODEL_NAME)
        return _model


def embed(texts: list[str]) -> np.ndarray:
    vecs = embedder().encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(vecs, dtype=np.float32)


def slugify(text: str, maxlen: int = 60) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", str(text).lower()).strip("_")
    return (s[:maxlen] or "unnamed")


# -- measures ---------------------------------------------------------------


def _load_measures(conn) -> tuple[list[int], np.ndarray | None, dict[str, int]]:
    rows = conn.execute(
        "SELECT id, slug, canonical_label, embedding FROM measure WHERE embedding IS NOT NULL"
    ).fetchall()
    ids = [r["id"] for r in rows]
    mat = (
        np.vstack([np.frombuffer(r["embedding"], dtype=np.float32) for r in rows])
        if rows
        else None
    )
    aliases = {
        r["phrase"]: r["measure_id"]
        for r in conn.execute("SELECT phrase, measure_id FROM measure_alias").fetchall()
    }
    return ids, mat, aliases


def resolve_measure(conn, phrase: str, dimension: str | None, witness_id: int | None) -> int | None:
    """Map a measure phrase onto the registry, growing it when nothing fits."""
    phrase = " ".join(str(phrase or "").split())
    if not phrase:
        return None
    key = phrase.lower()

    ids, mat, aliases = _load_measures(conn)
    if key in aliases:
        conn.execute("UPDATE measure SET n_claims = n_claims + 1 WHERE id = ?", (aliases[key],))
        return aliases[key]

    vec = embed([phrase])[0]
    best_id, best_score = None, 0.0
    if mat is not None and len(ids):
        sims = mat @ vec
        j = int(np.argmax(sims))
        best_id, best_score = ids[j], float(sims[j])

    if best_id is not None and best_score >= MERGE:
        conn.execute(
            "INSERT OR IGNORE INTO measure_alias (measure_id, phrase, similarity) VALUES (?,?,?)",
            (best_id, key, best_score),
        )
        conn.execute("UPDATE measure SET n_claims = n_claims + 1 WHERE id = ?", (best_id,))
        return best_id

    slug = slugify(phrase)
    n = 1
    while conn.execute("SELECT 1 FROM measure WHERE slug = ?", (slug,)).fetchone():
        n += 1
        slug = f"{slugify(phrase)}_{n}"
    cur = conn.execute(
        "INSERT INTO measure (slug, canonical_label, dimension, embedding, n_claims, first_seen_in)"
        " VALUES (?,?,?,?,1,?)",
        (slug, phrase, dimension, vec.tobytes(), witness_id),
    )
    mid = int(cur.lastrowid)
    conn.execute(
        "INSERT OR IGNORE INTO measure_alias (measure_id, phrase, similarity) VALUES (?,?,1.0)",
        (mid, key),
    )
    return mid


def measure_similarity(conn, a_id: int, b_id: int) -> float:
    rows = conn.execute(
        "SELECT id, embedding FROM measure WHERE id IN (?,?)", (a_id, b_id)
    ).fetchall()
    vecs = {r["id"]: np.frombuffer(r["embedding"], dtype=np.float32) for r in rows if r["embedding"]}
    if a_id not in vecs or b_id not in vecs:
        return 0.0
    return float(vecs[a_id] @ vecs[b_id])


def neighbours(conn, measure_id: int, floor: float = NEIGHBOUR) -> list[tuple[int, float]]:
    """Measures close enough to be worth comparing but not close enough to merge.
    This is what makes 'revenue from operations' and 'total income' meet at all."""
    ids, mat, _ = _load_measures(conn)
    if mat is None or measure_id not in ids:
        return []
    vec = mat[ids.index(measure_id)]
    sims = mat @ vec
    out = [
        (ids[i], float(s))
        for i, s in enumerate(sims)
        if ids[i] != measure_id and floor <= s < MERGE
    ]
    return sorted(out, key=lambda t: -t[1])


# -- entities ---------------------------------------------------------------


def _canonical_form(name: str) -> str:
    s = _LEGAL_SUFFIX.sub("", str(name or "").lower())
    return " ".join(re.sub(r"[^a-z0-9 ]+", " ", s).split())


def resolve_entity(conn, phrase: str, primary_hint: str | None, witness_id: int | None) -> int | None:
    """Entity resolution by surface form. Deliberately fuzzy-string rather than
    embedding-based: 'Delhivery Limited' and 'Delhivery Ltd' are a spelling
    question, and embeddings are worse than edit distance at spelling."""
    phrase = " ".join(str(phrase or "").split())
    if not phrase:
        phrase = primary_hint or ""
    if not phrase:
        return None

    if phrase.lower().strip(" .,") in SELF_REFERENCE and primary_hint:
        phrase = primary_hint

    key = phrase.lower()
    hit = conn.execute("SELECT entity_id FROM entity_alias WHERE phrase = ?", (key,)).fetchone()
    if hit:
        conn.execute("UPDATE entity SET n_claims = n_claims + 1 WHERE id = ?", (hit["entity_id"],))
        return hit["entity_id"]

    canon = _canonical_form(phrase)
    best_id, best_score = None, 0.0
    for row in conn.execute("SELECT id, canonical_name FROM entity").fetchall():
        score = fuzz.token_sort_ratio(canon, _canonical_form(row["canonical_name"])) / 100.0
        if score > best_score:
            best_id, best_score = row["id"], score

    if best_id is not None and best_score >= 0.90:
        conn.execute(
            "INSERT OR IGNORE INTO entity_alias (entity_id, phrase) VALUES (?,?)", (best_id, key)
        )
        conn.execute("UPDATE entity SET n_claims = n_claims + 1 WHERE id = ?", (best_id,))
        return best_id

    slug = slugify(canon or phrase)
    n = 1
    while conn.execute("SELECT 1 FROM entity WHERE slug = ?", (slug,)).fetchone():
        n += 1
        slug = f"{slugify(canon or phrase)}_{n}"
    cur = conn.execute(
        "INSERT INTO entity (slug, canonical_name, n_claims) VALUES (?,?,1)", (slug, phrase)
    )
    eid = int(cur.lastrowid)
    conn.execute("INSERT OR IGNORE INTO entity_alias (entity_id, phrase) VALUES (?,?)", (eid, key))
    return eid
