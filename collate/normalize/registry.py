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

# Calibrated against real pairs from the corpus rather than picked by feel.
# Measured cosines, MiniLM:
#
#   0.905  "Total income (I)"            / "Total income"                merge
#   0.874  "Revenues from express ..."   / "Express Parcel revenue"      merge
#   0.830  "Revenue from contract ..."   / "Revenues from customers"     merge
#   0.808  "Revenue from Operations (Consolidated)" / "(Standalone)"     merge*
#   ---------------------------------------------------------------- 0.80
#   0.738  "EBITDA"                      / "Adjusted EBITDA"             keep apart
#   0.601  "Total income (I)"            / "Total expenses (II)"         keep apart
#   0.541  "Loss for the year"           / "Profit for the year"         keep apart
#   0.469  "Revenue from operations"     / "Total income"                keep apart
#
# * correct only because the basis qualifier is stripped out of the measure
#   first - see split_qualifiers below. Consolidated and standalone revenue are
#   one measure on two bases, not two measures.
#
# At the old 0.86 the registry fragmented into 319 measures over 581 claims and
# only nine lemmas reached across two witnesses, so there was almost nothing to
# adjudicate. The gap between 0.738 and 0.808 is wide enough to sit in.
MERGE = 0.80
NEIGHBOUR = 0.68

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


_NOTE_MARKER = re.compile(r"\s*[\(\[]\s*(?:[ivxlcdm]{1,5}|\d{1,2}|[a-z])\s*[\)\]]\s*$", re.I)
_PARENTHETICAL = re.compile(r"[\(\[]([^)\]]{2,40})[\)\]]")
# Qualifier words that belong in the frame's basis, not in the measure's name.
_BASIS_WORDS = re.compile(
    r"^\s*(consolidated|standalone|separate|audited|unaudited|restated|revised|"
    r"provisional|estimated|net|gross|nominal|real|annualised|annualized|"
    r"seasonally\s+adjusted|per\s+capita|excluding[^)]*|including[^)]*)\s*$",
    re.I,
)


def split_qualifiers(phrase: str) -> tuple[str, list[str]]:
    """Separate a measure's name from qualifiers wedged into it.

    Documents write "Revenue from Operations (Consolidated)" as one label, but
    that is one measure on one basis, not a measure of its own. Leaving the
    qualifier in the name fragments the registry AND hides the difference from
    the basis rules, so BASIS_MISMATCH can never fire on it. Pulling it out
    fixes both at once.

    Also strips bare note markers - the "(I)" in "Total income (I)" is a
    reference into the statement's own numbering, not a qualifier.
    """
    text = " ".join(str(phrase or "").split())
    extracted: list[str] = []

    for inner in _PARENTHETICAL.findall(text):
        if _BASIS_WORDS.match(inner):
            extracted.append(inner.strip().lower())
            text = text.replace(f"({inner})", " ").replace(f"[{inner}]", " ")

    text = " ".join(text.split())
    while True:
        stripped = _NOTE_MARKER.sub("", text)
        if stripped == text:
            break
        text = stripped

    return (text.strip(" -–—:,") or phrase), extracted


_STOPWORDS = {
    "the", "a", "an", "of", "for", "from", "in", "on", "at", "to", "and", "or",
    "by", "with", "as", "is", "are", "s", "total",
}


def _stem(word: str) -> str:
    """Crudest possible stemmer: plurals only. Enough to see that 'revenues from
    customers' and 'revenue from customers' are one phrase, and not so clever
    that it starts equating things it should not."""
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    # Strip a trailing plural 's' only. An "es" rule looks tidier and is worse:
    # it turns "revenues" into "revenu", which then fails to match "revenue"
    # and reports the two as a substitution.
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def content_key(phrase: str) -> frozenset[str]:
    words = re.findall(r"[a-z0-9]+", str(phrase or "").lower())
    return frozenset(_stem(w) for w in words if w not in _STOPWORDS and len(w) > 1)


def substitutes(a: str, b: str) -> bool:
    """Do these two phrases swap one word for another, rather than elaborate?

    "net cash from FINANCING activities" and "net cash from OPERATING
    activities" each carry a content word the other lacks. That is substitution,
    and substitution means two different measures however closely they embed -
    these two score well above any workable threshold, and merging them invented
    a 177% contradiction between a cash-flow line and a different cash-flow line.

    Contrast elaboration, where one phrase's words are a subset of the other's
    ("revenue from customers" inside "revenue from contracts with customers").
    That is usually the same measure named at two levels of detail, and is left
    to the embedding to judge.

    Purely structural, so it carries no domain knowledge: it would separate
    "left ventricular volume" from "right ventricular volume" just as happily.
    """
    ka, kb = content_key(a), content_key(b)
    return bool(ka - kb) and bool(kb - ka)


def slugify(text: str, maxlen: int = 60) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", str(text).lower()).strip("_")
    return (s[:maxlen] or "unnamed")


# -- measures ---------------------------------------------------------------


class MeasureIndex:
    """The registry, held in memory for the length of a run.

    The first version of this re-read every embedding out of SQLite and encoded
    one phrase per call, which cost 1.4 seconds per claim - fine for the 69
    claims of a slide deck, an hour and a half for the full corpus. The vectors
    are a few hundred KB; keeping them in a matrix and appending on mint turns
    the inner loop into a single matrix-vector product.
    """

    def __init__(self, conn):
        self.conn = conn
        rows = conn.execute(
            "SELECT id, embedding FROM measure WHERE embedding IS NOT NULL ORDER BY id"
        ).fetchall()
        self.ids: list[int] = [r["id"] for r in rows]
        self.mat: np.ndarray | None = (
            np.vstack([np.frombuffer(r["embedding"], dtype=np.float32) for r in rows])
            if rows
            else None
        )
        self.labels: dict[int, str] = {
            r["id"]: r["canonical_label"]
            for r in conn.execute("SELECT id, canonical_label FROM measure").fetchall()
        }
        self.aliases: dict[str, int] = {
            r["phrase"]: r["measure_id"]
            for r in conn.execute("SELECT phrase, measure_id FROM measure_alias").fetchall()
        }

    def _append(self, mid: int, vec: np.ndarray) -> None:
        self.ids.append(mid)
        row = vec.reshape(1, -1)
        self.mat = row if self.mat is None else np.vstack([self.mat, row])

    def resolve(
        self, phrase: str, dimension: str | None, witness_id: int | None,
        vec: np.ndarray | None = None,
    ) -> int | None:
        phrase = " ".join(str(phrase or "").split())
        if not phrase:
            return None
        key = phrase.lower()

        if key in self.aliases:
            mid = self.aliases[key]
            self.conn.execute("UPDATE measure SET n_claims = n_claims + 1 WHERE id = ?", (mid,))
            return mid

        if vec is None:
            vec = embed([phrase])[0]

        # Rank by embedding, but let the strongest candidate that is not a
        # substitution win - the nearest neighbour is sometimes precisely the
        # phrase that swaps one word for another.
        best_id, best_score = None, 0.0
        if self.mat is not None and self.ids:
            sims = self.mat @ vec
            for j in np.argsort(-sims)[:8]:
                cand_id = self.ids[int(j)]
                if substitutes(phrase, self.labels.get(cand_id, "")):
                    continue
                best_id, best_score = cand_id, float(sims[int(j)])
                break

        # Identical content words are a merge on their own account: "Revenues
        # from customers" and "Revenue from customers" need no model to tell
        # them apart, and requiring the threshold as well would split them.
        exact_words = (
            best_id is not None
            and content_key(phrase) == content_key(self.labels.get(best_id, ""))
        )

        if best_id is not None and (best_score >= MERGE or exact_words):
            self.conn.execute(
                "INSERT OR IGNORE INTO measure_alias (measure_id, phrase, similarity)"
                " VALUES (?,?,?)",
                (best_id, key, best_score),
            )
            self.conn.execute(
                "UPDATE measure SET n_claims = n_claims + 1 WHERE id = ?", (best_id,)
            )
            self.aliases[key] = best_id
            return best_id

        slug = slugify(phrase)
        n = 1
        while self.conn.execute("SELECT 1 FROM measure WHERE slug = ?", (slug,)).fetchone():
            n += 1
            slug = f"{slugify(phrase)}_{n}"
        cur = self.conn.execute(
            "INSERT INTO measure (slug, canonical_label, dimension, embedding, n_claims,"
            " first_seen_in) VALUES (?,?,?,?,1,?)",
            (slug, phrase, dimension, vec.tobytes(), witness_id),
        )
        mid = int(cur.lastrowid)
        self.conn.execute(
            "INSERT OR IGNORE INTO measure_alias (measure_id, phrase, similarity) VALUES (?,?,1.0)",
            (mid, key),
        )
        self.aliases[key] = mid
        self.labels[mid] = phrase
        self._append(mid, vec)
        return mid


def resolve_measure(conn, phrase: str, dimension: str | None, witness_id: int | None) -> int | None:
    """Single-shot resolution. Convenient for one-off calls; batch through
    MeasureIndex when framing a whole witness."""
    return MeasureIndex(conn).resolve(phrase, dimension, witness_id)


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
    index = MeasureIndex(conn)
    if index.mat is None or measure_id not in index.ids:
        return []
    vec = index.mat[index.ids.index(measure_id)]
    sims = index.mat @ vec
    out = [
        (index.ids[i], float(s))
        for i, s in enumerate(sims)
        if index.ids[i] != measure_id and floor <= s < MERGE
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
