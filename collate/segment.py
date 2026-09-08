"""Turning a PDF into anchorable text.

Two things here are load-bearing and neither is obvious:

1.  We build the canonical page text ourselves, block by block, recording the
    character range each block occupies. Every later anchor is a character
    offset into that string, so it can always be walked back to a rectangle on
    a rendered page. Using fitz's own get_text() would have been one line, but
    then a claim's offsets would refer to a string we could not map back to
    geometry.

2.  Tables are lifted out of the prose flow, serialised as pipe tables, and
    stamped with whatever scoping header sits above them - "(Rs in millions)",
    "Consolidated", and friends. That header is the difference between 82,568
    meaning eighty-two thousand and eighty-two billion, and it routinely lives
    three rows above the number it governs. Chunking prose blindly severs the
    two; this is the whole reason the table lane exists.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

import fitz

# Scale and basis markers worth dragging downward into a table's context.
# Deliberately patterns, not a vocabulary list: a new document can introduce
# "Rs in thousands" or "Unaudited" without anyone editing this file.
SCALE_RE = re.compile(
    r"""\(?\s*
    (?:in\s+)?
    (?:(?P<cur>Rs\.?|INR|₹|US\s?\$|USD|\$|€|EUR)\s*)?
    (?:in\s+)?
    (?P<scale>million|billion|trillion|crore|lakh|lakhs|thousand|mn|bn|tn|cr|hundreds)
    s?\b""",
    re.IGNORECASE | re.VERBOSE,
)
BASIS_RE = re.compile(
    r"\b(consolidated|standalone|unaudited|audited|restated|provisional|revised|"
    r"proforma|pro\s?forma|estimated|projected|annualis?zed|seasonally\s+adjusted)\b",
    re.IGNORECASE,
)
# A page's printed number, which in these curated excerpts rarely matches its
# physical index. Checked only at the top and bottom margins.
PAGE_LABEL_RE = re.compile(r"^\s*(?:page\s+)?([0-9]{1,4}|[ivxlcdm]{1,7})\s*$", re.IGNORECASE)

# Passage sizing, arrived at by measurement rather than by argument.
#
# The published free tier is 1,500 requests/day. The API actually enforces
# GenerateRequestsPerDayPerProjectPerModel-FreeTier = 20/day/model, two orders
# of magnitude lower, which made a strong case for using the million-token
# context and batching ~25 pages per request.
#
# That was tried and reversed. At 25 pages the model stops enumerating and
# starts summarising - the earnings deck fell from 84 claims to 2, and the whole
# corpus produced 8 verdicts with no contradictions in them. Recall collapses
# long before the context window does, and a failed 25-page passage loses 25
# pages where a failed 4-page one loses four.
#
# So the quota is absorbed elsewhere: by rotating models (nine of them, 20/day
# each) and by a disk cache that deliberately does not key on the model.
MAX_PASSAGE_CHARS = 24_000
MAX_PASSAGE_PAGES = 4
SPARSE_DENSITY = 0.0035  # chars per pt^2; tuned against the slide deck


def _columns(blocks: list, page_width: float) -> int:
    """How many text columns is this page laid out in? 1 or 2.

    Reading order matters more than it looks. Sorting blocks by (y, x) is right
    for a single column and catastrophic for two: it interleaves the left and
    right columns line by line, so every sentence in the document arrives cut in
    half and spliced to an unrelated one. The model then does the reasonable
    thing and reconstructs fluent prose out of the fragments - which is to say
    it produces sentences that are not in the document, and the anchoring gate
    rejects them. On the RBI annual report that was 587 claims, a 39% anchor
    rate, and the cause was invisible from the claim side.

    Detection is deliberately conservative: a page counts as two-column only
    when blocks sit cleanly either side of the centre and few straddle it.
    Tables and full-width headings straddle, which is why they are counted.
    """
    mid = page_width / 2
    left = right = straddle = 0
    for b in blocks:
        x0, x1 = b.bbox[0], b.bbox[2]
        if x1 < mid + 8:
            left += 1
        elif x0 > mid - 8:
            right += 1
        else:
            straddle += 1
    sided = left + right
    if sided < 6 or right < 2 or left < 2:
        return 1
    return 2 if straddle <= sided * 0.35 else 1


def _reading_order(blocks: list, page_width: float) -> list:
    columns = _columns(blocks, page_width)
    if columns == 1:
        return sorted(blocks, key=lambda b: (round(b.bbox[1], 1), b.bbox[0]))
    mid = page_width / 2
    def key(b):
        # Full-width blocks keep their vertical position and sort ahead of the
        # column they open; column blocks read top-to-bottom, left before right.
        straddles = b.bbox[0] <= mid - 8 and b.bbox[2] >= mid + 8
        column = 0 if straddles else (0 if b.bbox[2] < mid + 8 else 1)
        return (column, round(b.bbox[1], 1), b.bbox[0])
    return sorted(blocks, key=key)


@dataclass
class Block:
    page_no: int
    idx: int
    kind: str  # prose | table
    text: str
    bbox: tuple[float, float, float, float]
    char_start: int = 0
    char_end: int = 0


@dataclass
class Page:
    page_no: int
    printed_label: str | None
    text: str
    blocks: list[Block]
    width: float
    height: float
    context_header: str | None = None

    @property
    def density(self) -> float:
        area = max(self.width * self.height, 1.0)
        return len(self.text) / area

    @property
    def is_sparse(self) -> bool:
        return self.density < SPARSE_DENSITY


@dataclass
class Passage:
    page_start: int
    page_end: int
    text: str
    context_header: str | None
    pages: list[int] = field(default_factory=list)

    @property
    def hash(self) -> str:
        payload = f"{self.context_header or ''}\x00{self.text}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _serialise_table(rows: list[list[str | None]]) -> str:
    """Pipe-table serialisation. Models read these far more reliably than they
    read whitespace-aligned columns, and the pipes survive tokenisation."""
    if not rows:
        return ""
    width = max(len(r) for r in rows)
    out = []
    for i, row in enumerate(rows):
        cells = [(c or "").replace("\n", " ").strip() for c in row]
        cells += [""] * (width - len(cells))
        out.append("| " + " | ".join(cells) + " |")
        if i == 0:
            out.append("|" + "---|" * width)
    return "\n".join(out)


def _header_above(prose: list[Block], table_bbox, page_height: float) -> str | None:
    """Find the scoping header for a table: the nearest text above it carrying a
    scale or basis marker. Searches upward within a third of the page, because
    a caption further away than that usually belongs to something else."""
    top = table_bbox[1]
    reach = page_height / 3
    best: tuple[float, str] | None = None
    for b in prose:
        if b.bbox[3] > top or top - b.bbox[3] > reach:
            continue
        if SCALE_RE.search(b.text) or BASIS_RE.search(b.text):
            gap = top - b.bbox[3]
            if best is None or gap < best[0]:
                best = (gap, " ".join(b.text.split())[:200])
    return best[1] if best else None


def _printed_label(raw_lines: list[str]) -> str | None:
    for line in list(raw_lines[:2]) + list(raw_lines[-2:]):
        m = PAGE_LABEL_RE.match(line.strip())
        if m:
            return m.group(1)
    return None


def read_pages(pdf_path: Path) -> list[Page]:
    """Parse a PDF into pages whose text we can anchor into."""
    doc = fitz.open(str(pdf_path))
    pages: list[Page] = []
    try:
        for page_no, page in enumerate(doc):
            rect = page.rect

            # Tables first, so their regions can be carved out of the prose.
            tables = []
            try:
                for t in page.find_tables().tables:
                    body = _serialise_table(t.extract())
                    if body.strip():
                        tables.append((tuple(t.bbox), body))
            except Exception:
                # find_tables is heuristic and does occasionally throw on
                # malformed page graphics. A page without table detection is
                # still a usable page.
                pass

            raw = page.get_text("blocks")
            prose: list[Block] = []
            for x0, y0, x1, y1, text, *_ in sorted(raw, key=lambda b: (round(b[1], 1), b[0])):
                text = text.strip()
                if not text:
                    continue
                inside = any(
                    x0 >= tb[0] - 2 and y0 >= tb[1] - 2 and x1 <= tb[2] + 2 and y1 <= tb[3] + 2
                    for tb, _ in tables
                )
                if inside:
                    continue  # the table lane owns this text
                prose.append(Block(page_no, 0, "prose", text, (x0, y0, x1, y1)))

            table_blocks = [
                Block(page_no, 0, "table", body, bbox) for bbox, body in tables
            ]

            ordered = _reading_order(prose + table_blocks, rect.width)

            parts: list[str] = []
            cursor = 0
            for idx, b in enumerate(ordered):
                b.idx = idx
                b.char_start = cursor
                b.char_end = cursor + len(b.text)
                parts.append(b.text)
                cursor = b.char_end + 2  # the "\n\n" join below

            page_text = "\n\n".join(parts)

            header = None
            for bbox, _ in tables:
                header = _header_above(prose, bbox, rect.height) or header
            if header is None:
                for b in prose[:4]:
                    if SCALE_RE.search(b.text):
                        header = " ".join(b.text.split())[:200]
                        break

            pages.append(
                Page(
                    page_no=page_no,
                    printed_label=_printed_label(page_text.splitlines()),
                    text=page_text,
                    blocks=ordered,
                    width=rect.width,
                    height=rect.height,
                    context_header=header,
                )
            )
    finally:
        doc.close()
    return pages


def build_passages(pages: list[Page]) -> list[Passage]:
    """Batch consecutive pages into model-sized passages.

    Page-batched rather than sliding-window chunked, for two reasons that
    happen to agree: a table header and its rows stay together, and the free
    tier meters requests-per-minute far more tightly than tokens-per-minute,
    so fewer-and-fatter beats many-and-thin.
    """
    passages: list[Passage] = []
    bucket: list[Page] = []
    size = 0
    carried: str | None = None

    def flush() -> None:
        nonlocal bucket, size
        if not bucket:
            return
        body = "\n\n".join(
            f"[page {p.page_no + 1}"
            + (f" | printed {p.printed_label}" if p.printed_label else "")
            + f"]\n{p.text}"
            for p in bucket
        )
        header = next((p.context_header for p in bucket if p.context_header), carried)
        passages.append(
            Passage(
                page_start=bucket[0].page_no,
                page_end=bucket[-1].page_no,
                text=body,
                context_header=header,
                pages=[p.page_no for p in bucket],
            )
        )
        bucket, size = [], 0

    for p in pages:
        if p.context_header:
            carried = p.context_header  # scale markers persist across a section
        if bucket and (size + len(p.text) > MAX_PASSAGE_CHARS or len(bucket) >= MAX_PASSAGE_PAGES):
            flush()
        bucket.append(p)
        size += len(p.text)
    flush()
    return [p for p in passages if p.text.strip()]
