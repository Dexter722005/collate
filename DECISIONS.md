# Decisions

A working log. Kept in order, including the things that turned out to be wrong,
because the wrong turns are usually where the real constraint showed itself.

---

### 1 — A fact is not a triple · 8 Sep

The obvious model is `(subject, predicate, value)`: Delhivery, revenue, ₹8,142 Cr.
I got about ten minutes into that before noticing it cannot express the case the
brief cares most about. If a fact is just a triple, then FY22 revenue and FY24
revenue are two triples with the same subject and predicate and different values
— structurally identical to a contradiction. There is nothing left to explain
*with*.

So the qualifiers become part of the identity of the claim rather than metadata
hanging off it:

```
frame = (entity, measure, period, basis, unit dimension)
```

Two claims in the same frame are asserting the same thing and must agree. Two
claims in different frames are not in conflict at all, and **the coordinate that
differs is the explanation**. `PERIOD_DISJOINT`, `BASIS_MISMATCH`,
`VINTAGE_REVISION` are not special cases bolted on afterwards — they are just
"which axis did they differ on".

That one reframing is the whole system. Everything below is consequence.

### 2 — Borrowing vocabulary from textual criticism · 8 Sep

Comparing variant readings across imperfect copies of a text is a discipline
that has existed for centuries, and it already has precise words for all of
this. I took them, because they carry the right assumptions:

| term | here |
| --- | --- |
| **witness** | a source document — it *testifies*, it does not define truth |
| **lemma** | `(entity, measure)`; the thing two claims are arguing about |
| **reading** | what one witness says at one frame |
| **apparatus** | the record of variant readings and why they vary |

Calling a document a witness rather than a source changes how you write the
code. A witness can be wrong, can be out of date, can be arguing. `truth` never
became a column, and that was not discipline on my part — the vocabulary made it
awkward to want one.

### 3 — Rules decide, the model explains · 8 Sep

The default design hands two claims to an LLM and asks "do these conflict?". I
did not build that, for three reasons: the answer is unstable between runs, it
cites no mechanism, and it is wrong in ways that look exactly like being right.

Instead there is an ordered cascade of deterministic rules
([collate/adjudicate.py](collate/adjudicate.py)). Each inspects one coordinate;
the first to fire claims the pair and names itself in the output. The model is
called **only** for pairs the rules have already concluded are contradictions,
and its answer is written to `llm_note` beside the verdict rather than into it.

When the two disagree, the UI shows both and says so. That is more informative
than either silently winning, and it is honest about which part of the system is
deterministic.

Trade-off I am accepting: the cascade cannot spot a reconciliation nobody
anticipated. The escalation pass exists to catch some of those, and
`reason_code` is deliberately free-text so the model can name a mechanism I did
not think of.

### 4 — The free tier shaped the architecture, and improved it · 8 Sep

Gemini's free tier meters **15 requests/minute against 1,000,000 tokens/minute**.
Requests are the scarce resource by a factor of thousands.

The natural chunking strategy — sliding windows of ~1k tokens — would have made
~2,000 thin requests for this corpus and taken over two hours. So passages batch
**whole pages**, 3–4 at a time. ~200 fat requests, about fifteen minutes.

The part I did not expect: this is *better extraction*, not merely cheaper.
A table's scale header on one page and its rows on the next now land in the same
request. The constraint and the quality argument pointed the same way, which is
rare enough to be worth writing down.

### 5 — Negotiate the model, do not name it · 8 Sep

I started with `model="gemini-2.5-flash"` hard-coded. Then I listed what the key
could actually reach and found `gemini-3.6-flash`, several generations past what
I assumed. Hard-coding would have quietly used a worse model forever.

`Gemini._negotiate()` now walks a preference list and **probes each with a
one-token call**, because a model appearing in `models.list()` does not mean the
key may call it — free tiers gate models. First one that answers wins.

### 6 — Anchoring is mechanical, on purpose · 8 Sep

No model is asked whether a model told the truth. Every claim must be found
again in the page it came from: exact, then whitespace-flattened, then fuzzy
above 88. Below that it goes to `quarantine` **with its payload kept**.

First run on the earnings deck: 84 proposed, 69 anchored, **82%**. That number
is on a page in the UI. A system that reports only its successes is not
reporting.

`value_present()` was added after watching the subtler failure: a genuine quote
with a number attached that was not in it — a real citation wrapped around a
figure from elsewhere on the page. An anchor proves the quote exists; that check
proves it is the *right* quote.

### 7 — Wrong turn: the registry re-read every vector, every claim · 8 Sep

`resolve_measure()` loaded all measure embeddings out of SQLite and encoded one
phrase per call. Framing 69 claims took **101.9 seconds**. Fine for a slide
deck; roughly **an hour and a half** for the full corpus. I only caught it
because I printed a timer on a run I did not need to time.

Fixed by holding the registry in memory for the length of a run (`MeasureIndex`),
appending on mint instead of re-querying, and batch-encoding every distinct
phrase in one pass. **101.9s → 18.8s**, most of the remainder being the one-time
MiniLM load.

Lesson worth keeping: the embedding call was not the bottleneck. The `SELECT`
around it was.

### 8 — Wrong turn: a cache key that ignored the prompt · 8 Sep

Model calls are cached on disk by content hash, and I keyed on
`schema.__name__`. Then I rewrote a field description — the one that tells the
model a business segment is not an entity — re-ran, and got byte-identical
results.

Field descriptions **are** the prompt. The key now hashes the fully expanded
`model_json_schema()`. This was live for about an hour and would have made every
subsequent prompt iteration a no-op, which is the kind of bug that does not fail,
it just quietly stops you learning anything.

### 9 — Wrong turn: an empty unit is not the document's currency · 8 Sep

`parse_unit("")` fell back to the witness's default currency. Reasonable-looking,
and it turned `Processing centers: 160` into 160 *rupees* — which then sat in the
same dimension as revenue and was eligible to be compared against it.

An absent unit now means `count`. The currency fallback fires only when the unit
is a bare scale word (`"million"` with no symbol), which is the one case where
the currency really was elided.

### 10 — SQLite, and no ORM · 8 Sep

One file, no daemon, no container, no connection string. The reviewer's setup
instructions stay at one command, and the seeded database can be committed so
the demo does not depend on anyone holding an API key.

No ORM because the apparatus view is joins — claim to anchor to witness to
measure to entity — and an ORM would only obscure them. The schema is 14 tables
and fits in your head.

### 11 — Jinja and htmx, not React · 8 Sep

No bundler, no `node_modules`, no build step. `uvicorn collate.api:app` and it
runs.

Partly deadline pragmatism, but mostly fit: the interface is three panes of
dense tabular reading with one interaction (click a verdict, load its reasoning
and evidence). That is a document, and server rendering is good at documents.
The visual model is a printed critical edition — paper ground, hairline rules,
serif for prose, mono for anything a machine produced. No cards, no shadows.

### 12 — Evidence is a rendered page, not a citation · 8 Sep

`/evidence/{claim_id}.png` re-opens the source PDF, draws the anchor rectangle
on the page, and returns it. Every other view in the app is a *claim about* the
data; that endpoint is the data. It is also the cheapest possible defence
against the thing reviewers are right to suspect, which is that the numbers were
made up somewhere in the middle.

---

## Still wrong, or not done

- **Sparse pages are counted, not read.** 24 of 27 pages of the earnings deck
  have numbers locked inside chart graphics. Nothing is extracted from them and
  nothing is invented; they are reported on the quarantine page. The fix is a
  vision pass over flagged pages, which is a model call per page and did not fit
  the budget.
- **Entity resolution is edit-distance only.** Good for `Delhivery Limited` vs
  `Delhivery Ltd`, useless for `the Company` in a document whose front matter
  failed to parse.
- **`identity` claims are extracted but barely used.** The address-equivalence
  case works only when both surface forms appear with enough context.
- **The cascade is order-dependent and that order is a judgement.** Moving
  `r_agreement` above `r_basis_consolidation` would change verdicts. It is
  written to read like an argument, but it is still one person's argument.
