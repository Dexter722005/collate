# Collate

**A critical apparatus for documents.**

At each *lemma*, the *witnesses* offer *readings*. Two readings in the same
*frame* must agree; readings in different frames differ for a reason, and naming
that reason is the whole job.

---

Collate reads PDFs, extracts assertions, ties every one of them to a rectangle
on a page you can look at, and then works out — deterministically, with reasons
you can audit — which assertions corroborate each other, which genuinely
conflict, and which only *appear* to conflict until you notice they are measured
over different periods, on a different basis, or at a different stage of
revision.

The name is from textual criticism, where a *critical apparatus* is the record
of how surviving copies of a text disagree and why. That is exactly this
problem, and the discipline already had better words for it than I would have
invented. See [DECISIONS.md](DECISIONS.md) §2.

## The idea in one paragraph

The obvious way to model a fact is `(subject, predicate, value)`. That model
cannot express the interesting case: if a fact is just a triple, FY22 revenue
and FY24 revenue are two triples with the same subject and predicate and
different values — structurally indistinguishable from a contradiction, with
nothing left to explain *with*. So in Collate the qualifiers are part of the
claim's identity, not metadata hanging off it:

```
frame = (entity, measure, period, basis, unit dimension)
```

Two claims in the same frame are asserting the same thing and must agree. Two
claims in different frames are not in conflict at all — **and the coordinate
that differs is the explanation.**

---

## Setup and Run Instructions

Requires Python 3.11+. No database server, no node, no build step.

```bash
git clone <this repo> && cd collate
python -m venv .venv && . .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env                              # then paste a key into it
```

Get a free Gemini key at <https://aistudio.google.com/apikey> — the free tier is
a permanent rate-limited tier, not a trial, and it is all this project has ever
used.

**Run the interface against the committed corpus:**

```bash
uvicorn collate.api:app --reload
# open http://127.0.0.1:8000
```

**Rebuild the apparatus from the PDFs** (~12 minutes for all six documents;
every model call is cached on disk, so a second run is free):

```bash
python -m collate build data/starter-datasets
```

Both starter datasets are ingested in the committed apparatus. Gemini's free
tier enforces **20 requests per day per model**, not the 1,500 every published
source claims (DECISIONS §13), so the client rotates across nine flash models
and caches every call on disk — which is what makes 511 pages fit in a day.

**Add your own documents** — through the UI, or:

```bash
python -m collate ingest path/to/anything.pdf
python -m collate frame
python -m collate adjudicate
python -m collate stats
```

`build` is incremental. Documents already in the corpus by content hash are
skipped, and only new claims get framed and compared — adding the seventh
document does not rebuild the first six.

**Tests:**

```bash
python -m pytest -q
```

## Video Demo

**→ [demo video](PASTE_LINK_HERE)** (under 3 minutes)

## The four cases

`python -m collate cases` has the system find these itself, by querying the live
apparatus rather than reading a fixture — change the corpus and the examples
change with it. A captured run is in [samples/four-cases.txt](samples/four-cases.txt).

**Corpus:** all six starter documents — 511 pages, **1,967 anchored claims**,
391 quarantined, 852 measures discovered, 384 entities, **704 adjudicated pairs**
(34 corroborate, 77 contradict, 417 reconciled, 176 distinct).

### 1 · Corroborated across documents, expressed differently

| | Annual Report FY24, p.36 | Earnings deck, p.17 |
| --- | --- | --- |
| as printed | `₹85,942.34 million` | `8,594` under `₹ Cr` |
| normalised | **85,942,340,000 INR** | **85,940,000,000 INR** |

> "Total income increased by 14.13% to ₹85,942.34 million for FY24…"
> `Total income | 1,934 | 2,325 | 2,195 | (5.6%) | 13.5% | | 7,530 | 8,594 | 14.1%`

`UNIT_SCALE` → **corroborates**. One is prose, the other a nine-cell table row,
and they share almost no characters. Same number.

### 2 · A genuine contradiction

Two institutions, same period, same units, both calling it net FDI:

| | Economic Survey 2024-25, p.66 | RBI Annual Report 2024-25, p.85 |
| --- | --- | --- |
| | "For FY24 as a whole, the net FDI was **USD 10.1 billion**." | `Net Inward FDI (1.1.1 - 1.1.2) … 26.8  29.6` |
| period | FY24 | FY24 (2023-24 column) |

`CONTRADICTS` → escalated → the model **upholds** it (`UNRESOLVED_DISCREPANCY`),
noting the quotes carry no methodological context that would bridge a 62% gap.

That is the correct answer *on the evidence shown* — and the evidence pane
shows why it is also an incomplete one. Open the anchor and **footnote 95 is
sitting on the same page**, three inches below the highlighted paragraph:

> *Net FDI is calculated as follows: (1) FDI by foreigners: inflows + retained
> earnings − repatriation. (2) FDI by Indians: Indian investment overseas +
> retained earnings − repatriation. (3) The net FDI figure: (1)−(2)*

So the Survey nets out Indian outward investment and the RBI's "Net Inward FDI"
does not, and the two figures never disagreed. The reconciliation was in the
document all along, just outside the span the claim was anchored to.

I have deliberately left this as the headline example rather than tuning it
away, because it is the sharpest statement of where the system currently stops.
The model was not hallucinating and the rules were not wrong; the *unit of
evidence* was too small. Footnotes and table notes govern the numbers above them
exactly as scale headers govern the rows beneath them, and the table lane
already solves that problem in one direction. Extending governing-context
propagation to footnote markers is the single highest-value thing I would build
next, and it is a segmentation change, not a model change.

### 3 · An apparent contradiction explained by context

| Revenue from Operations, FY24 | `74,540.82 ₹ Million` | `81,415.38 ₹ Million` |
| --- | --- | --- |
| basis | **standalone** | **consolidated** |

`BASIS_MISMATCH` → **reconciled**. Same entity, measure, period and unit; a 9.2%
gap and no disagreement whatever. Both come from one four-column row, and this
only works because `(Standalone)` is lifted out of the measure's *name* into the
frame's basis — otherwise they are two unrelated measures that never meet.

Two more flavours the cascade produces on this corpus: `SIGN_CONVENTION` (the
annual report's table writes `(2,491.86)` where its own prose writes
`₹2,491.86 million` — same magnitude, accounting parentheses) and
`PERIOD_NESTED` (an eight-month figure inside its fiscal year).

### 4 · Failures, measured rather than asserted

**Extraction.** 391 of 2,358 proposed claims (**16.6%**) failed the anchoring gate
and were quarantined with their payloads, not silently dropped.

The instructive one: the RBI annual report first anchored at **39%**, with 587
rejections. The quarantined quotes read as fluent, plausible prose — e.g.
*"Global inflation eased to 5.7 per cent in 2024 from 6.6 per cent in 2023"* —
but the page actually says *"global inflation is expected to moderate from 5.7
per cent in 2024 to 4.3 per cent in 2025"*. The cause was in the segmenter, not
the model: **80 of its 100 pages are two-column**, and sorting blocks by `(y, x)`
interleaved the columns and shredded every sentence. The model was reconstructing
prose from fragments. Column-aware reading order took that document from
**39% → 71%** and added 419 claims. The gate had been catching all of it.

Separately, **24 of the earnings deck's 27 pages** carry almost no extractable
text — their numbers live inside chart graphics. Nothing was extracted and
nothing invented; they are counted as sparse at `/quarantine`.

**Reasoning.** Escalating contradictions to the model overruled a substantial
fraction of them as not contradictions at all — different denominators, different
service lines, managerial versus non-managerial staff, Part Truck Load versus
Truck Load. Those are measure-registry over-merges. The structural substitution
guard (DECISIONS §15) removed the worst class; the rest stay visible precisely
because the model's note sits *beside* the rule's verdict instead of replacing
it. A design where the LLM decided silently would have shown confident false
contradictions with no way to notice.

## Architecture

```
PDF
 │
 ├─ intake        content hash; one model pass over the front matter to fix
 │                document-level properties: publisher, VINTAGE (publication
 │                date, kept separate from any period), default currency + scale
 │
 ├─ segment       PyMuPDF. Page text is rebuilt block by block so every char
 │                offset maps back to a rectangle. Tables are carved out of the
 │                prose flow, serialised as pipe tables, and stamped with the
 │                scale/basis header above them.
 │
 ├─ elicit        passages of 3-4 whole pages -> structured claims. The prompt
 │                names no metric, currency or fiscal convention; the model
 │                reports the document's own words.
 │
 ├─ ANCHOR GATE   every quote must be found again in its page: exact, then
 │                whitespace-flattened, then fuzzy >= 88. Below that the claim is
 │                quarantined with its payload. No model is asked whether a
 │                model told the truth.
 │
 ├─ normalize     values ((452) is negative, 1,23,456 is Indian grouping),
 │                units (crore/lakh/mn/bn -> dimension + magnitude),
 │                periods (FY24 vs 2024-25 vs FY2024/25 -> real intervals),
 │                entities (alias sets), measures (the dynamic registry)
 │
 ├─ collate       block on the lemma; expand to embedding-neighbour measures
 │
 └─ adjudicate    the rule cascade -> verdict + reason; contradictions only
                  are escalated to the model, advisory, never overriding
```

### The rule cascade

Rules decide; the model explains. Each rule inspects one coordinate of the
frame, the first to fire claims the pair and names itself in the output.

| rule | fires when | verdict |
| --- | --- | --- |
| `DIMENSION_MISMATCH` | a level vs a rate | distinct |
| `STATE_SUPERSEDED` | a condition changed between vintages | reconciled |
| `PERIOD_UNSTATED` | either claim gives no period | distinct |
| `PERIOD_DISJOINT` | periods do not overlap | reconciled |
| `PERIOD_NESTED` | a quarter inside its year — part vs whole | reconciled |
| `PERIOD_PARTIAL` | periods overlap only partly | reconciled |
| `BASIS_MISMATCH` | consolidated vs standalone | reconciled |
| `ADJUSTMENT_MISMATCH` | real vs nominal, gross vs net | reconciled |
| `UNIT_SCALE` | equal after conversion, written differently | **corroborates** |
| `ROUNDING` | agree to 3 significant figures | corroborates |
| `EXACT` | identical | corroborates |
| `VINTAGE_REVISION` | advance estimate → provisional → actual | **reconciled** |
| `DEFINITION_DRIFT` | adjacent measures that did not merge | reconciled |
| `CONTRADICTS` | same frame, same period, same basis, different numbers | **contradicts** |

Only `CONTRADICTS` pairs reach the model, and its answer is written *beside* the
verdict, never into it. When the two disagree the UI shows both and says so —
more informative than either silently winning, and honest about which half of
the system is deterministic.

## Approach, and what it cost

**Why rules rather than an LLM judge.** The default design hands two claims to a
model and asks "do these conflict?". It is unstable between runs, cites no
mechanism, and is wrong in ways that look exactly like being right. It also
cannot produce the case the brief cares most about: saying *why* two figures
differ is date arithmetic and unit algebra, not language. The model is used
where it is genuinely better than code — reading messy prose into structure —
and kept away from the part that has to be auditable.

**Why the free tier changed the architecture, for the better.** Gemini's free
tier meters 15 requests/minute against 1,000,000 tokens/minute. Requests are the
scarce resource by a factor of thousands, so passages batch *whole pages* rather
than sliding windows: ~200 fat requests instead of ~2,000 thin ones. The
unplanned benefit is better extraction, because a table's scale header and its
rows now land in the same request.

**Why anchoring is a gate and not a score.** A claim that cannot be found in the
page it allegedly came from is not a low-confidence claim, it is not a claim.
Everything that fails is kept in `quarantine` and surfaced on its own page in
the UI, with the rejection reasons broken out.

**AI tools used.** Claude Code (Opus 5) for implementation throughout, pair-
programming style — I directed the architecture and the vocabulary, reviewed
every file, and several of the design arguments in `DECISIONS.md` came out of
that back-and-forth. Gemini 2.5/3.x Flash is the runtime model inside the
product itself, for extraction and for contradiction escalation. Embeddings are
local (MiniLM via sentence-transformers) so the dynamic schema keeps working
without network or quota.

**Things I got wrong and fixed**, kept with their diagnoses in
[DECISIONS.md](DECISIONS.md): a measure registry that re-read every embedding
per claim (101.9s → 18.8s once held in memory); a cache key that hashed the
schema's *class name*, so editing a prompt's field descriptions silently reused
stale answers; an empty unit inheriting the document's currency, which turned
"160 processing centers" into 160 rupees and made it comparable with revenue.

## Limitations and Next Steps

**Sparse pages are counted, not read.** 24 of the 27 pages of the Q4 FY24
earnings deck carry their numbers inside chart graphics rather than in the text
layer. Nothing is extracted from them, nothing is invented for them, and they
are reported as sparse on the quarantine page. This is the honest failure case;
the fix is a vision pass over flagged pages, which is one model call per page
and did not fit the request budget.

**Entity resolution is edit-distance only.** Fine for `Delhivery Limited` vs
`Delhivery Ltd`; useless for `the Company` in a document whose front-matter pass
failed. A coreference model, or simply trusting the front matter more
aggressively, would help.

**The cascade's order is a judgement.** Moving `r_agreement` above
`r_basis_consolidation` would change verdicts. It is written to read like an
argument, but it is one person's argument.

**`identity` claims are extracted but barely exercised.** Address equivalence
works only when both surface forms appear with enough surrounding context.

**Next, in order:** footnote-and-note propagation, so a claim inherits the
governing footnote the way a table row already inherits its scale header — the
net-FDI case above is one anchor span away from reconciling itself; the vision
pass over sparse pages; a proper confidence
interval on `CONTRADICTS` severity rather than a hand-rolled score; per-lemma
timeline view so a measure's revision history reads as a sequence rather than a
set of pairs; and moving embeddings into `sqlite-vec` so neighbour search stops
being a full scan.

## Additional Notes

- **No credentials in the repository.** `.env` is gitignored; `.env.example` is
  a blank template. A pre-built `collate.db` and a JSON dump in `samples/` are
  committed so the apparatus can be inspected, and the demo reproduced, without
  anyone holding a key.
- **Nothing is hard-coded to these documents.** No filename checks, no metric
  whitelist, no document-specific branches. The measure and entity registries
  start empty and are grown by whatever gets ingested. Point it at a lease or a
  clinical trial and the same machinery applies.
- **Page numbers.** The curated excerpts keep the page numbering printed in the
  original filings, so physical page 5 may print "26". Both are shown in the
  evidence pane.
- The starter PDFs are committed so the repo is self-contained.
