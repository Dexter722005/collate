"""Command line for Collate.

    python -m collate build data/starter-datasets   ingest, frame and adjudicate
    python -m collate ingest FILE...                add documents
    python -m collate frame                         resolve frames for new claims
    python -m collate adjudicate                    run the cascade
    python -m collate stats                         what is in the apparatus
    python -m collate cases                         the four required cases, found live
    python -m collate checkpoint                    fold the WAL in before committing
    python -m collate reset                         start over

build is incremental: documents already in the corpus by content hash are
skipped, and only their new claims are framed and compared.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from . import cases, collation, db, llm, normalize, pipeline


def _client(conn):
    return llm.Gemini(conn=conn)


def cmd_ingest(args, conn) -> None:
    client = _client(conn)
    print(f"model: {client.model}\n")
    paths: list[Path] = []
    for p in args.paths:
        path = Path(p)
        paths.extend(sorted(path.rglob("*.pdf")) if path.is_dir() else [path])

    for path in paths:
        t0 = time.time()
        try:
            out = pipeline.ingest(conn, client, path)
        except Exception as exc:  # noqa: BLE001 - one bad document must not stop a corpus
            print(f"  {path.name}: FAILED {type(exc).__name__}: {exc}")
            continue
        if out.get("skipped"):
            print(f"  {path.name}: already ingested (witness {out['witness_id']})")
            continue
        print(
            f"  {path.name}: {out['pages']}p, {out['proposed']} proposed, "
            f"{out['anchored']} anchored ({(out.get('anchor_rate') or 0):.0%}), "
            f"{out['quarantined']} quarantined, {out['sparse_pages']} sparse  "
            f"[{time.time() - t0:.0f}s]"
        )


def cmd_frame(args, conn) -> None:
    ids = [r["id"] for r in db.rows(conn, "SELECT id FROM witness ORDER BY id")]
    total = 0
    for wid in ids:
        n = normalize.normalize_witness(conn, wid)
        if n:
            print(f"  witness {wid}: framed {n}")
        total += n
    print(f"framed {total} claims")


def cmd_adjudicate(args, conn) -> None:
    client = None
    if not args.no_escalate:
        try:
            client = _client(conn)
        except Exception as exc:  # noqa: BLE001
            print(f"  (escalation disabled: {exc})")
    stats = collation.run(conn, client=client, escalate=client is not None)
    print(json.dumps(stats, indent=1))


def cmd_stats(args, conn) -> None:
    q = lambda sql: db.one(conn, sql)["c"]  # noqa: E731
    print(f"witnesses    {q('SELECT COUNT(*) c FROM witness')}")
    print(f"pages        {q('SELECT COUNT(*) c FROM page')}")
    print(f"claims       {q('SELECT COUNT(*) c FROM claim')}")
    print(f"  framed     {q('SELECT COUNT(*) c FROM claim WHERE frame_key IS NOT NULL')}")
    print(f"quarantined  {q('SELECT COUNT(*) c FROM quarantine')}")
    print(f"measures     {q('SELECT COUNT(*) c FROM measure')}")
    print(f"entities     {q('SELECT COUNT(*) c FROM entity')}")
    print(f"verdicts     {q('SELECT COUNT(*) c FROM verdict')}")
    print("\nby verdict:")
    for r in db.rows(conn, "SELECT verdict, COUNT(*) c FROM verdict GROUP BY verdict ORDER BY c DESC"):
        print(f"  {r['verdict']:<14} {r['c']}")
    print("\nby rule:")
    for r in db.rows(conn, "SELECT rule_id, COUNT(*) c FROM verdict GROUP BY rule_id ORDER BY c DESC"):
        print(f"  {r['rule_id']:<22} {r['c']}")
    print("\nanchoring:")
    for r in db.rows(conn, "SELECT match_kind, COUNT(*) c FROM anchor GROUP BY match_kind ORDER BY c DESC"):
        print(f"  {r['match_kind']:<14} {r['c']}")
    for r in db.rows(conn, "SELECT reason, COUNT(*) c FROM quarantine GROUP BY reason ORDER BY c DESC"):
        print(f"  quarantined: {r['reason']:<22} {r['c']}")
    calls = db.one(conn, "SELECT COUNT(*) c, SUM(cache_hit) h FROM llm_call")
    if calls and calls["c"]:
        print(f"\nmodel calls  {calls['c']} ({calls['h'] or 0} served from cache)")


def cmd_build(args, conn) -> None:
    cmd_ingest(args, conn)
    print()
    cmd_frame(args, conn)
    print()
    cmd_adjudicate(args, conn)
    print()
    cmd_stats(args, conn)
    print()
    cmd_checkpoint(args, conn)


def cmd_checkpoint(args, conn) -> None:
    """Fold the write-ahead log back into collate.db and compact it.

    Needed because the apparatus is committed to git. SQLite in WAL mode keeps
    recent writes in a `-wal` sidecar that is gitignored, so a freshly ingested
    corpus can look complete locally and arrive at a cloner missing everything
    since the last checkpoint. Run this before committing the database.
    """
    import sqlite3

    conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    conn.commit()
    conn.close()
    raw = sqlite3.connect(str(args.db or db.DB_PATH))
    raw.execute("VACUUM")
    raw.close()
    size = Path(args.db or db.DB_PATH).stat().st_size
    print(f"checkpointed and compacted: {size / 1e6:.1f} MB, safe to commit")


def cmd_reset(args, conn) -> None:
    if not args.yes:
        print("refusing without --yes (this drops every claim and verdict)")
        return
    for t in ["verdict", "anchor", "quarantine", "claim", "passage", "block", "page",
              "measure_alias", "measure", "entity_alias", "entity", "run_log", "llm_call",
              "witness"]:
        conn.execute(f"DELETE FROM {t}")
    conn.commit()
    print("apparatus emptied (the model-call cache on disk is kept)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="collate", description=__doc__)
    parser.add_argument("--db", default=None, help="path to the sqlite file")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("ingest"); p.add_argument("paths", nargs="+"); p.set_defaults(fn=cmd_ingest)
    p = sub.add_parser("frame"); p.set_defaults(fn=cmd_frame)
    p = sub.add_parser("adjudicate")
    p.add_argument("--no-escalate", action="store_true", help="rules only, no model calls")
    p.set_defaults(fn=cmd_adjudicate)
    p = sub.add_parser("stats"); p.set_defaults(fn=cmd_stats)
    p = sub.add_parser("cases", help="the four required cases, found in the data")
    p.set_defaults(fn=lambda a, c: cases.run(c))
    p = sub.add_parser("build")
    p.add_argument("paths", nargs="+")
    p.add_argument("--no-escalate", action="store_true")
    p.set_defaults(fn=cmd_build)
    p = sub.add_parser("checkpoint", help="fold the WAL back in before committing the db")
    p.set_defaults(fn=cmd_checkpoint)
    p = sub.add_parser("reset"); p.add_argument("--yes", action="store_true"); p.set_defaults(fn=cmd_reset)

    args = parser.parse_args(argv)
    conn = db.connect(args.db)
    args.fn(args, conn)
    return 0


if __name__ == "__main__":
    sys.exit(main())
