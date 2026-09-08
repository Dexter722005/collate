"""The one place Collate talks to a model.

Three things this file is careful about, all of them consequences of running on
a free tier rather than a billed one:

*   It negotiates a model instead of hard-coding one. The preference list is
    ordered by capability, but a name being *listed* does not mean the key can
    *call* it - free tiers gate models, and vendors rename them. So we probe
    with a one-token request and take the first that actually answers. It also
    means this still runs in six months when today's names are gone.

*   Every call is cached on disk by content hash, and the model is deliberately
    NOT part of that key. A re-ingest, a prompt tweak, a demo recorded three
    times - all free.

*   Requests are the scarce resource by a very wide margin. Every published
    source says the free tier allows 1,500 requests/day; the API actually
    enforces 20 per day, per model, against a million-token context. So the
    segmenter sends a handful of very large passages rather than hundreds of
    small ones, and this client rotates models as each allowance runs out.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

CACHE_DIR = Path(__file__).resolve().parent.parent / ".cache" / "llm"

# Best first. Probed in order, and rotated through as each one's daily
# allowance runs out - see `_rotate`. Aliases like gemini-flash-latest are kept
# near the end deliberately: they are a moving target, which is useful as a
# backstop and unhelpful as a default.
MODEL_PREFERENCE = [
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3-flash-preview",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-flash-latest",
    "gemini-flash-lite-latest",
]

_DAILY_QUOTA = re.compile(r"PerDay|RequestsPerDay", re.I)


def load_env(path: str | Path = ".env") -> None:
    """Minimal .env reader. A dependency for this would be silly."""
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$", line)
        if m and not line.lstrip().startswith("#"):
            os.environ.setdefault(m.group(1), m.group(2).strip().strip('"').strip("'"))


class RateLimiter:
    """Sliding-window limiter. Threads block here rather than eating a 429."""

    def __init__(self, per_minute: int):
        self.per_minute = max(1, per_minute)
        self._hits: list[float] = []
        self._lock = threading.Lock()

    def acquire(self) -> None:
        while True:
            with self._lock:
                now = time.monotonic()
                self._hits = [t for t in self._hits if now - t < 60.0]
                if len(self._hits) < self.per_minute:
                    self._hits.append(now)
                    return
                wait = 60.0 - (now - self._hits[0]) + 0.05
            time.sleep(max(wait, 0.05))


class ModelUnavailable(RuntimeError):
    pass


class Gemini:
    def __init__(
        self,
        conn: sqlite3.Connection | None = None,
        model: str | None = None,
        rpm: int | None = None,
        cache_dir: Path | None = None,
    ):
        load_env()
        key = os.environ.get("GEMINI_API_KEY", "").strip()
        if not key:
            raise ModelUnavailable(
                "GEMINI_API_KEY is unset. Copy .env.example to .env and paste a key "
                "from https://aistudio.google.com/apikey"
            )
        from google import genai  # imported late so tests can run without the SDK

        self._genai = genai
        self.client = genai.Client(api_key=key)
        self.conn = conn
        self.cache_dir = cache_dir or CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.limiter = RateLimiter(int(rpm or os.environ.get("COLLATE_RPM", 15)))
        self._lock = threading.Lock()
        self._pending: list[tuple] = []
        self._exhausted: set[str] = set()
        self._rotations = 0
        listed = self._listed()
        self._pool = [m for m in MODEL_PREFERENCE if not listed or m in listed]
        self.model = model or os.environ.get("COLLATE_MODEL") or self._negotiate()
        if self.model not in self._pool:
            self._pool.insert(0, self.model)

    # -- model discovery ----------------------------------------------------

    def _listed(self) -> set[str]:
        try:
            return {
                m.name.replace("models/", "")
                for m in self.client.models.list()
                if "generateContent" in (getattr(m, "supported_actions", None) or [])
            }
        except Exception:
            return set()

    def _step(self) -> str:
        """Move to the next healthy model without retiring the current one.

        For transient trouble - a 503 - where the model is fine and simply busy.
        Distinct from _rotate, which is for a model whose daily allowance is
        genuinely spent and which must not be tried again this session.
        """
        with self._lock:
            healthy = [m for m in self._pool if m not in self._exhausted]
            if len(healthy) > 1:
                i = healthy.index(self.model) if self.model in healthy else -1
                self.model = healthy[(i + 1) % len(healthy)]
            return self.model

    def _rotate(self) -> bool:
        """Retire the current model for this session and move to the next.

        Called when a model reports its per-day allowance gone. Returns False
        once the pool is empty, at which point the caller should surface the
        failure rather than spin. Thread-safe because a dozen workers will all
        discover the exhaustion within the same second and must not each burn a
        different model doing so.
        """
        with self._lock:
            if self.model not in self._exhausted:
                self._exhausted.add(self.model)
                self._rotations += 1
            for name in self._pool:
                if name not in self._exhausted:
                    if name != self.model:
                        self.model = name
                    return True
            return False

    def _negotiate(self) -> str:
        """Pick a model this key can genuinely call, not merely see."""
        for name in self._pool:
            try:
                self.limiter.acquire()
                self.client.models.generate_content(
                    model=name,
                    contents="ping",
                    config={"max_output_tokens": 1, "temperature": 0},
                )
                return name
            except Exception:
                continue
        raise ModelUnavailable(
            "No model in the pool answered. Tried: " + (", ".join(self._pool) or "(none)")
        )

    # -- calling ------------------------------------------------------------

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    def _log(self, purpose: str, cache_hit: bool, key: str, resp: Any, ms: int) -> None:
        """Buffer call telemetry in memory; the caller flushes it.

        This used to INSERT and commit inline. Every thread shares one sqlite
        connection, so each worker's commit had to queue behind whatever bulk
        inserting the main thread was doing - and measured throughput collapsed
        to zero calls per minute while claims kept landing. The workers were not
        waiting on the API at all, they were waiting on the database, to write a
        statistics row.
        """
        usage = getattr(resp, "usage_metadata", None)
        with self._lock:
            self._pending.append(
                (
                    purpose,
                    self.model,
                    int(cache_hit),
                    key[:16],
                    getattr(usage, "prompt_token_count", None) if usage else None,
                    getattr(usage, "candidates_token_count", None) if usage else None,
                    ms,
                )
            )

    def flush_log(self) -> int:
        """Write buffered telemetry. Called from the owning thread, never a worker."""
        if self.conn is None:
            return 0
        with self._lock:
            rows, self._pending = self._pending, []
        if rows:
            self.conn.executemany(
                "INSERT INTO llm_call (purpose, model, cache_hit, prompt_key, in_tokens,"
                " out_tokens, latency_ms) VALUES (?,?,?,?,?,?,?)",
                rows,
            )
            self.conn.commit()
        return len(rows)

    def json(
        self,
        purpose: str,
        system: str,
        user: str,
        schema: Any,
        *,
        temperature: float = 0.0,
        max_attempts: int = 4,
    ) -> Any:
        """One structured call. Returns parsed JSON, or raises after retries.

        The cache key deliberately includes the system prompt and the schema:
        editing a prompt should invalidate its cached answers, silently reusing
        them would make prompt iteration a lie.
        """
        # The full expanded schema, not its class name. Field descriptions ARE
        # the prompt - they are what tells the model an entity is not a segment -
        # so editing one has to invalidate its cached answers. Keying on
        # `schema.__name__` looked equivalent and quietly served stale results
        # through an entire prompt revision.
        if hasattr(schema, "model_json_schema"):
            schema_repr = json.dumps(schema.model_json_schema(), sort_keys=True)
        else:
            schema_repr = json.dumps(schema, sort_keys=True, default=str)

        # Note what is NOT in the key: the model. A cache entry means "this
        # passage, read under this prompt, into this schema" - and because the
        # daily allowance is per-model, a long ingest rotates through several.
        # Keying on the model would throw away most of the cache exactly when a
        # rerun matters most. Which model produced a given answer is recorded in
        # llm_call; this is a deliberate trade of provenance for reuse.
        key = hashlib.sha256(
            "\x00".join([purpose, system, user, schema_repr]).encode("utf-8")
        ).hexdigest()

        cached = self._cache_path(key)
        if cached.exists():
            try:
                payload = json.loads(cached.read_text(encoding="utf-8"))
                self._log(purpose, True, key, None, 0)
                return payload
            except ValueError:
                cached.unlink(missing_ok=True)

        cfg: dict[str, Any] = {
            "system_instruction": system,
            "response_mime_type": "application/json",
            "response_schema": schema,
            "temperature": temperature,
        }

        last: Exception | None = None
        for attempt in range(max_attempts):
            try:
                self.limiter.acquire()
                t0 = time.monotonic()
                resp = self.client.models.generate_content(
                    model=self.model, contents=user, config=cfg
                )
                ms = int((time.monotonic() - t0) * 1000)
                text = (resp.text or "").strip()
                if not text:
                    raise ValueError("empty response")
                payload = json.loads(text)
                cached.write_text(json.dumps(payload, indent=1), encoding="utf-8")
                self._log(purpose, False, key, resp, ms)
                return payload
            except Exception as exc:  # noqa: BLE001 - retried below, surfaced at the end
                last = exc
                msg = str(exc)
                low = msg.lower()
                if "api key" in low or "permission" in low:
                    break

                # A daily-allowance 429 is not worth waiting out - it clears at
                # midnight Pacific, not in thirty seconds. Retire the model for
                # this session and carry on with the next one.
                if "429" in msg and _DAILY_QUOTA.search(msg):
                    if self._rotate():
                        continue
                    break

                # 503 means that model is busy, not that we are out of budget.
                # Backing off on the same name just waits for someone else's
                # traffic to subside; stepping sideways to a peer usually costs
                # nothing. Five passages of the prospectus were lost to this
                # before the sideways step existed.
                if "503" in msg or "unavailable" in low or "overloaded" in low:
                    self._step()
                    time.sleep(1.5 + random.uniform(0, 1.5))
                    continue

                if attempt == max_attempts - 1:
                    break
                # Per-minute 429s do carry an honest retry hint. Honour it.
                hinted = re.search(r"retry.{0,12}?(\d+(?:\.\d+)?)\s*s", low)
                delay = float(hinted.group(1)) if hinted else (2.0 ** attempt) * 2.0
                time.sleep(min(delay, 45.0) + random.uniform(0, 1.0))
        raise RuntimeError(f"{purpose}: model call failed after {max_attempts} attempts: {last}")
