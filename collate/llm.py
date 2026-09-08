"""The one place Collate talks to a model.

Three things this file is careful about, all of them consequences of running on
a free tier rather than a billed one:

*   It negotiates a model instead of hard-coding one. The preference list is
    ordered by capability, but a name being *listed* does not mean the key can
    *call* it - free tiers gate models, and vendors rename them. So we probe
    with a one-token request and take the first that actually answers. It also
    means this still runs in six months when today's names are gone.

*   Every call is cached on disk by content hash. A re-ingest, a re-run while
    tuning prompts, a demo recorded three times - all free. This is what makes
    a 600-page corpus tractable inside 1,500 requests/day.

*   Requests-per-minute is the binding constraint, not tokens-per-minute
    (15 RPM against 1M TPM). That asymmetry is why the segmenter batches whole
    pages: we would rather send 200 fat requests than 2,000 thin ones.
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

# Best first. Probed in order; the first that answers wins.
MODEL_PREFERENCE = [
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3-flash-preview",
    "gemini-2.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-2.5-flash-lite",
]


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
        self.model = model or os.environ.get("COLLATE_MODEL") or self._negotiate()

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

    def _negotiate(self) -> str:
        """Pick a model this key can genuinely call, not merely see."""
        listed = self._listed()
        candidates = [m for m in MODEL_PREFERENCE if not listed or m in listed]
        for name in candidates:
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
            "No model from the preference list answered. Listed for this key: "
            + (", ".join(sorted(listed)[:12]) or "(none)")
        )

    # -- calling ------------------------------------------------------------

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    def _log(self, purpose: str, cache_hit: bool, key: str, resp: Any, ms: int) -> None:
        if self.conn is None:
            return
        usage = getattr(resp, "usage_metadata", None)
        with self._lock:
            self.conn.execute(
                "INSERT INTO llm_call (purpose, model, cache_hit, prompt_key, in_tokens,"
                " out_tokens, latency_ms) VALUES (?,?,?,?,?,?,?)",
                (
                    purpose,
                    self.model,
                    int(cache_hit),
                    key[:16],
                    getattr(usage, "prompt_token_count", None) if usage else None,
                    getattr(usage, "candidates_token_count", None) if usage else None,
                    ms,
                ),
            )
            self.conn.commit()

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
        key = hashlib.sha256(
            "\x00".join([self.model, purpose, system, user, schema_repr]).encode("utf-8")
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
                msg = str(exc).lower()
                fatal = "api key" in msg or "permission" in msg
                if fatal or attempt == max_attempts - 1:
                    break
                # 429s carry a retry hint often enough to be worth honouring.
                hinted = re.search(r"retry.{0,12}?(\d+(?:\.\d+)?)\s*s", msg)
                delay = float(hinted.group(1)) if hinted else (2.0 ** attempt) * 2.0
                time.sleep(min(delay, 45.0) + random.uniform(0, 1.0))
        raise RuntimeError(f"{purpose}: model call failed after {max_attempts} attempts: {last}")
