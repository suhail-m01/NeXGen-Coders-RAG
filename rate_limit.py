"""A per-client rate limiter for the routes that run models.

Why this exists
---------------
`/query`, `/query_file` and `/ingest` each load and run local models. (`/login`
is not limited here: it burns CPU on PBKDF2 rather than on a model, and
`auth_max_failed_attempts` already locks an account after five wrong passwords,
which is the attack that endpoint actually faces.) A warm question takes about eleven seconds on the reference laptop and a
cold one half a minute; two at once contend for the same 4 GB of VRAM. A client
stuck in a retry loop, a browser tab left open on a polling script, or somebody
dragging a folder of a hundred PDFs onto the upload box can make the service
useless for everyone else on that machine -- and unlike a normal web service,
this one degrades the whole computer, not just itself. Refusing early with 429
is much cheaper than queueing work that would be slow anyway.

What this is not
----------------
It is not abuse-proof, not distributed and not a replacement for authentication.
State lives in process memory, so it counts per replica: run two uvicorn workers
and the effective limit is twice what is configured. That is stated rather than
hidden, because a limit that silently doubles is a limit nobody can reason about.
It also cannot see through a proxy unless `rate_limit_trust_forwarded_for` is on,
and a client with several tokens is several clients as far as this is concerned.

Why hand-rolled
---------------
`slowapi` does exactly this job with a decorator, and the design of `client_key`
below is ported from a project that uses it. It is not used here because this
ships as an offline Windows kit: every extra dependency is another download,
another pin to audit and another thing that can fail on a machine with no
network. A token bucket in ~140 lines with no dependencies is the better trade
for a service that is meant to run on one laptop.

The algorithm is a token bucket rather than a fixed window because a fixed
window lets a client send twice the limit straddling a window boundary, and
because a bucket gives the two things a demo actually needs: a burst allowance
for a presenter clicking through questions, and a steady sustained rate for
everything else.
"""

from __future__ import annotations

import ipaddress
import logging
import threading
import time

logger = logging.getLogger(__name__)

# Buckets, and the settings that size them. A bucket is a named cost class, not a
# route: /query, /query_file and /query_stream all draw from "queries" because
# they run the same models, and /ingest draws from "ingest" because embedding a
# document is a different shape of work and a folder drop should not be refused
# for the same price as a question.
BUCKETS = {
    "queries": ("rate_limit_queries_per_minute", "rate_limit_queries_burst"),
    "ingest": ("rate_limit_ingest_per_minute", "rate_limit_ingest_burst"),
}

# A client that has been silent this long stops being tracked, so the table cannot
# grow without bound on a service that faces the internet.
IDLE_SECONDS = 900.0

# How often to sweep idle clients, in number of checks rather than wall time: a
# sweep on every call would make the limiter quadratic under load.
_SWEEP_EVERY = 256


class _Bucket:
    """One client's allowance in one cost class."""

    __slots__ = ("tokens", "updated")

    def __init__(self, tokens: float) -> None:
        self.tokens = tokens
        self.updated = time.monotonic()


class RateLimiter:
    """Token buckets keyed by client, sized from settings on every check.

    Settings are read per check rather than cached at construction so an operator
    can change `.env` and restart, or a test can tighten a limit, without the
    limiter holding on to numbers from import time.
    """

    def __init__(self) -> None:
        self._buckets: dict[str, dict[str, _Bucket]] = {name: {} for name in BUCKETS}
        self._lock = threading.Lock()
        self._checks = 0
        self._refusals: dict[str, int] = {name: 0 for name in BUCKETS}

    # -- configuration ------------------------------------------------------
    @staticmethod
    def limits(bucket: str) -> tuple[float, float, bool]:
        """(refill per second, burst capacity, enabled) for one bucket."""
        from config import settings

        per_minute_field, burst_field = BUCKETS[bucket]
        per_minute = float(getattr(settings, per_minute_field, 0.0) or 0.0)
        burst = float(getattr(settings, burst_field, 0.0) or 0.0)
        enabled = bool(settings.rate_limit_enabled) and per_minute > 0 and burst > 0
        return per_minute / 60.0, burst, enabled

    # -- the check ----------------------------------------------------------
    def check(self, bucket: str, key: str, cost: int = 1) -> tuple[bool, float]:
        """Take `cost` tokens for `key` from `bucket`.

        Returns (allowed, retry_after_seconds). `retry_after_seconds` is 0.0 when
        the request is allowed and the time until `cost` tokens exist when it is
        not, which is what a `Retry-After` header should carry.

        `cost` matters because `/ingest` accepts a *list* of files: charging one
        token per request would let a single call do a hundred documents' worth of
        embedding work for the price of one. A request is charged one token per
        file, capped at the burst capacity, so a large first indexing run is never
        refused for being large -- it drains the bucket instead, and the next
        request waits. Refusing a legitimate 200-file upload outright would be a
        worse outcome than the unevenness this leaves.
        """
        if bucket not in BUCKETS:
            raise KeyError(f"Unknown rate-limit bucket '{bucket}'.")
        rate, capacity, enabled = self.limits(bucket)
        if not enabled:
            return True, 0.0
        wanted = max(1.0, min(float(cost), capacity))

        now = time.monotonic()
        with self._lock:
            table = self._buckets[bucket]
            self._checks += 1
            if self._checks % _SWEEP_EVERY == 0:
                self._sweep(now)
            entry = table.get(key)
            if entry is None:
                # A new client starts full: the limit is on sustained load, and
                # refusing somebody's very first question would be nonsense.
                entry = _Bucket(capacity)
                table[key] = entry

            elapsed = max(0.0, now - entry.updated)
            entry.tokens = min(capacity, entry.tokens + elapsed * rate)
            entry.updated = now

            if entry.tokens >= wanted:
                entry.tokens -= wanted
                return True, 0.0

            self._refusals[bucket] += 1
            return False, (wanted - entry.tokens) / rate

    def _sweep(self, now: float) -> None:
        """Drop clients that have been idle past IDLE_SECONDS. Caller holds the lock."""
        removed = 0
        for table in self._buckets.values():
            stale = [key for key, entry in table.items() if now - entry.updated > IDLE_SECONDS]
            for key in stale:
                del table[key]
            removed += len(stale)
        if removed:
            logger.debug("Rate limiter dropped %d idle client buckets", removed)

    # -- reporting and tests ------------------------------------------------
    def snapshot(self) -> dict:
        """What is configured and how many clients are being tracked.

        Reported through /health so an operator can see a limit is on, and at what
        number, without reading the source or guessing from a 429.
        """
        from config import settings

        out: dict = {"enabled": bool(settings.rate_limit_enabled), "buckets": {}, "refusals": {}}
        with self._lock:
            for bucket in BUCKETS:
                rate, capacity, enabled = self.limits(bucket)
                out["buckets"][bucket] = {
                    "per_minute": round(rate * 60.0, 3),
                    "burst": capacity,
                    "active": enabled,
                    "tracked_clients": len(self._buckets[bucket]),
                }
                out["refusals"][bucket] = self._refusals[bucket]
        out["scope"] = "this process only -- two workers mean twice the limit"
        return out

    def reset(self) -> None:
        """Clear every bucket. Tests call this; nothing in the service does."""
        with self._lock:
            for table in self._buckets.values():
                table.clear()
            self._checks = 0
            for bucket in self._refusals:
                self._refusals[bucket] = 0


def client_key(request, account: dict | None = None) -> str:
    """The identity whose requests share one bucket.

    An authenticated account is keyed by name, not by address: a token is issued
    to a person, and keying on it means rotating addresses does not create fresh
    allowance. When `api_auth_enabled` is false there is no such identity -- the
    dependency hands every caller the same placeholder account -- so the key falls
    back to the address, or every client on the network would share one bucket.

    On the address itself, the reasoning is borrowed from a deployment that ran
    behind a managed front end. When `rate_limit_trust_forwarded_for` is on, the
    socket peer is the proxy for every visitor, so keying on it puts the whole
    world in one bucket; the front end appends the address it received the request
    from to `X-Forwarded-For`, and the *rightmost* entry is the one a client cannot
    forge -- anything to its left is whatever the client chose to send. The flag
    defaults to off because trusting a header any client can set turns the limiter
    into a suggestion.

    An IPv6 client typically holds a whole /64 and can rotate inside it, so its
    bucket is the /64.
    """
    if account and account.get("via") != "disabled":
        username = str(account.get("username") or "").strip()
        if username:
            return f"account:{username}"

    address = None
    client = getattr(request, "client", None)
    if client is not None:
        address = client.host

    from config import settings

    if getattr(settings, "rate_limit_trust_forwarded_for", False):
        forwarded = request.headers.getlist("x-forwarded-for") if request.headers else []
        hops = [hop.strip() for chunk in forwarded for hop in chunk.split(",") if hop.strip()]
        if hops:
            address = hops[-1]

    if not address:
        return "address:unknown"
    try:
        ip = ipaddress.ip_address(address)
    except ValueError:
        return f"address:{address}"
    if ip.version == 6:
        return f"address:{ipaddress.ip_network(f'{ip}/64', strict=False)}"
    return f"address:{ip}"


_limiter = RateLimiter()
_limiter_lock = threading.Lock()


def get_limiter() -> RateLimiter:
    """The process-wide limiter. One instance, so buckets are shared across routes."""
    global _limiter
    if _limiter is None:  # pragma: no cover - defensive, reset() never sets None
        with _limiter_lock:
            if _limiter is None:
                _limiter = RateLimiter()
    return _limiter
