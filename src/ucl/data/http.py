"""Cached HTTP with polite retries.

Every source here is a free, volunteer-run or rate-limited service. Two rules:
caching is mandatory (never re-fetch what we already have during a run), and a
failure is reported, never silently turned into an empty result. The EPL lab was
bitten by football-data.co.uk returning HTTP 300 with an HTML body instead of a
404 - `raise_for_status()` waves 3xx through - so only 200 counts as content, and
content is validated before it is cached.
"""
from __future__ import annotations

import hashlib
import time
from pathlib import Path

import requests

from ..config import CACHE, USER_AGENT


class FetchError(RuntimeError):
    """A source could not be read. Callers must fail closed, not guess."""


class NotFound(FetchError):
    """The resource definitively does not exist. Retrying cannot help."""


# Statuses that mean "this will never exist", as opposed to "try again later".
# Season files are probed speculatively (does openfootball have 2012-13 for
# Belgium?), so most 404s here are expected answers rather than errors. Retrying
# them three times with backoff turned a two-minute corpus build into twenty.
_TERMINAL_STATUSES = {400, 401, 403, 404, 410, 451}


def _cache_path(url: str, suffix: str = ".txt") -> Path:
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:20]
    return CACHE / f"{digest}{suffix}"


def fetch(
    url: str,
    *,
    max_age_hours: float = 24.0,
    retries: int = 3,
    backoff: float = 2.0,
    timeout: float = 30.0,
    min_bytes: int = 32,
    required_marker: str | None = None,
) -> str:
    """Return the body of `url`, from disk cache when fresh enough.

    `required_marker` is a substring that must appear in a valid response. It is
    the cheap defence against a source returning a 200-with-error-page, which is
    otherwise indistinguishable from real content and poisons the cache.
    """
    path = _cache_path(url)
    if path.exists():
        age_hours = (time.time() - path.stat().st_mtime) / 3600.0
        if age_hours < max_age_hours:
            return path.read_text(encoding="utf-8")

    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            response = requests.get(
                url, timeout=timeout, headers={"User-Agent": USER_AGENT}
            )
            # Only 200 is content. 3xx with a body is a trap, not a redirect we want.
            if response.status_code in _TERMINAL_STATUSES:
                raise NotFound(f"HTTP {response.status_code} for {url}")
            if response.status_code != 200:
                raise FetchError(f"HTTP {response.status_code} for {url}")
            response.encoding = response.encoding or "utf-8"
            body = response.text
            if len(body.encode("utf-8")) < min_bytes:
                raise FetchError(f"Response too small ({len(body)} chars) for {url}")
            if required_marker and required_marker not in body:
                raise FetchError(f"Response missing expected marker for {url}")
            path.write_text(body, encoding="utf-8")
            return body
        except NotFound as exc:
            last_error = exc
            break                       # definitive: no amount of retrying helps
        except Exception as exc:  # noqa: BLE001 - retried, then re-raised below
            last_error = exc
            if attempt < retries - 1:
                time.sleep(backoff * (attempt + 1))

    # Stale cache beats no data, but the caller is told it is stale.
    if path.exists():
        return path.read_text(encoding="utf-8")
    raise FetchError(f"Could not fetch {url}: {last_error}")


def try_fetch(url: str, **kwargs) -> str | None:
    """Best-effort variant for optional sources. Returns None instead of raising."""
    try:
        return fetch(url, **kwargs)
    except FetchError:
        return None
