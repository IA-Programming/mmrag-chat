"""The ONLY place that creates HTTP clients (NFR-SEC-EGRESS, enforced by tests/unit/test_architecture.py).

Every request is checked against an allowlist before it leaves the process:
  * loopback (the local llama-server child),
  * the bundle release host(s) (GitHub Releases + their redirect CDNs),
  * the Gemini endpoint, only when the fallback is enabled.
User-supplied URLs are never fetched.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from urllib.parse import urlsplit

import httpx

from core.errors import EgressDeniedError

LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})
# GitHub release downloads redirect to these CDNs (verified in S01 against a real release).
RELEASE_HOSTS = frozenset(
    {
        "github.com",
        "objects.githubusercontent.com",
        "release-assets.githubusercontent.com",
    }
)
GEMINI_HOSTS = frozenset({"generativelanguage.googleapis.com"})


@dataclass
class EgressPolicy:
    extra_hosts: set[str] = field(default_factory=set)
    gemini_enabled: bool = False

    def allowed_hosts(self) -> frozenset[str]:
        hosts = set(LOOPBACK_HOSTS) | set(RELEASE_HOSTS) | self.extra_hosts
        if self.gemini_enabled:
            hosts |= GEMINI_HOSTS
        return frozenset(hosts)

    def check(self, url: str | httpx.URL) -> None:
        parts = urlsplit(str(url))
        host = (parts.hostname or "").lower()
        if host in LOOPBACK_HOSTS:
            if parts.scheme != "http":
                raise EgressDeniedError("Blocked a request to an unexpected local address.", detail=str(url))
            return
        if parts.scheme != "https":
            raise EgressDeniedError("Blocked a non-HTTPS request.", detail=f"scheme={parts.scheme} host={host}")
        if host not in self.allowed_hosts():
            raise EgressDeniedError("Blocked a request to a host outside the allowlist.", detail=f"host={host}")


_POLICY = EgressPolicy()


def policy() -> EgressPolicy:
    return _POLICY


def configure(*, gemini_enabled: bool | None = None, extra_hosts: Iterable[str] | None = None) -> None:
    """Called once at bootstrap (e.g. a fork's bundle_base_url host)."""
    if gemini_enabled is not None:
        _POLICY.gemini_enabled = gemini_enabled
    if extra_hosts:
        _POLICY.extra_hosts |= {h.lower() for h in extra_hosts}


def _guard(request: httpx.Request) -> None:
    _POLICY.check(request.url)


def _guard_response(response: httpx.Response) -> None:
    # Redirect targets are requests too: check the final URL of every hop.
    _POLICY.check(response.request.url)


def client(
    *,
    base_url: str = "",
    timeout: float | httpx.Timeout = 30.0,
    headers: dict[str, str] | None = None,
    follow_redirects: bool = False,
    transport: httpx.BaseTransport | None = None,
) -> httpx.Client:
    """Create an allowlisted synchronous client. Proxy env vars are ignored for loopback."""
    if base_url:
        _POLICY.check(base_url)
    return httpx.Client(
        base_url=base_url,
        timeout=timeout,
        headers=headers or {},
        follow_redirects=follow_redirects,
        trust_env=not base_url.startswith("http://127.0.0.1"),
        event_hooks={"request": [_guard], "response": [_guard_response]},
        transport=transport,
    )
