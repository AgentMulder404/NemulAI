# Copyright 2026 Kevin (NemulAI)
# SPDX-License-Identifier: Apache-2.0
#
# NemulAI — https://github.com/AgentMulder404/NemulAI
"""
ThroughputProbe — application-level throughput (tokens/s, samples/s).

NVML "utilization" only says a kernel was resident — a memory-stalled kernel
reads 100% at any clock, so it cannot tell us whether an optimization hurt
real work. This probe scrapes the inference server's own counters and turns
them into a true tokens/s signal for the autopilot observation window and
the bandit reward.

Sources are Prometheus-format /metrics endpoints. Known counters:

  vLLM    vllm:generation_tokens_total
  TGI     tgi_request_generated_tokens_sum
  SGLang  sglang:generation_tokens_total
  custom  nemulai_throughput_total  (emit this from your own server)

Configuration (env):

  THROUGHPUT_SOURCES="http://localhost:8000/metrics=0,1;http://localhost:8001/metrics"

Each entry is URL[=gpu,gpu,...]. Without an explicit GPU list the endpoint's
throughput applies to every GPU (single-server single-GPU is the common case).
Rates are computed from counter deltas between scrapes, so the first sample
returns nothing.
"""
from __future__ import annotations

import logging
import re
import time
import urllib.request
from dataclasses import dataclass, field
from typing import Optional

log = logging.getLogger(__name__)

# Counter names recognized as "generated work" — summed per endpoint
KNOWN_COUNTERS = (
    "vllm:generation_tokens_total",
    "vllm_generation_tokens_total",
    "tgi_request_generated_tokens_sum",
    "sglang:generation_tokens_total",
    "sglang_generation_tokens_total",
    "nemulai_throughput_total",
)

# Prometheus line: name{labels} value [timestamp]
_PROM_LINE = re.compile(
    r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{[^}]*\})?\s+([0-9eE+.\-]+)"
)

MIN_SCRAPE_INTERVAL_S = 10.0


@dataclass
class EndpointSource:
    url: str
    gpu_indices: Optional[list[int]]  # None = applies to all GPUs
    last_total: float = 0.0
    last_scrape_ts: float = 0.0
    last_rate: float = 0.0
    consecutive_failures: int = 0


def parse_sources(spec: str) -> list[EndpointSource]:
    """Parse THROUGHPUT_SOURCES: "url[=g,g,...];url2..." -> sources."""
    sources: list[EndpointSource] = []
    for raw in spec.split(";"):
        raw = raw.strip()
        if not raw:
            continue
        if "=" in raw:
            url, _, gpus = raw.rpartition("=")
            try:
                indices = [int(g) for g in gpus.split(",") if g.strip() != ""]
            except ValueError:
                log.warning("THROUGHPUT_SOURCES: bad GPU list in %r, applying to all GPUs", raw)
                url, indices = raw, None
            sources.append(EndpointSource(url=url, gpu_indices=indices or None))
        else:
            sources.append(EndpointSource(url=raw, gpu_indices=None))
    return sources


def parse_prometheus_total(text: str, counters=KNOWN_COUNTERS) -> Optional[float]:
    """Sum all samples of known work counters in a Prometheus exposition body.

    Returns None when no known counter is present (so callers can distinguish
    "endpoint up but not an inference server" from a zero counter).
    """
    total = 0.0
    found = False
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        m = _PROM_LINE.match(line)
        if not m:
            continue
        name, value = m.group(1), m.group(2)
        if name in counters:
            try:
                total += float(value)
                found = True
            except ValueError:
                continue
    return total if found else None


class ThroughputProbe:
    """Scrapes configured endpoints and exposes per-GPU tokens/s."""

    def __init__(
        self,
        sources_spec: str = "",
        scrape_interval_s: float = MIN_SCRAPE_INTERVAL_S,
        timeout_s: float = 2.0,
    ):
        self._sources = parse_sources(sources_spec)
        self._interval = max(MIN_SCRAPE_INTERVAL_S, float(scrape_interval_s))
        self._timeout = timeout_s

    @property
    def configured(self) -> bool:
        return len(self._sources) > 0

    @property
    def sources(self) -> list[EndpointSource]:
        return self._sources

    def sample(self) -> dict[int, float]:
        """Return {gpu_index: tokens_per_sec}. -1 key = unmapped (all GPUs).

        Respects the scrape interval per endpoint; between scrapes the last
        computed rate is reused so the agent loop gets a stable signal.
        """
        now = time.time()
        rates: dict[int, float] = {}

        for src in self._sources:
            if now - src.last_scrape_ts >= self._interval:
                self._scrape(src, now)

            if src.last_rate <= 0:
                continue
            targets = src.gpu_indices if src.gpu_indices is not None else [-1]
            for idx in targets:
                rates[idx] = rates.get(idx, 0.0) + src.last_rate

        return rates

    def rate_for_gpu(self, gpu_index: int, rates: dict[int, float]) -> float:
        """Resolve a GPU's throughput from a sample() result (0.0 if none)."""
        if gpu_index in rates:
            return rates[gpu_index]
        return rates.get(-1, 0.0)

    def _scrape(self, src: EndpointSource, now: float) -> None:
        try:
            req = urllib.request.Request(src.url, headers={"Accept": "text/plain"})
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                body = resp.read().decode("utf-8", errors="replace")
        except Exception as exc:
            src.consecutive_failures += 1
            if src.consecutive_failures in (1, 10):
                log.warning("Throughput scrape failed for %s: %s", src.url, exc)
            src.last_scrape_ts = now
            src.last_rate = 0.0
            return

        total = parse_prometheus_total(body)
        if total is None:
            src.consecutive_failures += 1
            if src.consecutive_failures == 1:
                log.warning(
                    "No known throughput counters at %s (expected one of %s)",
                    src.url, ", ".join(KNOWN_COUNTERS[:3]) + ", ...",
                )
            src.last_scrape_ts = now
            src.last_rate = 0.0
            return

        src.consecutive_failures = 0

        if src.last_scrape_ts > 0 and total >= src.last_total:
            dt = now - src.last_scrape_ts
            if dt > 0:
                src.last_rate = (total - src.last_total) / dt
        else:
            # First scrape, or counter reset (server restart) — no rate yet
            src.last_rate = 0.0

        src.last_total = total
        src.last_scrape_ts = now
