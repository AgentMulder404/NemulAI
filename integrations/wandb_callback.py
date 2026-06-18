# Copyright 2026 Kevin (NemulAI)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# NemulAI — https://github.com/AgentMulder404/NemulAI
"""
NemulAI Weights & Biases integration.

Auto-logs GPU energy and cost to W&B runs.

Usage:
    import wandb
    from agent.integrations.wandb_callback import NemulAIWandbCallback

    callback = NemulAIWandbCallback()

    run = wandb.init(project="my-project", entity="my-team")
    callback.on_run_start(run)
    # ... training ...
    callback.on_run_end(run)

Or use the context manager:
    with wandb.init(...) as run, callback.track(run):
        # ... training ...

Env vars:
    NEMULAI_API_KEY        Required for cost lookup
    NEMULAI_API_ENDPOINT   API base URL (default: production)
"""

from __future__ import annotations

import logging
import os

try:
    from ..envcompat import env
except (ImportError, ValueError):  # bare execution with repo root on sys.path
    from envcompat import env
import time
from contextlib import contextmanager
from typing import Generator, Optional

logger = logging.getLogger(__name__)

try:
    import wandb
    _WANDB = True
except ImportError:
    _WANDB = False


class NemulAIWandbCallback:
    """
    Logs GPU energy and cost to W&B runs.

    On run start:
      - Sets NEMULAI_MODEL to wandb.run.name
      - Sets NEMULAI_TEAM to wandb.run.entity/project

    On run end:
      - Fetches energy from NemulAI API
      - Logs to wandb.run.summary: energy_kwh, cost_usd, co2_kg
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_endpoint: Optional[str] = None,
    ):
        self.api_key = api_key or env("NEMULAI_API_KEY", "")
        self.api_endpoint = api_endpoint or env("NEMULAI_API_ENDPOINT", "https://nemulai.com/api/metrics/ingest"
        )
        self._start_times: dict[str, float] = {}

    def on_run_start(self, run) -> None:
        """Call after wandb.init()."""
        if not _WANDB:
            return

        run_id = run.id
        self._start_times[run_id] = time.time()

        try:
            team_id = f"{run.entity}/{run.project}" if run.entity else run.project
            os.environ["NEMULAI_MODEL"] = run.name or run_id
            os.environ["NEMULAI_TEAM"] = team_id
            logger.info("NemulAI: tracking W&B run '%s' (%s)", run.name, team_id)
        except Exception as exc:
            logger.warning("NemulAI WandbCallback on_run_start: %s", exc)

    def on_run_end(self, run) -> None:
        """Call before wandb.finish()."""
        if not _WANDB or not self.api_key:
            return

        run_id = run.id
        start_time = self._start_times.pop(run_id, None)
        if start_time is None:
            return

        try:
            energy = self._fetch_energy(start_time)
            if energy:
                run.summary.update({
                    "nemulai_energy_kwh": energy.get("today_kwh", 0),
                    "nemulai_cost_usd": energy.get("today_cost_usd", 0),
                })
                logger.info(
                    "NemulAI: logged %.4f kWh / $%.4f to W&B run %s",
                    energy.get("today_kwh", 0),
                    energy.get("today_cost_usd", 0),
                    run_id,
                )
        except Exception as exc:
            logger.warning("NemulAI WandbCallback on_run_end: %s", exc)
        finally:
            os.environ.pop("NEMULAI_MODEL", None)
            os.environ.pop("NEMULAI_TEAM", None)

    @contextmanager
    def track(self, run) -> Generator[None, None, None]:
        """Context manager wrapping on_run_start / on_run_end."""
        self.on_run_start(run)
        try:
            yield
        finally:
            self.on_run_end(run)

    def _fetch_energy(self, since: float) -> Optional[dict]:
        import urllib.request
        import json

        base = self.api_endpoint.removesuffix("/api/metrics/ingest").rstrip("/")
        url = f"{base}/api/metrics/cost-estimate"
        req = urllib.request.Request(
            url,
            headers={"X-API-Key": self.api_key},
            method="GET",
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                return json.loads(resp.read())
        except Exception as exc:
            logger.debug("energy fetch failed: %s", exc)
            return None
