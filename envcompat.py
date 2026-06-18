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
Backward-compatible environment-variable access.

The agent's configuration variables are prefixed ``NEMULAI_``. Agents deployed
before the rebrand used the legacy ``ALUMINATAI_`` prefix. To avoid breaking
those deployments, every variable is read through :func:`env`, which prefers the
current ``NEMULAI_`` name and transparently falls back to the legacy name. The
current name always wins when both are set.

This module is intentionally free of import-time side effects so it can be
imported from anywhere (including experiment-tracker callbacks running inside a
user's training process) without touching the filesystem.
"""
import os

_LEGACY_PREFIX = "ALUMINATAI_"
_CURRENT_PREFIX = "NEMULAI_"


def legacy_name(name: str) -> str:
    """Return the legacy ``ALUMINATAI_*`` equivalent of a ``NEMULAI_*`` name.

    Returns an empty string for names without the current prefix.
    """
    if name.startswith(_CURRENT_PREFIX):
        return _LEGACY_PREFIX + name[len(_CURRENT_PREFIX):]
    return ""


def is_set(name: str) -> bool:
    """True if ``name`` or its legacy equivalent is present in the environment."""
    if name in os.environ:
        return True
    legacy = legacy_name(name)
    return bool(legacy) and legacy in os.environ


def env_from(environ, name: str, default: str = "") -> str:
    """Read ``name`` from a mapping, falling back to its legacy alias.

    Looks up the current ``NEMULAI_*`` name first, then the legacy
    ``ALUMINATAI_*`` name, then returns ``default``. Use this for environments
    other than the current process — e.g. another process's ``/proc/<pid>/environ``
    captured during job attribution.
    """
    val = environ.get(name)
    if val is not None:
        return val
    legacy = legacy_name(name)
    if legacy:
        val = environ.get(legacy)
        if val is not None:
            return val
    return default


def env(name: str, default: str = "") -> str:
    """Read ``name`` from ``os.environ``, falling back to its legacy alias.

    Looks up the current ``NEMULAI_*`` name first, then the legacy
    ``ALUMINATAI_*`` name, then returns ``default``.
    """
    return env_from(os.environ, name, default)


def has_known_prefix(name: str) -> bool:
    """True if ``name`` carries the current or legacy attribution prefix."""
    return name.startswith(_CURRENT_PREFIX) or name.startswith(_LEGACY_PREFIX)
