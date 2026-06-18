# Security Policy

## Reporting a vulnerability

If you discover a security vulnerability in the NemulAI agent, please report it
responsibly. **Do not open a public GitHub issue.**

Email **security@nemulai.com** with:

- A description of the vulnerability and its impact
- Steps to reproduce (proof-of-concept if possible)
- Affected version(s) and environment

We aim to acknowledge reports within **2 business days** and to provide a
remediation timeline within **5 business days**. We'll keep you updated through
disclosure and credit you in the release notes unless you prefer to remain
anonymous.

## Supported versions

Security fixes are provided for the latest released minor version on PyPI.
Please upgrade to the latest `nemulai` release before reporting.

## Scope

In scope:

- The `nemulai` agent (this repository)
- The documented ingest/command protocol between agent and API

Out of scope (report to the relevant surface instead):

- The hosted dashboard and web application at nemulai.com
- Third-party dependencies (report upstream; we'll track and bump)

## What the agent collects

By design, the agent collects **GPU telemetry** (power, utilization, memory,
temperature) and **metadata you explicitly tag** (team, model, job). It does
**not** read your model weights, datasets, or source code. It is **read-only by
default** and never changes workloads unless you opt into Advisor/Swarm mode,
which apply changes only with an observation window and automatic rollback.

Secrets (API keys) are read from environment variables and are never logged or
transmitted in metric payloads.
