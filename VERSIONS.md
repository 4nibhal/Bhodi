# Versioning Policy and Audit

## Policy

All direct dependencies in `pyproject.toml` are pinned with `==X.Y.Z`.
Container images in `Containerfile` and `podman-compose.yml` are pinned
to specific version tags (not `:latest`).

We do NOT use version ranges (`>=X,<Y`). The reason: a range lets a
newer release land in a build without any human review, and a malicious
or accidentally-broken release can slip in.

Dependabot (`.github/dependabot.yml`) generates PRs on every new release.
A human reviews and merges. CI enforces that the new lockfile resolves
cleanly via `uv lock --check`.

## Audit (2026-06-08)

Source of truth for CVEs: GitHub Advisory DB (OSV.dev mirror).

| Dep | Pinned | Reason / advisory |
|-----|--------|-------------------|
| fastapi | 0.136.3 | latest stable, no CVEs |
| starlette | 1.2.1 | transitive CVEs (PYSEC-2026-161, CVE-2025-54121, CVE-2025-62727) |
| uvicorn[standard] | 0.49.0 | latest stable, no CVEs |
| pydantic | 2.13.4 | latest stable (2.14.0a1 is alpha) |
| pypdf | 6.13.0 | latest stable, no CVEs |
| httpx | 0.28.1 | latest stable, no CVEs |
| aiohttp | 3.14.1 | latest stable, no CVEs |
| chromadb | 1.5.9 | CVE-2026-45829 server-side; not reachable from our embedded `PersistentClient` usage |
| openai | 2.41.0 | latest stable, no CVEs |
| numpy | 2.4.6 | 2.5.0rc1 is RC, not stable |
| llama-cpp-python | 0.3.28 | latest stable, no CVEs |
| ollama | 0.6.2 | latest stable, no CVEs |
| textual | 8.2.7 | latest stable, no CVEs |
| opentelemetry-api | 1.42.1 | latest stable, no CVEs |
| opentelemetry-sdk | 1.42.1 | latest stable, no CVEs |
| opentelemetry-exporter-otlp | 1.42.1 | latest stable, no CVEs |
| pytest | 9.0.3 | latest stable, no CVEs |
| pytest-asyncio | 1.4.0 | latest stable, no CVEs |
| pytest-cov | 7.1.0 | latest stable, no CVEs |
| pylint | 4.0.5 | latest stable, no CVEs |
| ruff | 0.15.16 | latest stable, no CVEs |
| mypy | 2.1.0 | latest stable, no CVEs |
| deptry | 0.25.1 | latest stable, no CVEs |

| Image | Pinned | Where |
|-------|--------|-------|
| python | 3.11-slim-bookworm | Containerfile (builder + runtime) |
| ghcr.io/astral-sh/uv | 0.11.19 | Containerfile (builder) |
| chromadb/chroma | -- (removed) | was unused and itself vulnerable; see `podman-compose.yml` history |

## How to bump

1. `uv lock --upgrade-package <name>` (or just `uv lock` for all)
2. `uv sync --frozen` to validate
3. `uv run pytest` to confirm nothing breaks
4. If the dep has a security advisory, update the comment in `pyproject.toml` and the table above
5. Open a PR

## How to audit

Re-run periodically (or on Dependabot alerts):
- `pip-audit` (already in CI's `security` job)
- `bandit` (already in CI's `security` job)
- Manual review against OSV.dev: `https://api.osv.dev/v1/query`
- Manual review against GitHub Advisory DB: `https://github.com/advisories`
