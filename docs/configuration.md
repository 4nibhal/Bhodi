# bodhi-rag configuration

`bodhi-rag` reads its runtime configuration from four layers, applied
in this order (highest priority first):

1. **CLI flags** — `--config / -c PATH` on `bodhi-rag`
2. **env vars** — `BODHI_*` (and `BODHI_API_*` for the API server)
3. **TOML file** — `./bodhi.toml` (override path via `BODHI_CONFIG_PATH`
   or `--config`)
4. **built-in defaults** — Pydantic `default_factory` lambdas in
   `bodhi_rag/application/config.py`

Each higher layer **overrides** the fields in the lower layer; nothing
is merged across layers for the same field. The `tomllib` parser is
part of the Python 3.11+ standard library; no new runtime dependency
is added by this config infrastructure.

## TOML schema

The schema mirrors the Pydantic sub-configs in `application/config.py`.
The full example lives at [`bodhi.toml.example`](../bodhi.toml.example);
uncomment the sections you want to override.

```toml
[parser]
provider = "pypdf"

[chunker]
provider = "recursive"
chunk_size = 512
overlap = 64

[embedding]
provider = "openai"
model = "text-embedding-3-small"
batch_size = 100

[vector_store]
provider = "chroma"
persist_directory = "./data/chroma"
collection_name = "bodhi-rag"

[llm]
provider = "openai"
model = "gpt-4o-mini"
temperature = 0.7

[conversation]
provider = "volatile"
max_history = 50

[reranker]
provider = "noop"           # noop | cross_encoder
# model = "<required when provider = 'cross_encoder'>"
# top_k = 5
# overfetch_factor = 4
# batch_size = 32

[telemetry]
enabled = true
service_name = "bodhi-rag"
exporter = "console"         # console | otlp | none
otlp_endpoint = "http://localhost:4317"
```

Unknown top-level sections and unknown keys inside a known section
are ignored (the root model uses `model_config = ConfigDict(extra="ignore")`).
A typo in a section name will silently fall through to the default
layer; use the env-var or CLI flag if you need a fast feedback loop.

## Env-var ↔ TOML mapping

Every field in every sub-config has an `os.getenv("BODHI_*")` default
in `application/config.py`. The pattern is:

| TOML key | Env var | Default |
|----------|---------|---------|
| `parser.provider` | `BODHI_PARSER_PROVIDER` | `"pypdf"` |
| `chunker.provider` | `BODHI_CHUNKER_PROVIDER` | `"recursive"` |
| `embedding.provider` | `BODHI_EMBEDDING_PROVIDER` | `"openai"` |
| `embedding.model` | — | `None` |
| `vector_store.provider` | `BODHI_VECTOR_STORE_PROVIDER` | `"chroma"` |
| `llm.provider` | `BODHI_LLM_PROVIDER` | `"openai"` |
| `llm.model` | — | `None` |
| `conversation.provider` | `BODHI_CONVERSATION_PROVIDER` | `"volatile"` |
| `reranker.provider` | `BODHI_RERANKER_PROVIDER` | `"noop"` |
| `reranker.model` | `BODHI_RERANKER_MODEL` | `None` |

API-server-only env vars (consumed by `interfaces/api/app.py` and
`interfaces/api/server.py`, not by `BhodiConfig`):

| Env var | Default | Purpose |
|---------|---------|---------|
| `BODHI_API_HOST` | `127.0.0.1` | API server bind host (overridden by `bodhi-rag-api --host`) |
| `BODHI_API_PORT` | `8000` | API server bind port |
| `BODHI_API_SOURCE_ROOT` | unset | When set, constrains local-file ingest to that directory |
| `BODHI_CONFIG_PATH` | unset | Path to a TOML config file |

## Config path resolution

When you start `bodhi-rag` (or `bodhi-rag-api`), the loader searches for
the TOML file in this order:

1. `--config / -c PATH` on the CLI (highest priority)
2. `BODHI_CONFIG_PATH` env var
3. `./bodhi.toml` (relative to the current working directory)
4. **skip the file layer silently** (no error)

A `--config` path that points at a non-existent file is **silently
skipped** (per the spec: "skip file layer silently"). The implicit
`./bodhi.toml` default is also allowed to be missing. A malformed TOML
is the only file-level error you can get from the loader.

## ConfigError cases

`ConfigError` is a subclass of `ValueError`. The loader raises it for:

- **Malformed TOML** — syntax error in the file. The error message
  contains the file path and the line / column from `tomllib.TOMLDecodeError`.
- **Missing TOML file** — the loader silently skips a missing file
  (whether the path came from `--config`, `BODHI_CONFIG_PATH`, or the
  `./bodhi.toml` default) and falls through to the env / default layer.
  No `ConfigError` is raised.
- **Invalid section value** — e.g. `chunk_size = "abc"`. The error
  message contains the file path and the section name.
- **Missing required field** — e.g. `[reranker] provider = "cross_encoder"`
  with no `model`. The Pydantic `model_validator` on `RerankerConfig`
  raises `ConfigError` with the field name and a remediation hint.
- **Explicit `--config` path that does not exist** — the error
  message contains the path.

`ConfigError` messages always include the file path (when the TOML
layer is the source), the field name, and the layer where the problem
was detected. Example:

```
ConfigError: RerankerConfig.model is required when provider is
"cross_encoder". Set BODHI_RERANKER_MODEL or define [reranker]
model in bodhi.toml.
```

## Worked example: cross-encoder reranker

The `cross_encoder` reranker provider (Wave 3a) requires an explicit
model name. There is no built-in default — that is the policy. To
turn it on:

1. **TOML file** (recommended for production):

   ```toml
   [reranker]
   provider = "cross_encoder"
   model = "cross-encoder/ms-marco-MiniLM-L-6-v2"
   ```

   Save this as `bodhi.toml` at the repo root. `bodhi-rag` will load
   it on startup.

2. **Env var** (recommended for one-off runs):

   ```bash
   export BODHI_RERANKER_PROVIDER=cross_encoder
   export BODHI_RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
   bodhi-rag query "What is the termination clause?"
   ```

3. **CLI override** (highest priority; not yet wired for the reranker
   in the `bodhi-rag` CLI, but available to library users via
   `load_bodhi_config(cli_overrides={"reranker": {"provider": "cross_encoder"}})`):

   ```python
   from bodhi_rag.application.config_loader import load_bodhi_config
   config = load_bodhi_config(
       cli_overrides={"reranker": {"provider": "cross_encoder"}},
   )
   ```

If you forget to set `model`, the validator raises `ConfigError` at
construction time (not at first query), with a clear remediation hint.
The cross-encoder adapter itself is a Wave 3a deliverable; the
`RerankerConfig` schema and the `ConfigError` validator are part of
Wave 1 to establish the no-hardcoded-defaults contract for all
adapters going forward.

## Validation before process startup

Both the `bodhi-rag` CLI and the `bodhi-rag-api` server call
`load_bodhi_config()` at startup, so a misconfigured TOML file is
reported immediately and the process exits with code 2 (CLI) or a
clear `ConfigError: ...` traceback (API). The API server wraps the
`ConfigError` in `SystemExit` so the uvicorn worker does not start
with a broken config.
