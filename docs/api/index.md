# Bhodi API Reference

## Base URL

```
http://localhost:8000
```

The default bind address is `127.0.0.1:8000`. Override with `BHODI_API_HOST` / `BHODI_API_PORT` or with `bhodi-api --host / --port`.

## Authentication

The API has **no authentication**. It is intended to run behind a VPN or a reverse proxy that enforces authentication. Do not expose it directly to the internet.

## Endpoints

### `GET /health`

Liveness and adapter-readiness probe. Calls `app.health_check()` and returns the resulting `HealthStatus` (from `bhodi_platform.application.models`). If any of `embedding`, `vector_store`, or `llm` is missing, the response is `degraded` and the server returns **HTTP 503**. The response body is the `HealthStatus` model:

```json
{
  "status": "healthy",
  "version": "1.0.0",
  "services": {
    "embedding": true,
    "vector_store": true,
    "llm": true
  }
}
```

The `/health` route is excluded from the API rate limiter.

---

### `POST /documents`

Index a document for later querying. The body is an `IndexDocumentRequest` and the response is an `IndexDocumentResponse` (HTTP `201 Created` on success).

**Request body**

```json
{
  "source": "./document.pdf",
  "metadata": { "author": "John Doe", "date": "2024-01-15" },
  "chunk_size": 512,
  "overlap": 64
}
```

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `source` | string | Yes | — | Path to a local document (`pdf`, `txt`, `md`, `rst`) |
| `metadata` | object | No | `{}` | Arbitrary metadata merged into the parsed document |
| `chunk_size` | integer | No | From config | Target chunk size in characters |
| `overlap` | integer | No | From config | Overlap between consecutive chunks |

When `BHODI_API_SOURCE_ROOT` is set, the resolved path of `source` must stay within that root, and the file extension must be one of `.pdf`, `.txt`, `.md`, `.rst`. Without `BHODI_API_SOURCE_ROOT`, local file ingest via the API is rejected with HTTP 400.

**Response body (201 Created)**

```json
{
  "document_id": "550e8400-e29b-41d4-a716-446655440000",
  "chunk_count": 42
}
```

**Status codes**

- `201 Created` — document indexed successfully
- `400 Bad Request` — invalid request (path traversal, missing file, unsupported extension, etc.)
- `500 Internal Server Error` — unexpected server error

---

### `DELETE /documents/{document_id}`

Remove a document and all of its chunks from the vector store.

**Path parameters**

| Parameter | Type | Description |
|-----------|------|-------------|
| `document_id` | string | Document id returned by `POST /documents` |

**Response body (200 OK)**

```json
{
  "deleted": true,
  "document_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

**Status codes**

- `200 OK` — document deleted
- `400 Bad Request` — invalid `document_id` value
- `404 Not Found` — no document with that id
- `500 Internal Server Error` — unexpected server error

---

### `POST /query`

Ask a question against the indexed documents. The body is a `QueryRequest`; the response is a `QueryResponse` with the generated answer and citations.

**Request body**

```json
{
  "question": "What is the main topic?",
  "conversation_id": "550e8400-e29b-41d4-a716-446655440001",
  "top_k": 5,
  "temperature": 0.7
}
```

| Field | Type | Required | Default | Constraints | Description |
|-------|------|----------|---------|-------------|-------------|
| `question` | string | Yes | — | — | The question to answer |
| `conversation_id` | string | No | `null` | — | Session id for conversation continuity |
| `top_k` | integer | No | `5` | `1 ≤ top_k ≤ 100` | Number of chunks to retrieve |
| `temperature` | float | No | `0.7` | `0.0 ≤ temperature ≤ 2.0` | LLM sampling temperature |

**Response body (200 OK)**

```json
{
  "answer_text": "The main topic of the document is...",
  "citations": [
    {
      "chunk_id": "550e8400-e29b-41d4-a716-446655440000:0",
      "text": "First paragraph of the document...",
      "source_document": "manual.pdf",
      "page": 1
    }
  ],
  "conversation_id": "550e8400-e29b-41d4-a716-446655440001"
}
```

| Field | Type | Description |
|-------|------|-------------|
| `answer_text` | string | Generated answer |
| `citations` | array of `CitationResponse` | Source chunks that grounded the answer |
| `conversation_id` | string \| null | Echo of the request id, or the generated one |

Each `CitationResponse` has `chunk_id` (string), `text` (truncated source text), `source_document` (filename or document id), and optional `page` (int).

**Status codes**

- `200 OK` — query processed
- `400 Bad Request` — invalid request body
- `500 Internal Server Error` — unexpected server error

---

### `GET /conversations/{conversation_id}`

Return the turns of a conversation. The response body is shaped by the route handler (it is not a generated Pydantic model):

**Response body (200 OK)**

```json
{
  "conversation_id": "550e8400-e29b-41d4-a716-446655440001",
  "turns": [
    {
      "user_message": "What is this about?",
      "assistant_message": "It's about RAG pipelines...",
      "turn_index": 0
    }
  ]
}
```

**Status codes**

- `200 OK` — history returned (may be an empty `turns` array)
- `400 Bad Request` — invalid `conversation_id` value
- `500 Internal Server Error` — unexpected server error

---

## Error format

All errors are returned as `{"detail": "..."}` JSON bodies, with the appropriate HTTP status code. The full set of statuses the API emits is:

| Status | When |
|--------|------|
| `400` | Bad request — invalid body, invalid `document_id` / `conversation_id`, source policy violation, or missing `BHODI_API_SOURCE_ROOT` |
| `404` | Document not found on `DELETE /documents/{document_id}` |
| `422` | Pydantic validation error (FastAPI default) |
| `429` | Rate limit exceeded (100 requests / 60s per IP, `/health` excluded) |
| `500` | Unhandled internal error |
| `503` | Health check reports a degraded adapter |

## Rate limiting

A simple in-memory limiter in `interfaces/api/app.py` enforces **100 requests per 60 seconds per client IP** on every route except `/health`. When the limit is exceeded the API returns `429 Too Many Requests` with `{"detail": "Rate limit exceeded. Try again later."}`.

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `BHODI_API_HOST` | `127.0.0.1` | API server bind host |
| `BHODI_API_PORT` | `8000` | API server bind port |
| `BHODI_API_SOURCE_ROOT` | unset | When set, constrains local file ingest for `POST /documents` to that directory |
| `OPENAI_API_KEY` | — | Required when any `openai` adapter is selected |

To configure the underlying adapters (LLM, embeddings, vector store, etc.), use the `BHODI_*_PROVIDER` environment variables or instantiate `BhodiConfig` directly and pass it to `create_app(config)` from `bhodi_platform.interfaces.api.app`.

## OpenAPI documentation

When the server is running, interactive documentation is available at:

- Swagger UI: <http://localhost:8000/docs>
- ReDoc: <http://localhost:8000/redoc>
- OpenAPI JSON: <http://localhost:8000/openapi.json>
