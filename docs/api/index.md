# Bhodi API Reference

## Base URL

```
http://localhost:8000
```

## Overview

Bhodi exposes a REST API for indexing documents and querying answers. The API is built with FastAPI and provides automatic OpenAPI documentation.

## Authentication

Currently no authentication is required. Future versions will support API key authentication.

## Endpoints

### Health Check

Check if the service is running and healthy.

```
GET /health
```

**Response:**

```json
{
  "status": "healthy",
  "version": "1.0.0"
}
```

**Status Codes:**
- `200 OK` - Service is healthy

---

### Index Document

Upload and index a document for later querying.

```
POST /documents
```

**Request Body:**

```json
{
  "source": "path/to/document.pdf",
  "metadata": {
    "author": "John Doe",
    "date": "2024-01-15"
  },
  "chunk_size": 512,
  "overlap": 64
}
```

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| source | string | Yes | - | Path to document (local file path) |
| metadata | object | No | {} | Arbitrary metadata attached to document |
| chunk_size | integer | No | From config | Target chunk size |
| overlap | integer | No | From config | Overlap between chunks |

**Response:**

```json
{
  "document_id": "550e8400-e29b-41d4-a716-446655440000",
  "chunk_count": 42
}
```

**Status Codes:**
- `201 Created` - Document indexed successfully
- `400 Bad Request` - Invalid request (e.g., file not found)
- `500 Internal Server Error` - Server error

---

### Delete Document

Remove a document from the index.

```
DELETE /documents/{document_id}
```

**Path Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| document_id | string | UUID of the document to delete |

**Response:**

```json
{
  "deleted": true,
  "document_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

**Status Codes:**
- `200 OK` - Document deleted
- `404 Not Found` - Document not found
- `500 Internal Server Error` - Server error

---

### Query

Query the indexed documents and get an answer with citations.

```
POST /query
```

**Request Body:**

```json
{
  "question": "What is the main topic?",
  "conversation_id": "550e8400-e29b-41d4-a716-446655440001",
  "top_k": 5,
  "temperature": 0.7
}
```

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| question | string | Yes | - | The query question |
| conversation_id | string | No | - | Session ID for conversation continuity |
| top_k | integer | No | 5 | Number of chunks to retrieve |
| temperature | float | No | 0.7 | LLM temperature (0.0-2.0) |

**Response:**

```json
{
  "answer_text": "The main topic of the document is...",
  "citations": [
    {
      "chunk_id": "550e8400-e29b-41d4-a716-446655440000:0",
      "text": "First paragraph of the document...",
      "source_document": "550e8400-e29b-41d4-a716-446655440000",
      "page": 1
    }
  ],
  "conversation_id": "550e8400-e29b-41d4-a716-446655440001"
}
```

**Status Codes:**
- `200 OK` - Query successful
- `400 Bad Request` - Invalid request
- `500 Internal Server Error` - Server error

---

### Get Conversation History

Retrieve the history of a conversation session.

```
GET /conversations/{conversation_id}
```

**Path Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| conversation_id | string | UUID of the conversation |

**Response:**

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

**Status Codes:**
- `200 OK` - History retrieved
- `404 Not Found` - Conversation not found
- `500 Internal Server Error` - Server error

---

## Error Responses

All errors follow a consistent format:

```json
{
  "detail": "Error description here"
}
```

| Status Code | Description |
|-------------|--------------|
| 400 | Bad Request - Invalid input or file not found |
| 404 | Not Found - Document or conversation doesn't exist |
| 422 | Validation Error - Pydantic validation failed |
| 500 | Internal Server Error - Unexpected server error |

---

## Configuration

The API server can be configured via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| BHODI_API_HOST | 0.0.0.0 | Server host |
| BHODI_API_PORT | 8000 | Server port |
| OPENAI_API_KEY | - | OpenAI API key (required for OpenAI adapters) |

Or programmatically:

```python
from bhodi_platform.application.config import BhodiConfig
from bhodi_platform.interfaces.api.app import create_app

config = BhodiConfig(
    embedding={"provider": "openai", "model": "text-embedding-3-small"},
    # ... other config
)

app = create_app(config)
```

---

## OpenAPI Documentation

When the server is running, interactive API documentation is available at:

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
- OpenAPI JSON: http://localhost:8000/openapi.json
