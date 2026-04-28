# Sentiment Analysis API

A lightweight REST API that stores labelled text in **Qdrant** as vector embeddings (via **Jina AI**) and predicts sentiment for new content using K-nearest-neighbour search.

---

## Architecture

```
Client ──► FastAPI ──► Jina AI (jina-embeddings-v5-text-small)
                  └──► Qdrant (cosine similarity, 1024-dim vectors)
```

| Component | Role |
|-----------|------|
| **FastAPI** | REST layer, request validation, CSV streaming |
| **Jina AI** | Embeds text → 1024-dim float32 vectors (base64-encoded, `classification` task for sentiment, `clustering` task for `/cluster-batch`) |
| **Qdrant** | Stores vectors + payload; deterministic UUIDs prevent duplicates |

---

## Quick Start

### 1 – Clone & configure

```bash
cp .env.example .env
# fill in JINA_API_KEY and optionally QDRANT_API_KEY / API_KEY
```

### 2a – Local (bare metal)

```bash
# Start Qdrant
docker run -p 6333:6333 qdrant/qdrant

# Install deps & run
pip install -r requirements.txt
uvicorn main:app --reload
```

### 2b – Docker Compose (recommended)

```bash
docker compose up --build
```

API will be available at `http://localhost:8000`.
Interactive docs: `http://localhost:8000/docs`

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `JINA_API_KEY` | *(required)* | Jina AI API key |
| `JINA_TIMEOUT` | `30` | Timeout (seconds) for Jina embedding calls |
| `QDRANT_URL` | `http://localhost:6333` | Qdrant server URL |
| `QDRANT_API_KEY` | *(empty)* | Qdrant API key (leave blank for local) |
| `COLLECTION_NAME` | `sentiments` | Qdrant collection name |
| `API_KEY` | *(empty)* | Set to require `X-API-Key` header on all endpoints (leave blank to disable auth) |
| `CORS_ORIGINS` | `*` | Comma-separated list of allowed CORS origins |
| `MAX_CONTENT_LEN` | `5000` | Max characters per text input |
| `MAX_TRAIN_ITEMS` | `256` | Max items per `/train` request |
| `MAX_IMPORT_ROWS` | `10000` | Max rows per CSV import |
| `MAX_UPLOAD_MB` | `10` | Max CSV file size in MB |
| `MAX_ANALYZE_BATCH` | `64` | Max items per `/analyze-batch` request |
| `MAX_CLUSTER_BATCH` | `16` | Max items per `/cluster-batch` request (per-text length follows `MAX_CONTENT_LEN`) |
| `LOG_LEVEL` | `INFO` | Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |

---

## Authentication

Authentication is **opt-in**. Set the `API_KEY` environment variable to any secret string.
When set, all endpoints (except `/health`) require an `X-API-Key` header:

```bash
curl -H "X-API-Key: your-secret" http://localhost:8000/analyze ...
```

Leave `API_KEY` empty to disable authentication entirely.

---

## API Reference

### `POST /train` – Store training data

```json
{
  "items": [
    { "content": "I love this product!", "sentiment": "positive" },
    { "content": "Terrible experience.",  "sentiment": "negative" },
    { "content": "Click here to subscribe", "sentiment": "ignore" }
  ]
}
```

**Response**

```json
{ "embedded": 2, "payload_updated": 0, "unchanged": 1 }
```

> Point IDs are deterministic UUID v5 values derived from the content string.
> **Smart dedup**: existing points are checked first. Unchanged rows are skipped, sentiment-only changes update the payload without calling Jina, and only new content is embedded. This dramatically reduces Jina API usage on repeated calls.
> Valid sentiments: `positive`, `negative`, `neutral`, `ignore`. Other values return HTTP 422.

---

### `POST /analyze` – Predict sentiment

```json
{ "content": "Shipping was super fast and packaging was great!" }
```

**Response**

```json
{
  "content": "Shipping was super fast ...",
  "sentiment": "positive",
  "confidence": { "positive": 0.812, "neutral": 0.188 },
  "matches": [
    { "point_id": "...", "content": "...", "sentiment": "positive", "score": 0.94 }
  ]
}
```

> Uses `query_points` to retrieve the **top-10** nearest neighbours by cosine similarity. Each neighbour's score is weighted by similarity and aggregated per sentiment class. Matches with a score **≥ 0.9** receive a **2× weight boost** to strengthen high-confidence predictions.

---

### `POST /analyze-batch` – Predict sentiment for multiple inputs

Embed **all** inputs in a single Jina round-trip, then query Qdrant via `query_batch_points` (one vector search per item, sent as a single server-side batch request). Returns results in the same order as the input list.

```json
{
  "items": [
    "Shipping was super fast!",
    "Total waste of money.",
    "Product arrived on time."
  ]
}
```

**Response**

```json
{
  "results": [
    {
      "content": "Shipping was super fast!",
      "sentiment": "positive",
      "confidence": { "positive": 0.91, "neutral": 0.09 },
      "matches": [ { "point_id": "...", "content": "...", "sentiment": "positive", "score": 0.96 } ]
    },
    {
      "content": "Total waste of money.",
      "sentiment": "negative",
      "confidence": { "negative": 1.0 },
      "matches": [ { "point_id": "...", "content": "...", "sentiment": "negative", "score": 0.93 } ]
    },
    {
      "content": "Product arrived on time.",
      "sentiment": "neutral",
      "confidence": { "neutral": 0.78, "positive": 0.22 },
      "matches": [ { "point_id": "...", "content": "...", "sentiment": "neutral", "score": 0.89 } ]
    }
  ]
}
```

> Max batch size is controlled by `MAX_ANALYZE_BATCH` (default `64`). Items with no neighbours return `"sentiment": null`.
> The same top-10 / 2× weight-boost logic as `/analyze` applies to each item in the batch.

---

### `POST /cluster-batch` – Get clustering embeddings

Embed multiple texts using the **`clustering` task** of `jina-embeddings-v5-text-small` and return the raw float32 vectors. Useful for downstream clustering algorithms (k-means, HDBSCAN, etc.) where distance-preserving representations are preferred over classification-optimised ones.

```json
{
  "items": [
    "The battery life is outstanding.",
    "Delivery was delayed by two weeks.",
    "Great value for the price."
  ]
}
```

**Response**

```json
{
  "results": [
    { "content": "The battery life is outstanding.", "embedding": [0.021, -0.043, "..."] },
    { "content": "Delivery was delayed by two weeks.", "embedding": [0.011, 0.067, "..."] },
    { "content": "Great value for the price.", "embedding": [-0.005, 0.031, "..."] }
  ]
}
```

> Each `embedding` is a 1024-element float32 array ready to feed into a clustering library.
> Max items: `MAX_CLUSTER_BATCH` (default **16**). Per-text length follows the global `MAX_CONTENT_LEN` (default **5000**). Auth required.
> Worst-case response size: 16 × 1024 floats ≈ **~130 KB**.

---

### `PATCH /points/{point_id}` – Update a point's payload

```json
{ "sentiment": "neutral" }
```

Pass `content`, `sentiment`, or both. If `content` changes the embedding is regenerated automatically.

**Response**

```json
{ "id": "<uuid>", "content": "...", "sentiment": "neutral" }
```

---

### `DELETE /points/{point_id}` – Delete a point

```bash
curl -X DELETE http://localhost:8000/points/<uuid>
```

**Response**

```json
{ "deleted": "<uuid>" }
```

> Returns HTTP 404 if the point does not exist, or HTTP 502 if the vector store is unreachable.

---

### `GET /export` – Export as CSV

Downloads `training_data.csv`:

```
Point,Content,Sentiment
<uuid>,I love this product!,positive
...
```

> Uses cursor-based scrolling (100 points per page) to stream all points.

---

### `POST /import` – Bulk import from CSV file

Upload a `.csv` file with columns `Content` and `Sentiment` (an optional `No` column is ignored).

```bash
curl -X POST http://localhost:8000/import \
  -F "file=@training_data.csv"
```

**CSV format:**

```csv
No,Content,Sentiment
1,I love this product!,positive
2,Terrible experience.,negative
3,Click here to subscribe,ignore
```

> Header matching is case-insensitive. Rows with empty content or sentiment are skipped.
> Rows exceeding `MAX_CONTENT_LEN` characters are also skipped.
> Files must be `.csv` and under `MAX_UPLOAD_MB` in size. Supports UTF-8 (with BOM) and Latin-1 encoding.
> **Smart dedup**: like `/train`, existing points are diffed first — only genuinely new texts hit Jina. Re-importing the same CSV is essentially a no-op.

**Response**

```json
{ "embedded": 2, "payload_updated": 1, "unchanged": 0, "skipped": 0 }
```

---

### `GET /health` – Health check

```json
{ "status": "ok", "collection": "sentiments", "points_count": 42 }
```

Returns `503` with `"status": "unhealthy"` if Qdrant is unreachable. This endpoint does **not** require authentication.

---

## Response Headers

Every response includes:

| Header | Description |
|--------|-------------|
| `X-Request-ID` | Unique request ID (pass your own via request header or auto-generated) |
| `X-Response-Time-Ms` | Server-side processing time in milliseconds |

---

## Embedding Details

Jina AI is called with the following payload. The `task` field varies by endpoint:

| Endpoint(s) | `task` value |
|---|---|
| `/train`, `/analyze`, `/analyze-batch`, `/import` | `"classification"` |
| `/cluster-batch` | `"clustering"` |

```json
{
  "model": "jina-embeddings-v5-text-small",
  "task": "classification",
  "truncate": true,
  "normalized": true,
  "embedding_type": "base64",
  "input": ["text1", "text2"]
}
```

The base64-encoded response is decoded and unpacked as 32-bit IEEE 754 floats:

```python
raw    = base64.b64decode(item["embedding"])
n      = len(raw) // 4
vector = list(struct.unpack(f"{n}f", raw))  # 1024 floats
```

---

## Deterministic Point IDs

```python
import uuid
point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, content))
```

This ensures that upserting the same content text always targets the same Qdrant point — no duplicate entries, safe to re-run training scripts.

---

## Content Cleansing

All content text is automatically normalised before embedding:

1. Carriage returns and newlines are replaced with spaces.
2. Zero-width characters (`\u200b`, `\u200c`, `\u200d`, `\ufeff`) are stripped.
3. Consecutive whitespace (spaces, tabs) is collapsed into a single space.
4. Leading and trailing whitespace is trimmed.

This ensures consistent embeddings regardless of how the source text is formatted.

---

## Sentiment Prediction Logic

1. Query vector is compared against all stored vectors using **cosine similarity** via Qdrant's `query_points` API.
2. **Top-10** nearest neighbours are retrieved.
3. A weighted vote is computed: each neighbour's similarity score contributes to its sentiment class.
4. Matches with a score **≥ 0.9** receive a **2× weight boost** to strengthen high-confidence predictions.
5. The class with the highest total weighted score is returned as the predicted sentiment.
6. Confidence values are normalised across all classes.

---

## Error Handling

| HTTP Code | Scenario |
|-----------|----------|
| `400` | Invalid CSV format, missing headers, or no valid rows |
| `401` | Missing or invalid API key (when `API_KEY` is configured) |
| `404` | Point not found, or no training data for analysis |
| `413` | CSV file exceeds size or row limit |
| `422` | Invalid sentiment value, empty content, or validation error |
| `502` | Jina AI or Qdrant upstream error |
| `503` | Jina API key not configured, or Qdrant unreachable |
| `504` | Jina embedding call timed out |

---

## Project Structure

```
sentiment-analysis/
├── main.py               # FastAPI application (single-file)
├── requirements.txt      # Python dependencies
├── Dockerfile            # Python 3.12-slim, non-root user
├── compose.yaml          # API + Qdrant with persistent volume
├── .env.example          # Environment variable template
├── .gitignore
├── .dockerignore
├── sample_training.csv   # Example training data (10 rows)
├── SYSTEM_PROMPT.txt     # LLM system prompt for API assistant
└── README.md             # This file
```

---

## Resources

- [Jina AI Embeddings API](https://jina.ai/embeddings/)
- [jina-embeddings-v5-text-small model card](https://huggingface.co/jinaai/jina-embeddings-v5-text-small)
- [Qdrant Documentation](https://qdrant.tech/documentation/)
- [Qdrant Python Client](https://github.com/qdrant/qdrant-client)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
