import os
import re
import uuid
import base64
import struct
import csv
import io
import time
import logging
from typing import Optional, Annotated
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile, File, Request, Security, Depends
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
import httpx
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
  Distance, VectorParams, PointStruct,
  Filter, FieldCondition, MatchValue,
  QueryRequest,
)

API_NAME = os.getenv("API_NAME", "Sentiment Analysis API")

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper())
logger = logging.getLogger("sentiment-api")

# ── Config ────────────────────────────────────────────────────────────────────
JINA_API_KEY      = os.getenv("JINA_API_KEY", "")
QDRANT_URL        = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY    = os.getenv("QDRANT_API_KEY", "")
COLLECTION_NAME   = os.getenv("COLLECTION_NAME", "sentiments")
VECTOR_SIZE       = 1024  # jina-embeddings-v5-text-small output dim

API_KEY           = os.getenv("API_KEY", "")              # leave empty to disable auth
CORS_ORIGINS      = os.getenv("CORS_ORIGINS", "*")        # comma-separated
MAX_CONTENT_LEN   = int(os.getenv("MAX_CONTENT_LEN", "5000"))  # chars per text
MAX_TRAIN_ITEMS   = int(os.getenv("MAX_TRAIN_ITEMS", "256"))    # items per /train call
MAX_IMPORT_ROWS   = int(os.getenv("MAX_IMPORT_ROWS", "10000")) # rows per CSV import
MAX_UPLOAD_MB     = int(os.getenv("MAX_UPLOAD_MB", "10"))       # CSV file size cap
MAX_ANALYZE_BATCH = int(os.getenv("MAX_ANALYZE_BATCH", "64"))   # items per /analyze-batch call

JINA_EMBED_URL    = "https://api.jina.ai/v1/embeddings"
JINA_TIMEOUT      = int(os.getenv("JINA_TIMEOUT", "30"))       # seconds

VALID_SENTIMENTS  = {"positive", "negative", "neutral", "ignore"}

# ── Shared resources (initialised in lifespan) ────────────────────────────────
http_client: httpx.AsyncClient | None = None
qdrant: AsyncQdrantClient | None = None


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
  global http_client, qdrant

  # Startup
  http_client = httpx.AsyncClient(
    timeout=httpx.Timeout(JINA_TIMEOUT, connect=10),
    limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
  )
  qdrant = AsyncQdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY or None,
    timeout=10,
  )
  await ensure_collection()
  logger.info("Sentiment API started  ✓")

  yield

  # Shutdown
  await http_client.aclose()
  await qdrant.close()
  logger.info("Sentiment API shut down ✓")


app = FastAPI(
  title=API_NAME,
  version="1.0.0",
  lifespan=lifespan,
)

# ── Middleware: CORS ──────────────────────────────────────────────────────────
origins = [o.strip() for o in CORS_ORIGINS.split(",") if o.strip()]
app.add_middleware(
  CORSMiddleware,
  allow_origins=origins,
  allow_credentials=True,
  allow_methods=["*"],
  allow_headers=["*"],
)


# ── Middleware: Request ID + Timing ───────────────────────────────────────────

@app.middleware("http")
async def request_meta_middleware(request: Request, call_next):
  request_id = request.headers.get("X-Request-ID", uuid.uuid4().hex[:12])
  start = time.perf_counter()
  response = await call_next(request)
  elapsed_ms = (time.perf_counter() - start) * 1000
  response.headers["X-Request-ID"] = request_id
  response.headers["X-Response-Time-Ms"] = f"{elapsed_ms:.1f}"
  logger.info(
    "%s %s → %s  (%.0f ms)",
    request.method, request.url.path, response.status_code, elapsed_ms,
  )
  return response


# ── Auth (opt-in) ─────────────────────────────────────────────────────────────
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(key: Optional[str] = Security(api_key_header)):
  """If API_KEY env is set, every request must provide a matching X-API-Key header."""
  if not API_KEY:
    return  # auth disabled
  if key != API_KEY:
    raise HTTPException(status_code=401, detail="Invalid or missing API key.")


# ── Helpers ───────────────────────────────────────────────────────────────────

def cleanse_text(text: str) -> str:
  """Normalise whitespace: collapse runs of spaces/tabs/newlines into a single space and strip."""
  text = text.replace("\r\n", " ").replace("\r", " ").replace("\n", " ")
  text = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", text)   # zero-width chars
  text = re.sub(r"\s+", " ", text)                          # collapse whitespace
  return text.strip()


def validate_sentiment(value: str) -> str:
  """Lower-case and validate sentiment against the allowed set."""
  value = value.lower().strip()
  if value not in VALID_SENTIMENTS:
    raise HTTPException(
      status_code=422,
      detail=f"Invalid sentiment '{value}'. Must be one of: {', '.join(sorted(VALID_SENTIMENTS))}",
    )
  return value


def deterministic_uuid(content: str) -> str:
  """Generate a stable UUID v5 from content text."""
  return str(uuid.uuid5(uuid.NAMESPACE_DNS, content))


async def embed(texts: list[str]) -> list[list[float]]:
  """Call Jina AI and decode base64 → float32 vectors.  Uses the shared httpx client."""
  if not JINA_API_KEY:
    raise HTTPException(status_code=503, detail="JINA_API_KEY is not configured.")

  headers = {
    "Authorization": f"Bearer {JINA_API_KEY}",
    "Content-Type": "application/json",
  }
  payload = {
    "model": "jina-embeddings-v5-text-small",
    "task": "classification",
    "truncate": True,
    "normalized": True,
    "embedding_type": "base64",
    "input": texts,
  }
  try:
    resp = await http_client.post(JINA_EMBED_URL, json=payload, headers=headers)
    resp.raise_for_status()
  except httpx.TimeoutException:
    raise HTTPException(status_code=504, detail="Embedding service timed out.")
  except httpx.HTTPStatusError as exc:
    logger.error("Jina API error: %s %s", exc.response.status_code, exc.response.text[:200])
    raise HTTPException(
      status_code=502,
      detail=f"Embedding service returned {exc.response.status_code}.",
    )
  except httpx.HTTPError as exc:
    logger.error("Jina connection error: %s", exc)
    raise HTTPException(status_code=502, detail="Could not reach embedding service.")

  data = resp.json()["data"]
  vectors = []
  for item in sorted(data, key=lambda x: x["index"]):
    raw = base64.b64decode(item["embedding"])
    n = len(raw) // 4  # 4 bytes per float32
    vector = list(struct.unpack(f"{n}f", raw))
    vectors.append(vector)
  return vectors


async def ensure_collection():
  """Create collection if it doesn't exist yet."""
  existing = [c.name for c in (await qdrant.get_collections()).collections]
  if COLLECTION_NAME not in existing:
    await qdrant.create_collection(
      collection_name=COLLECTION_NAME,
      vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )
    logger.info("Created Qdrant collection '%s'", COLLECTION_NAME)


# ── Schemas ───────────────────────────────────────────────────────────────────

class TrainingItem(BaseModel):
  content: str = Field(..., min_length=1, max_length=MAX_CONTENT_LEN)
  sentiment: str = Field(..., min_length=1)

class TrainingRequest(BaseModel):
  items: list[TrainingItem] = Field(..., min_length=1, max_length=MAX_TRAIN_ITEMS)

class AnalyzeRequest(BaseModel):
  content: str = Field(..., min_length=1, max_length=MAX_CONTENT_LEN)

class AnalyzeBatchRequest(BaseModel):
  items: list[Annotated[str, Field(min_length=1, max_length=MAX_CONTENT_LEN)]] = Field(..., min_length=1, max_length=MAX_ANALYZE_BATCH, description=f"List of content strings to analyse (max {MAX_ANALYZE_BATCH} items, each up to {MAX_CONTENT_LEN} chars).")

class UpdateRequest(BaseModel):
  content: Optional[str] = Field(None, max_length=MAX_CONTENT_LEN)
  sentiment: Optional[str] = None

  @field_validator("content", "sentiment", mode="before")
  @classmethod
  def reject_empty_strings(cls, v):
    if v is not None and isinstance(v, str) and not v.strip():
      raise ValueError("Field must not be blank.")
    return v


# ── Routes ────────────────────────────────────────────────────────────────────

@app.post("/train", summary="Store training data into Qdrant", dependencies=[Depends(verify_api_key)])
async def train(req: TrainingRequest):
  """
  Embed each item with Jina AI and upsert into Qdrant.
  Point IDs are deterministic UUIDs derived from the content text.

  Optimization: existing points are checked first.
  - Identical content + sentiment → skip (no Jina call).
  - Same content, different sentiment → payload-only update (no Jina call).
  - New or changed content → embed via Jina and upsert.
  """
  texts      = [cleanse_text(item.content) for item in req.items]
  sentiments = [validate_sentiment(item.sentiment) for item in req.items]
  ids        = [deterministic_uuid(t) for t in texts]

  # Batch-retrieve existing points
  existing_map: dict[str, dict] = {}
  retrieve_batch = 256
  for i in range(0, len(ids), retrieve_batch):
    try:
      results = await qdrant.retrieve(
        collection_name=COLLECTION_NAME,
        ids=ids[i : i + retrieve_batch],
        with_payload=True,
        with_vectors=False,
      )
      for pt in results:
        existing_map[pt.id] = pt.payload
    except Exception as exc:
      logger.warning("Qdrant retrieve during train: %s (will embed all)", exc)
      existing_map = {}
      break

  # Categorise
  new_indices: list[int] = []
  payload_update_points: list[PointStruct] = []
  unchanged = 0

  for idx, (pid, text, sent) in enumerate(zip(ids, texts, sentiments)):
    existing = existing_map.get(pid)
    if existing is not None:
      if existing.get("content") == text and existing.get("sentiment") == sent:
        unchanged += 1
        continue
      if existing.get("content") == text:
        payload_update_points.append(
          PointStruct(id=pid, vector=[0.0], payload={"content": text, "sentiment": sent})
        )
        continue
    new_indices.append(idx)

  # Payload-only updates
  payload_updated = 0
  for pt in payload_update_points:
    await qdrant.set_payload(
      collection_name=COLLECTION_NAME,
      payload=pt.payload,
      points=[pt.id],
    )
    payload_updated += 1

  # Embed only new / changed-content items
  embedded = 0
  if new_indices:
    new_texts = [texts[i] for i in new_indices]
    new_sents = [sentiments[i] for i in new_indices]
    new_ids   = [ids[i] for i in new_indices]

    all_vectors: list[list[float]] = []
    batch_size = 64
    for i in range(0, len(new_texts), batch_size):
      all_vectors.extend(await embed(new_texts[i : i + batch_size]))

    points = [
      PointStruct(
        id=pid, vector=vec,
        payload={"content": text, "sentiment": sent},
      )
      for pid, text, sent, vec in zip(new_ids, new_texts, new_sents, all_vectors)
    ]

    await qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
    embedded = len(points)

  return {
    "embedded": embedded,
    "payload_updated": payload_updated,
    "unchanged": unchanged,
  }


@app.post("/analyze", summary="Predict sentiment from training data", dependencies=[Depends(verify_api_key)])
async def analyze(req: AnalyzeRequest):
  """
  Embed the input, search Qdrant for nearest neighbours, then
  return a majority-vote sentiment with confidence breakdown.
  """
  content = cleanse_text(req.content)
  if not content:
    raise HTTPException(status_code=422, detail="Content is empty after cleansing.")

  vectors = await embed([content])

  try:
    response = await qdrant.query_points(
      collection_name=COLLECTION_NAME,
      query=vectors[0],
      limit=5,
      with_payload=True,
    )
  except Exception as exc:
    logger.error("Qdrant query error: %s", exc)
    raise HTTPException(status_code=502, detail="Vector search failed.")

  if not response.points:
    raise HTTPException(status_code=404, detail="No training data found.")

  votes: dict[str, float] = {}
  matches = []
  for hit in response.points:
    sent = hit.payload["sentiment"]
    votes[sent] = votes.get(sent, 0.0) + hit.score
    matches.append({
      "point_id":  hit.id,
      "content":   hit.payload["content"],
      "sentiment": sent,
      "score":     round(hit.score, 4),
    })

  total = sum(votes.values())
  sentiment = max(votes, key=votes.__getitem__)
  confidence = {k: round(v / total, 4) for k, v in votes.items()}

  return {
    "content":    content,
    "sentiment":  sentiment,
    "confidence": confidence,
    "matches":    matches,
  }


@app.post("/analyze-batch", summary="Predict sentiment for multiple inputs in one call", dependencies=[Depends(verify_api_key)])
async def analyze_batch(req: AnalyzeBatchRequest):
  """
  Embed all inputs in a single Jina call, then fan out to Qdrant via
  query_batch_points (one search per input, executed server-side in one
  round-trip).  Returns a result object for every input, preserving order.
  """
  # Cleanse all inputs up front
  contents = [cleanse_text(c) for c in req.items]
  empty = [i for i, c in enumerate(contents) if not c]
  if empty:
    raise HTTPException(
      status_code=422,
      detail=f"Items at index {empty} are empty after cleansing.",
    )

  # Single Jina round-trip for all items
  vectors = await embed(contents)

  # Build one query per input
  queries = [
    QueryRequest(query=vec, limit=5, with_payload=True)
    for vec in vectors
  ]

  try:
    batch_response = await qdrant.query_batch_points(
      collection_name=COLLECTION_NAME,
      requests=queries,
    )
  except Exception as exc:
    logger.error("Qdrant batch query error: %s", exc)
    raise HTTPException(status_code=502, detail="Vector batch search failed.")

  results = []
  for content, scored_points in zip(contents, batch_response):
    points = scored_points.points
    if not points:
      results.append({
        "content":    content,
        "sentiment":  None,
        "confidence": {},
        "matches":    [],
      })
      continue

    votes: dict[str, float] = {}
    matches = []
    for hit in points:
      sent = hit.payload["sentiment"]
      votes[sent] = votes.get(sent, 0.0) + hit.score
      matches.append({
        "point_id":  hit.id,
        "content":   hit.payload["content"],
        "sentiment": sent,
        "score":     round(hit.score, 4),
      })

    total = sum(votes.values())
    sentiment = max(votes, key=votes.__getitem__)
    confidence = {k: round(v / total, 4) for k, v in votes.items()}

    results.append({
      "content":    content,
      "sentiment":  sentiment,
      "confidence": confidence,
      "matches":    matches,
    })

  return {"results": results}


@app.patch("/points/{point_id}", summary="Update payload of a point", dependencies=[Depends(verify_api_key)])
async def update_point(point_id: str, req: UpdateRequest):
  """
  Update content and/or sentiment payload for an existing point.
  If content changes the embedding is re-generated.
  """
  if req.content is None and req.sentiment is None:
    raise HTTPException(status_code=422, detail="Provide at least one of 'content' or 'sentiment'.")

  try:
    results = await qdrant.retrieve(
      collection_name=COLLECTION_NAME,
      ids=[point_id],
      with_payload=True,
      with_vectors=True,
    )
  except Exception as exc:
    logger.error("Qdrant retrieve error: %s", exc)
    raise HTTPException(status_code=502, detail="Vector store lookup failed.")

  if not results:
    raise HTTPException(status_code=404, detail="Point not found.")

  point = results[0]
  new_content   = cleanse_text(req.content) if req.content is not None else point.payload["content"]
  new_sentiment = validate_sentiment(req.sentiment) if req.sentiment is not None else point.payload["sentiment"]

  # Re-embed only when content changed
  if req.content is not None and new_content != point.payload["content"]:
    vectors = await embed([new_content])
    new_vector = vectors[0]
  else:
    new_vector = point.vector

  await qdrant.upsert(
    collection_name=COLLECTION_NAME,
    points=[PointStruct(
      id=point_id,
      vector=new_vector,
      payload={"content": new_content, "sentiment": new_sentiment},
    )],
  )
  return {"id": point_id, "content": new_content, "sentiment": new_sentiment}


@app.get("/export", summary="Export all training data as CSV", dependencies=[Depends(verify_api_key)])
async def export_csv():
  """
  Stream all points as a CSV with columns: Point, Content, Sentiment.
  """
  output = io.StringIO()
  writer = csv.writer(output)
  writer.writerow(["Point", "Content", "Sentiment"])

  offset = None
  limit  = 100
  while True:
    scroll_result = await qdrant.scroll(
      collection_name=COLLECTION_NAME,
      offset=offset,
      limit=limit,
      with_payload=True,
      with_vectors=False,
    )
    points, next_offset = scroll_result
    for point in points:
      writer.writerow([
        point.id,
        point.payload.get("content", ""),
        point.payload.get("sentiment", ""),
      ])
    if next_offset is None:
      break
    offset = next_offset

  output.seek(0)
  return StreamingResponse(
    iter([output.getvalue()]),
    media_type="text/csv",
    headers={"Content-Disposition": "attachment; filename=training_data.csv"},
  )


@app.post("/import", summary="Import training data from a CSV file", dependencies=[Depends(verify_api_key)])
async def import_csv(file: UploadFile = File(...)):
  """
  Upload a CSV file with columns: Content, Sentiment.
  An optional 'No' column is ignored if present.
  IDs are deterministic UUIDs derived from content text.

  Optimization: existing points in Qdrant are looked up first.
  - If content AND sentiment are unchanged → skip entirely (no Jina call).
  - If only sentiment changed → payload-only update (no Jina call).
  - Only genuinely new texts are sent to the embedding API.
  """
  if not file.filename or not file.filename.lower().endswith(".csv"):
    raise HTTPException(status_code=400, detail="Only .csv files are accepted.")

  raw = await file.read()

  # Guard against oversized uploads
  if len(raw) > MAX_UPLOAD_MB * 1024 * 1024:
    raise HTTPException(
      status_code=413,
      detail=f"File exceeds {MAX_UPLOAD_MB} MB limit.",
    )

  try:
    text_content = raw.decode("utf-8-sig")  # handles BOM from Excel
  except UnicodeDecodeError:
    text_content = raw.decode("latin-1")

  reader = csv.DictReader(io.StringIO(text_content))

  # Normalise header names to lowercase for flexible matching
  if not reader.fieldnames:
    raise HTTPException(status_code=400, detail="CSV file is empty or has no header row.")

  header_map = {h.strip().lower(): h for h in reader.fieldnames}
  if "content" not in header_map or "sentiment" not in header_map:
    raise HTTPException(
      status_code=400,
      detail=f"CSV must have 'Content' and 'Sentiment' columns. Found: {reader.fieldnames}",
    )

  texts: list[str] = []
  sentiments: list[str] = []
  skipped = 0

  for row_num, row in enumerate(reader, start=2):  # start=2 because row 1 is header
    if len(texts) >= MAX_IMPORT_ROWS:
      raise HTTPException(
        status_code=413,
        detail=f"CSV exceeds the maximum of {MAX_IMPORT_ROWS} rows.",
      )
    content_val = (row.get(header_map["content"]) or "").strip()
    sentiment_val = (row.get(header_map["sentiment"]) or "").strip()

    if not content_val or not sentiment_val:
      skipped += 1
      continue

    if len(content_val) > MAX_CONTENT_LEN:
      skipped += 1
      continue

    texts.append(cleanse_text(content_val))
    sentiments.append(validate_sentiment(sentiment_val))

  if not texts:
    raise HTTPException(status_code=400, detail="No valid rows found in CSV.")

  # ── Diff against Qdrant: skip unchanged, payload-update changed, embed new ──
  ids = [deterministic_uuid(t) for t in texts]

  # Batch-retrieve existing points (Qdrant caps per-call; chunk at 256)
  existing_map: dict[str, dict] = {}  # id → payload
  retrieve_batch = 256
  for i in range(0, len(ids), retrieve_batch):
    try:
      results = await qdrant.retrieve(
        collection_name=COLLECTION_NAME,
        ids=ids[i : i + retrieve_batch],
        with_payload=True,
        with_vectors=False,
      )
      for pt in results:
        existing_map[pt.id] = pt.payload
    except Exception as exc:
      logger.warning("Qdrant retrieve during import: %s (will embed all)", exc)
      existing_map = {}  # fall back to embedding everything
      break

  # Categorise each row
  new_indices: list[int] = []           # need Jina embedding
  payload_update_points: list[PointStruct] = []  # sentiment-only change
  unchanged = 0

  for idx, (pid, text, sent) in enumerate(zip(ids, texts, sentiments)):
    existing = existing_map.get(pid)
    if existing is not None:
      if existing.get("content") == text and existing.get("sentiment") == sent:
        unchanged += 1
        continue  # identical → skip entirely
      if existing.get("content") == text:
        # Same content, different sentiment → reuse existing vector
        payload_update_points.append(
          PointStruct(id=pid, vector=[0.0], payload={"content": text, "sentiment": sent})
        )
        continue
    # New or changed content → needs embedding
    new_indices.append(idx)

  # Payload-only updates (use set_payload so we don't touch the vector)
  payload_updated = 0
  sp_batch = 256
  for i in range(0, len(payload_update_points), sp_batch):
    batch = payload_update_points[i : i + sp_batch]
    for pt in batch:
      await qdrant.set_payload(
        collection_name=COLLECTION_NAME,
        payload=pt.payload,
        points=[pt.id],
      )
    payload_updated += len(batch)

  # Embed only the truly new texts
  embedded = 0
  if new_indices:
    new_texts = [texts[i] for i in new_indices]
    new_sents = [sentiments[i] for i in new_indices]
    new_ids   = [ids[i] for i in new_indices]

    all_vectors: list[list[float]] = []
    batch_size = 64
    for i in range(0, len(new_texts), batch_size):
      all_vectors.extend(await embed(new_texts[i : i + batch_size]))

    points = [
      PointStruct(
        id=pid, vector=vec,
        payload={"content": text, "sentiment": sent},
      )
      for pid, text, sent, vec in zip(new_ids, new_texts, new_sents, all_vectors)
    ]

    upsert_batch = 256
    for i in range(0, len(points), upsert_batch):
      await qdrant.upsert(collection_name=COLLECTION_NAME, points=points[i : i + upsert_batch])
    embedded = len(points)

  return {
    "embedded": embedded,
    "payload_updated": payload_updated,
    "unchanged": unchanged,
    "skipped": skipped,
  }


@app.delete("/points/{point_id}", summary="Delete a point", dependencies=[Depends(verify_api_key)])
async def delete_point(point_id: str):
  """Delete a single point by ID."""
  try:
    results = await qdrant.retrieve(
      collection_name=COLLECTION_NAME,
      ids=[point_id],
      with_payload=False,
      with_vectors=False,
    )
  except Exception as exc:
    logger.error("Qdrant retrieve error: %s", exc)
    raise HTTPException(status_code=502, detail="Vector store lookup failed.")

  if not results:
    raise HTTPException(status_code=404, detail="Point not found.")

  await qdrant.delete(
    collection_name=COLLECTION_NAME,
    points_selector=[point_id],
  )
  return {"deleted": point_id}


@app.get("/health", summary="Health check")
async def health():
  try:
    info = await qdrant.get_collection(COLLECTION_NAME)
  except Exception as exc:
    logger.error("Health check failed: %s", exc)
    return JSONResponse(
      status_code=503,
      content={"status": "unhealthy", "detail": str(exc)},
    )
  return {
    "status": "ok",
    "collection": COLLECTION_NAME,
    "points_count": info.points_count,
  }
