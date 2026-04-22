# Phase 1 — Complete Setup Guide

Multi-agent sales intelligence pipeline that processes European news, maps city problems to product features, and aligns with EU directives.

## Architecture

```
n8n (RSS) ──POST──▶ FastAPI webhook ──▶ Redis Stream ──▶ Worker ──▶ LangGraph ──▶ PostgreSQL
                     validates           queues           consumes    agents 1-4    stores results
                     deduplicates        buffers          retries     (placeholders) ↓
                                                                                   NextJS dashboard
                                                                                   (Phase 3)
```

## What each component does

**FastAPI webhook (main.py)** — The front door. Receives JSON from n8n, validates the payload against the NewsItem schema, checks for duplicate URLs using a Redis SET, then pushes the item to a Redis Stream. Returns 202 immediately — never waits for processing.

**Redis Stream (news:incoming)** — The queue. Buffers news items between the webhook and the worker. Provides consumer groups (so multiple workers can run in parallel), message acknowledgment, and automatic pending-message tracking for crash recovery.

**Worker (worker.py)** — The engine. Runs as a separate process. Uses XREADGROUP to efficiently wait for new items (zero CPU when idle). Picks up items, runs them through the LangGraph pipeline, saves results to PostgreSQL, then acknowledges the message in Redis. Handles retries (3 attempts) and dead-lettering for persistent failures.

**LangGraph pipeline (pipeline.py)** — The brain. A directed graph of 4 agent nodes with a conditional gate after Agent 1. Currently using placeholder agents that return hardcoded data. In Phase 2, these become real LLM calls.

**PostgreSQL (database.py)** — The memory. Stores every pipeline result with indexed columns for filtering (country, city, sector, status, urgency_score). This is what the NextJS dashboard will read from.


## Project structure

```
news-pipeline/
├── app/
│   ├── __init__.py
│   ├── config.py        # Environment config (pydantic-settings)
│   ├── models.py        # Data contracts: NewsItem, PipelineState
│   ├── main.py          # FastAPI webhook + health + stats endpoints
│   ├── worker.py        # Redis consumer → LangGraph → PostgreSQL
│   ├── pipeline.py      # LangGraph agent graph (4 agents + routing)
│   └── database.py      # PostgreSQL schema + save with UPSERT
├── tests/
│   ├── test_1_webhook.py    # Test webhook accepts/rejects correctly
│   ├── test_2_redis.py      # Verify Redis streams and consumer groups
│   ├── test_3_pipeline.py   # Verify LangGraph runs all agents
│   ├── test_4_database.py   # Verify PostgreSQL has correct data
│   └── test_5_end_to_end.py # Full pipeline: webhook → DB verification
├── docker-compose.yml   # Redis + PostgreSQL
├── requirements.txt
└── .env.example         # Config template
```


## Setup instructions

### 1. Start infrastructure

```bash
docker compose up -d
```

This starts Redis (port 6379) and PostgreSQL (port 5432).

**If PostgreSQL fails with a password error**, the volume has stale credentials from a previous run:

```bash
docker compose down -v    # -v removes volumes
docker compose up -d      # recreates with fresh credentials
```

### 2. Install Python dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Note:** redis==5.0.0 is pinned specifically because newer versions have a `decode_response` vs `decode_responses` parameter bug.

### 3. Configure environment

```bash
cp .env.example .env
```

Edit `.env` and set:
- `WEBHOOK_API_KEY` — generate with `python -c "import secrets; print(secrets.token_urlsafe(32))"`
- `DATABASE_URL` — must match your docker-compose PostgreSQL credentials
- Update the API_KEY in `tests/test_1_webhook.py` and `tests/test_5_end_to_end.py` to match

### 4. Start the webhook (Terminal 1)

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

You should see:
```
Connected to Redis at redis://localhost:6379/0
Consumer group 'pipeline-workers' ready on stream 'news:incoming'
```

### 5. Start the worker (Terminal 2)

```bash
python -m app.worker
```

You should see:
```
Connected to Redis
Consumer group 'pipeline-workers' ready
Database tables ready
Pipeline graph compiled: classify → [analyze → solve → write_case] → END
Worker 'worker-1' listening (concurrency=2)
```

### 6. Run tests (Terminal 3)

Run them in order:

```bash
# Test 1: Webhook accepts and rejects correctly
python tests/test_1_webhook.py

# Test 2: Redis streams are properly configured
python tests/test_2_redis.py

# Test 3: LangGraph pipeline processes items correctly
python tests/test_3_pipeline.py

# Test 4: PostgreSQL has correct data (run after worker processes items)
python tests/test_4_database.py

# Test 5: End-to-end (sends item, waits for processing, verifies in DB)
python tests/test_5_end_to_end.py
```

### 7. Connect n8n

In your n8n workflow, add an HTTP Request node at the end:
- **URL:** `http://your-server:8000/webhook/news`
- **Method:** POST
- **Headers:** `X-API-Key: <your-key>`
- **Body:** Map RSS fields to the NewsItem schema:

```json
{
  "title": "{{ $json.title }}",
  "summary": "{{ $json.description }}",
  "source_url": "{{ $json.link }}",
  "published_at": "{{ $json.pubDate }}"
}
```

Country, city, tags, and raw_content are all optional.


## Gotchas we discovered (and fixed)

These are real issues we hit during development. They're all fixed in the code, but documented here for reference.

### 1. model_dump_json() needs parentheses
**File:** main.py, worker.py
**Symptom:** `Invalid input of type: 'method'`
**Cause:** `state.model_dump_json` passes the method object. `state.model_dump_json()` calls it and returns a string.

### 2. Consumer group name must match everywhere
**File:** config.py
**Symptom:** Worker runs but never picks up items
**Cause:** Webhook created group "pipeline-wokers" (typo), worker read from "pipeline-workers". Different groups = different message tracking.
**Debug:** `XINFO GROUPS news:incoming` shows all groups.

### 3. Pydantic HttpUrl is not a plain string
**File:** database.py, worker.py
**Symptom:** `expected str, got HttpUrl`
**Cause:** asyncpg doesn't know how to serialize Pydantic's HttpUrl type.
**Fix:** Wrap in `str()` everywhere it touches PostgreSQL.

### 4. Datetime strings vs datetime objects
**File:** database.py
**Symptom:** `expected a datetime.date or datetime.datetime instance, got 'str'`
**Cause:** `state_to_graph_input()` converts datetimes to ISO strings for LangGraph (which needs JSON-serializable state). But `save_result()` passes those strings to asyncpg, which expects real datetime objects.
**Fix:** `parse_dt()` helper that converts ISO strings back to datetime.

### 5. JSONB columns receiving string "null"
**File:** database.py
**Symptom:** JSONB insert fails or stores the string "null" instead of SQL NULL
**Cause:** Items that skip agents have None values that get JSON-serialized as the literal string "null".
**Fix:** `clean_jsonb()` helper that converts "null" strings to Python None.

### 6. Duplicate primary key on reprocessing
**File:** database.py
**Symptom:** `duplicate key value violates unique constraint "pipeline_results_pkey"`
**Cause:** Resetting the consumer group replays all messages. Items that were already saved to PostgreSQL get processed again, and INSERT fails because the pipeline_id already exists.
**Fix:** UPSERT via `on_conflict_do_update` — overwrites the existing row instead of crashing.

### 7. graph.ainvoke() not graph.run()
**File:** worker.py
**Symptom:** `'CompiledStateGraph' object has no attribute 'run'`
**Cause:** LangGraph's compiled graph uses `.ainvoke()` for async execution, not `.run()`.

### 8. Country/city are optional from RSS
**File:** models.py, main.py
**Symptom:** `'NewsItem' object has no attribute 'country'` or validation errors
**Cause:** Most RSS feeds don't provide country or city. These must be optional fields that Agent 1 extracts.


## Useful commands

### Check Redis state
```bash
# Items in queue
docker exec -it news-pipeline-redis-1 redis-cli XLEN news:incoming

# Consumer group status
docker exec -it news-pipeline-redis-1 redis-cli XINFO GROUPS news:incoming

# View recent items
docker exec -it news-pipeline-redis-1 redis-cli XREVRANGE news:incoming + - COUNT 3

# Dead letter queue
docker exec -it news-pipeline-redis-1 redis-cli XLEN news:dead_letter

# Reset consumer group (reprocess all items)
docker exec -it news-pipeline-redis-1 redis-cli XGROUP SETID news:incoming pipeline-workers 0
```

### Check PostgreSQL state
```bash
# Row count by status
docker exec -it news-pipeline-postgres-1 psql -U postgres -d news_pipeline \
  -c "SELECT status, COUNT(*) FROM pipeline_results GROUP BY status;"

# Recent completed items
docker exec -it news-pipeline-postgres-1 psql -U postgres -d news_pipeline \
  -c "SELECT pipeline_id, country, sector, title FROM pipeline_results WHERE status='completed' ORDER BY completed_at DESC LIMIT 5;"

# Full reset (drop all data)
docker exec -it news-pipeline-postgres-1 psql -U postgres -d news_pipeline \
  -c "TRUNCATE pipeline_results;"
```


## Next steps (Phase 2)

1. **Agent 1** — Wire to Ollama for real classification and country/city extraction
2. **Agent 2** — Wire to Claude API for problem analysis
3. **Agent 3** — Wire to Claude API + Qdrant RAG for product/directive matching
4. **Agent 4** — Wire to Claude API for business case generation
5. **Phase 3** — NextJS dashboard reading from PostgreSQL