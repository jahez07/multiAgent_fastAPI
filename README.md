# News Intelligence Pipeline

A multi-agent sales intelligence system that processes European infrastructure news, identifies city problems, maps them to product features, and aligns solutions with EU regulatory directives.

## What it does

The pipeline ingests news articles (via n8n RSS feeds), runs them through four AI agents, and produces structured sales intelligence: which products to pitch, to whom, and how they comply with EU law.

```
n8n (RSS) ──▶ FastAPI webhook ──▶ Redis Stream ──▶ Worker ──▶ LangGraph ──▶ PostgreSQL
              validates            queues           consumes    4 agents       stores results
              deduplicates         buffers          retries
```

## Agent pipeline

```
START
  │
  ▼
Agent 1: Classifier (Ollama / Llama 3.1:8b — local GPU)
  │  Extracts country, city, sector. Scores relevance 0.0–1.0.
  │
  ├── relevance ≥ 0.4 ──▶ Agent 2: Problem Analyzer (Claude Sonnet)
  │                         │  Identifies root causes, stakeholders,
  │                         │  urgency and opportunity scores.
  │                         │
  │                         ▼
  │                       Agent 3: Solution Analyst (Claude Sonnet + Qdrant RAG)
  │                         │  Searches product catalog and EU directive docs.
  │                         │  Maps product features → problems → directives.
  │                         │
  │                         ▼
  │                       Agent 4: Business Case Writer (Claude Sonnet)
  │                         │  Synthesizes a pitch document for city officials.
  │                         ▼
  │                        END
  │
  └── relevance < 0.4 ──▶ END (skipped — no Claude API calls wasted)
```

Agent 1 filters ~80% of irrelevant articles using a cheap local model before any Claude API calls are made.

## Architecture

| Component | File | Role |
|---|---|---|
| FastAPI webhook | `app/main.py` | Receives news from n8n, deduplicates, queues to Redis Stream |
| Worker | `app/worker.py` | Consumes from Redis, runs LangGraph, saves to PostgreSQL |
| Pipeline | `app/pipeline.py` | LangGraph graph with 4 agent nodes and conditional routing |
| Classifier | `app/agents/classifier.py` | Agent 1 — Ollama for relevance filtering |
| Analyzer | `app/agents/analyzer.py` | Agent 2 — Claude for problem analysis |
| Solution Analyst | `app/agents/solution_analyst.py` | Agent 3 — Claude + RAG for product/directive matching |
| Knowledge base | `knowledge_base/` | Qdrant ingestion and search for products and EU directives |
| Config | `app/config.py` | All settings via environment variables (pydantic-settings) |

## Project structure

```
multiAgent_fastAPI/
├── app/
│   ├── agents/
│   │   ├── analyzer.py          # Agent 2: Problem analysis (Claude)
│   │   ├── classifier.py        # Agent 1: News classification (Ollama)
│   │   ├── claude_client.py     # Claude API wrapper
│   │   ├── ollama_client.py     # Ollama API wrapper
│   │   └── solution_analyst.py  # Agent 3: Product/directive matching (Claude + RAG)
│   ├── config.py                # Environment config
│   ├── databse.py               # PostgreSQL schema and UPSERT
│   ├── main.py                  # FastAPI webhook
│   ├── models.py                # Pydantic data contracts
│   ├── pipeline.py              # LangGraph graph definition
│   └── worker.py                # Redis consumer loop
├── knowledge_base/
│   ├── data/
│   │   ├── directives/          # EU directive markdown documents
│   │   └── products_sample.csv  # Product feature catalog
│   ├── embeddings.py            # Ollama embedding helper
│   ├── ingest_directives.py     # Loads directives into Qdrant
│   ├── ingest_products.py       # Loads product catalog into Qdrant
│   └── search.py                # Qdrant search functions (used by Agent 3)
├── tests/
│   ├── agent1_test.py
│   ├── agent2_test.py
│   ├── test1_webhook.py
│   ├── test2_redis.py
│   ├── test3_pipeline.py
│   ├── test4_database.py
│   └── test5_phase1.py
├── scripts/
│   └── inspect_redis.py
├── assets/                      # Architecture diagrams
├── docker-compose.yml           # PostgreSQL + pgAdmin + Qdrant
└── requirements.txt
```

## Setup

### 1. Start infrastructure

```bash
docker compose up -d
```

Starts PostgreSQL (5432), pgAdmin (5050), and Qdrant (6333/6334).

If PostgreSQL fails with a credentials error, the volume has stale data:

```bash
docker compose down -v
docker compose up -d
```

### 2. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Configure environment

Create a `.env` file in the project root:

```env
WEBHOOK_API_KEY=<generate with: python -c "import secrets; print(secrets.token_urlsafe(32))">
DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/news_pipeline
ANTHROPIC_API_KEY=sk-ant-...
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1:8b
QDRANT_URL=http://localhost:6333
EMBEDDING_MODEL=nomic-embed-text
```

### 4. Ingest the knowledge base

Run once to populate Qdrant with product features and EU directive articles:

```bash
python -m knowledge_base.ingest_products
python -m knowledge_base.ingest_directives
```

Verify retrieval works:

```bash
python -m knowledge_base.search --query "city losing water through old pipes"
python -m knowledge_base.search --query "energy efficiency requirements" --collection directives
```

### 5. Start the webhook

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 6. Start the worker

```bash
python -m app.worker
```

### 7. Run tests

```bash
python tests/test1_webhook.py
python tests/test2_redis.py
python tests/test3_pipeline.py
python tests/test4_database.py
python tests/test5_phase1.py
```

## Connect n8n

Add an HTTP Request node at the end of your RSS workflow:

- **URL:** `http://your-server:8000/webhook/news`
- **Method:** POST
- **Headers:** `X-API-Key: <your key>`
- **Body:**

```json
{
  "title": "{{ $json.title }}",
  "summary": "{{ $json.description }}",
  "link": "{{ $json.link }}",
  "published_at": "{{ $json.pubDate }}"
}
```

`country`, `city`, `tags`, and `raw_content` are optional — Agent 1 extracts country and city from the article text if not provided.

## API endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Health check (Redis connectivity) |
| POST | `/webhook/news` | Receive a news item from n8n |
| GET | `/queue/stats` | Stream length, pending messages, dead letter count |

## Useful commands

### Redis

```bash
# Items in queue
docker exec -it <redis-container> redis-cli XLEN news:incoming

# Consumer group status
docker exec -it <redis-container> redis-cli XINFO GROUPS news:incoming

# Dead letter queue
docker exec -it <redis-container> redis-cli XLEN news:dead_letter

# Reprocess all items (reset consumer group)
docker exec -it <redis-container> redis-cli XGROUP SETID news:incoming pipeline-wokers 0
```

### PostgreSQL

```bash
# Row count by status
docker exec -it <postgres-container> psql -U postgres -d news_pipeline \
  -c "SELECT status, COUNT(*) FROM pipeline_results GROUP BY status;"

# Recent completed items
docker exec -it <postgres-container> psql -U postgres -d news_pipeline \
  -c "SELECT pipeline_id, country, sector, title FROM pipeline_results WHERE status='completed' ORDER BY completed_at DESC LIMIT 5;"

# Full reset
docker exec -it <postgres-container> psql -U postgres -d news_pipeline \
  -c "TRUNCATE pipeline_results;"
```

### Knowledge base search (interactive)

```bash
python -m knowledge_base.search
```

## Configuration reference

All settings are in `app/config.py` and can be overridden via `.env`:

| Variable | Default | Description |
|---|---|---|
| `WEBHOOK_API_KEY` | `change-me-to-a-real-secret` | API key for the webhook endpoint |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis connection URL |
| `DATABASE_URL` | `postgresql+asyncpg://...` | PostgreSQL connection URL |
| `ANTHROPIC_API_KEY` | — | Claude API key (required) |
| `CLAUDE_MODEL` | `claude-sonnet-4-6` | Model for Agents 2, 3, 4 |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_MODEL` | `llama3.1:8b` | Model for Agent 1 |
| `QDRANT_URL` | `http://192.168.2.185:6333` | Qdrant vector store URL |
| `EMBEDDING_MODEL` | `nomic-embed-text` | Ollama embedding model |
| `WORKER_CONCURRENCY` | `2` | Parallel pipeline executions |
| `MAX_RETRIES` | `3` | Retry attempts before dead-lettering |

## Product portfolio (Agent 1 sector taxonomy)

| Product | Monitors |
|---|---|
| Water Clamp Sensor | Leaks, consumption, freezing, stagnation, backflow |
| Gas Clamp Sensor | Leaks, consumption, methane, appliance efficiency |
| Electrical Panel Sensor | Load detection, ghost loads, safety, consumption |
| SensePod | Flood, humidity, condensation, vibration, mold |

Agent 1 classifies each article into one of: `water_infrastructure`, `gas_infrastructure`, `electrical_infrastructure`, `energy_efficiency`, `muncipal_budgets`, `environmental_compliance`, or `not_relevant`.
