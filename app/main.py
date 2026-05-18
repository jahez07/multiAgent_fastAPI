"""
main.py - FastAPI webhook that receives news from n8n and queues it.

This is the "front door" of the pipeline. It does three things fast
and returns immediately (<50ms):

    1. Validates the API key (rejects unauthorized callers)
    2. Validates the JSON body against the NewsItem schema
    3. Checks for duplicates, then pushes to a Redis Stream

The actual processing happens in worker.py, which consumes from
the same Redis Stream asynchronously.
"""

import hashlib
import json
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from uuid import uuid4

import redis.asyncio as redis
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from app.config import settings
from app.models import NewsItem, tempNewsItem, PipelineState



# Logging
# ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger("webhook")



# Redis Connection - created once at startup, closed at shutdown
# ──────────────────────────────────────────────────────────────
redis_client = redis.Redis()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup / shutdown lifecycle.

    On startup:
        - Connect to Redis
        - Create the consumer group for the news stream
        (MKSTREAM creates the stream if it does not exist yet)

    On shutdown:
        - Close the Redis connection cleanly
    """
    global redis_client
    redis_client = redis.from_url(settings.redis_url, decode_response=True)

    # Ping Redis to fail fast if its not running
    try: 
        await redis_client.ping()
        logger.info("Connected to Redis at %s", settings.redis_url)
    except Exception as e:
        logger.error("Cannot connect to Redis: %s", e)
        raise

    # Create the consumer group (idempotent - ignore "already exists error")
    try:
        await redis_client.xgroup_create(
            name=settings.stream_name,
            groupname=settings.consumer_group,
            id="0",         # Start reading from the beginning of the stream
            mkstream=True,  # Create the stream if it does not exist
        )
        logger.info(
            "Consumer group '%s' ready on stream '%s'",
            settings.consumer_group,
            settings.stream_name,
        )
    except redis.ResponseError as e:
        if "BUSYGROUP" in str(e):
            logger.info("Consumer group '%s' already exists (OK)", settings.consumer_group)
        else:
            raise
    
    yield # <- App runs here

    await redis_client.aclose()
    logger.info("Redis connection closed")



# FastAPI app
# ──────────────────────────────────────────────────────────────
app = FastAPI(
    title = "News Intelligence Pipeline - Webhook",
    version = "0.1.0",
)

def _url_hash(url: str)-> str:
    """
    Create a short hash of the URL for dedup.
    We use SHA-256 truncated to 16 chars - collistion-safe for our volume.
    """
    return hashlib.sha256(url.encode()).hexdigest()[:16]



# Routes
# ──────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    """
    Health check endpoint

    Use this to verify the service is running and Redis is reachable.
    n8n can also use it as a pre-check before posting news.
    """
    try:
        await redis_client.ping()
        return {"status":"healthy", "redis":"connected"}
    except Exception as e:
        return JSONResponse(
            status_code=503, 
            content={"status":"unhealthy", "redis":"disconnected", "error":str(e)},
        )


@app.post("/webhook/news", status_code=202)
async def receive_news(
    news: tempNewsItem
):
    """
    Receives a news item from n8n, validates it, and queues it.

    Flow:
        1.  Check API key -> 403 if invalid 
        2.  Pydantic validates the body automatically -> 422 if malformed
        3.  Check if we've already seen this URL -> 200 "already processed"
        4.  Push to Redis Stream -> 202 Accepted

    Returns the pipeline_id so n8n or a dashboard can track the flow. 
    """
    
    # Step 1 : API key check

    # Step 2 : Pydantic validation
    # Automatically done by fastAPI, parsed the body against NewsItem. 

    # Step 3 : Dedup check
    # We maintain a Redis SET of URL hashes we've already processed
    # SISMEMBER is O(1) - this adds ~0.1ms to the request.
    url_hash = _url_hash(str(news.link))
    already_seen = await redis_client.sismember(settings.dedup_set, url_hash)

    if already_seen:
        logger.info("Duplicate skipped: %s", news.title[:60])
        return{
            "status":"duplicate",
            "message":"This URL has already been processed",
            "link": str(news.link),
        }
    
    # Step 4: Build pipeline state and queue it
    pipeline_id = str(uuid4())
    state = PipelineState(
        news=news,
        pipeline_id=pipeline_id,
        ingested_at=datetime.now(timezone.utc),
    )

    # XADD pushes a new entry to the Redis Stream. 
    # We store the full state as a single JSON field. 
    # The stream auto-generates a unique message ID (timestamp-based).
    await redis_client.xadd(
        settings.stream_name,
        {"data":state.model_dump_json()}
    )

    # Mark the URL as seen (with a 30-day expiry so the set doesn't grow forever)
    await redis_client.sadd(settings.dedup_set, url_hash)

    logger.info(
        "Queued: [%s] %s",
        pipeline_id[:8],
        news.title[:60],
    )

    return {
        "status": "accepted",
        "pipeline_id": pipeline_id,
        "message": "News item queued for processing",
    }

@app.get("/queue/stats")
async def queue_status():
    """
    Quick dashboard endpoint - shows how many items are in the queue,
    how many have been processed, and how many are in the dead leter queue.

    Useful for debugging and for a future monitoring dashboard.
    """
    stream_len = await redis_client.xlen(settings.stream_name)
    dead_letter_len = await redis_client.xlen(settings.dead_letter_stream)
    seen_count = await redis_client.scard(settings.dedup_set)

    # Get pending message (items claimed by workers but not yet acknowledged)
    try:
        pending_info = await redis_client.xpending(
            settings.stream_name,
            settings.consumer_group,
        )
        pending_count = pending_info.get("pending", 0) if pending_info else 0
    except Exception:
        pending_count = "unknown"

    return {
        "stream_length": stream_len,
        "pending": pending_count,
        "dead_letter": dead_letter_len,
        "unique_urls_seen": seen_count,
    }