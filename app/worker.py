"""
worker.py - Async worker that bridges Redis -> LangGraph -> PostgreSQL

This is the engine of the whole system. It runs as a separate process:
    python -m app.worker

What happens when a news item arrives:

    1. WAIT     - XREADGROUPS blocks until Redis has a new item
    2. READ     - Gets the message ID + PipelineState JSON
    3. CONVERT  - Deserializes JSON -> PipelineState -> flat GraphState dict
    4. RUN      - Feeds GraphState into LangGraph (agents 1 > 2 > 3 > 4)
    5. SAVE     - Writes enriched result to PostgreSQL
    6. ACK      - Tells Redis the message is processed (XACK)
    7. LOOP     - Back to step 1

If step 4 fails:
    - Retry up to MAX_RETRIES times ( re-queue to the same stream )
    - After MAX_RETRIES: move to dead letter stream + save failure to DB

Run with:
    python -m app.worker
"""

import asyncio
import logging
import signal
from datetime import datetime, timezone

import redis.asyncio as redis

from app.config import settings
from app.models import PipelineState
from app.pipeline import build_graph
from app.databse import save_result, init_db

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)8s %(name)s %(message)s',
)
logger = logging.getLogger("worker")

# Convert between PipelineState (Pydantic) and GraphState (flat dict)
# ────────────────────────────────────────────────────────────────────────────

def state_to_graph(state: PipelineState) -> dict:
    """
    Convert PipelineState -> flat dict for LangGraph

    LangGraph works with flat TypeDict, not nested Pydantic models.
    We flatten the NewsItem/tempNewsItem fields into the top level so each agent
    can access them directly ( e.g: state['title] instead of state['news']['title]).
    """
    return{
        "pipeline_id": state.pipeline_id,
        "title": state.news.title,
        "summary": state.news.content if state.news.content else "",
        "source_url": state.news.link,
        "published_at": state.news.isoDate.isoformat() if state.news.isoDate else None,
        "ingested_at": state.ingested_at.isoformat() if state.ingested_at else None,
        "country": state.news.country if state.news.country else None,
        "city": state.news.city if state.news.city else None,
        "status": "processing",
        "retry_count": state.retry_count,
    }


# Worker Class
# ──────────────────────────────────────────────────────────────
class NewsWorker:
    """
    Async worker that pulls items from Redis and runs the pipeline.

    Key design choices:
        - XREADGROUP with BLOCK: efficient waiting, no CPU-burning polling.
        - Unique consumer name - Redis tracks what each worker has pending.
        - asyncio.Semaphore - controls parallel item count (default:2)
        - Graceful shutdown - SIGINT/SIGTERM finish current items, then exit
    """

    def __init__(self, consumer_name: str = "worker-1"):
        self.cosumer_name = consumer_name
        self.redis: redis.Redis | None = None
        self.graph = None
        self.running = False
        self.semaphore = asyncio.Semaphore(settings.worker_concurrency)

    async def startup(self):
        """
        Initialize all conncetions and compile the pipeline. 
        Called once before the worker loop starts.
        """
        # ───── Connect to Redis ─────
        self.redis = redis.from_url(settings.redis_url, decode_responses=True)
        await self.redis.ping()
        logger.info("Connected to Redis")

        # ───── Ensure consumer group exists ─────
        try:
            await self.redis.xgroup_create(
                name=settings.stream_name,
                groupname=settings.consumer_group,
                id="0",
                mkstream=True,
            )
        except redis.ResponseError as e:
            if "BUSYGROUP" in str(e):
                pass
            else:
                raise
            
        logger.info("Consumer group '%s' ready", settings.consumer_group)

        # ───── Initialize PostgreSQL tables ─────
        await init_db()

        # ───── Compile the LangGraph pipeline ( once, reused for every item) ─────
        self.graph = build_graph()

    async def process_message(self, message_id: str, data: dict):
        """
        Process one message: deserialize -> run pipeline -> save -> ack
        """
        async with self.semaphore:
            try:
                # ───── Step 2 & 3: Deserialize and convert ─────
                state = PipelineState.model_validate_json(data["data"])
                logger.info(
                    "Processing: [%s] %s",
                    state.pipeline_id[:8],
                    state.news.title[:60],
                )
                graph_input = state_to_graph(state)

                # ───── Step 4: Run LangGraph pipeline ─────
                # This calls: classify -> (if relevant) -> analyze -> solve -> write_case
                result = await self.graph.ainvoke(graph_input)

                # ───── Step 5: Save to PostgreSQL ─────
                await save_result(result)

                # ───── Step 6: Acknowledge in Redis ─────
                # XACK tells Redis this message is fully processed. 
                # If we crash BEFORE this line, the message stays in
                # the pending list and can be reclaimed later.
                await self.redis.xack(
                    settings.stream_name,
                    settings.consumer_group,
                    message_id,
                )

                logger.info(
                    "Done: [%s] status=%s | country=%s | sector=%s",
                    result.get("pipeline_id", "?")[:8],
                    result.get("status", "?"),
                    result.get("country", "?"),
                    result.get("sector", "?"),
                )

            except Exception as e:
                logger.error(
                    "Failed [%s]: %s - %s",
                    message_id, type(e).__name__, str(e)[:200],
                )
                await self._handle_failure(message_id, data, e)
    async def _handle_failure(self, message_id: str, data: dict, error: Exception):
        """
        Retry logic:
            - Under MAX_RETRIES -> re-queue (new message in stream)
            - At MAX_RETRIES -> dead letter stream + save failure to DB
            - Always ACK the original message
        """
        try:
            state = PipelineState.model_validate_json(data["data"])
            state.retry_count += 1

            if state.retry_count < settings.max_retries:
                logger.warning(
                    "Retrying (%d/%d): %s",
                    state.retry_count, settings.max_retries,
                    state.news.title[:60],
                )
                await self.redis.xadd(
                    settings.stream_name,
                    {"data": state.model_dump_json()},
                )
            else:
                state.status = "failed"
                state.error = f"{type(error).__name__} : {str(error)[:500]}"
                logger.error(
                    "Dead-lettered after %d retries: %s",
                    settings.max_retries, state.news.title[:60],
                )

                # Save to Redis dead letter stream
                await self.redis.xadd(
                    settings.dead_letter_stream,
                    {
                        "data": state.model_dump_json(),
                        "failed_at": datetime.now(timezone.utc).isoformat(),
                        "error": state.error,
                    },
                )

                # Also save failure to PostgreSQL (dashboard shows it)
                await save_result({
                    "pipeline_id": state.pipeline_id,
                    "title": state.news.title,
                    "summary": state.news.content if state.news.content else "",
                    "source_url": str(state.news.link),
                    "published_at": state.news.isoDate.isoformat() if state.news.isoDate else None,
                    "ingested_at": state.ingested_at.isoformat() if state.ingested_at else None,
                    "status": "failed",
                    "error": state.error,
                    "retry_count": state.retry_count,
                    "business_case_summary": state.business_case_summary if state.business_case_summary else "",
                })
        except Exception as inner_e:
            logger.error("Error in failure handler: %s", inner_e)

        
        # Always ACK the original - the retry is a NEW message
        await self.redis.xack(
            settings.stream_name, settings.consumer_group, message_id,
        )

    async def run(self):
        """
        Main loop: XREADGROUP -> process -> repeat.

        XREADGROUP params explained:
            groupname       - our consumer group name
            consumername    - this specific worker's name
            streams         - which stream to read from
            '>'             - "only give me NEW messages"
            count           - max items per batch
            block=5000      - wait up to 5s if nothing available
        """
        self.running = True
        logger.info(
            "Worker '%s' listening (concurrency=%d)",
            self.cosumer_name, settings.worker_concurrency,
        )

        while self.running:
            try:
                messages = await self.redis.xreadgroup(
                    groupname=settings.consumer_group,
                    consumername=self.cosumer_name,
                    streams={settings.stream_name: ">"},
                    count=settings.worker_concurrency,
                    block=5000,
                )

                if not messages:
                    continue

                for stream_name, entries in messages:
                    tasks = [
                        self.process_message(msg_id, fields)
                        for msg_id, fields in entries
                    ]
                    await asyncio.gather(*tasks)
            
            except redis.ConnectionError:
                logger.error("Redis connection lost - recoonnecting in 5s")
                await asyncio.sleep(5)

                try:
                    self.redis = redis.from_url(settings.redis_url, decode_response=True)
                    await self.redis.ping()
                except Exception:
                    pass
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Unexpected error: %s", e)
                await asyncio.sleep(1)
            
        logger.info("Worker '%s' stopped", self.cosumer_name)
        
    async def shutdown(self):
        logger.info("Shutdown signal received")
        self.running = False
    
    async def cleanup(self):
        if self.redis:
            await self.redis.aclose()

# ────────────── Main entry point ───────────────────────────
async def main():
    worker = NewsWorker(consumer_name="worker-1")
    await worker.startup()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda: asyncio.create_task(worker.shutdown()))

    try:
        await worker.run()
    finally:
        await worker.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
    
