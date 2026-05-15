"""
database.py - PostgreSQL storage for completed pipeline results.

This module handles:
    - Creating the table schema on startup
    - Saving enriched pipeline results after all agents run
    - Providing a query interface for the NextJS dashoard API

The table stores the FULL output of every pipeline run - the original
news, every agen't output, scores, and timestamps. This is the single 
source of truth that the NextJS dashboard reads from.

Schema design rationale:
    - JSON columns for flexible nested data (problems, matches, stakeholders)
    - Indexed columns for things the dashboard will filter/sort on
    - Timestamps for audit trail and time-based queries
"""

import json
import logging
import traceback
from datetime import datetime, timezone

from sqlalchemy import (
    Column, String, Float, Boolean, DateTime, Text, Integer,
    create_engine, text,
)
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, DeclarativeBase
from sqlalchemy.dialects.postgresql import JSONB, insert as pg_insert

from app.config import settings

logger = logging.getLogger("database")



# SQLAlchemy setup
# ──────────────────────────────────────────────────────────────

engine = create_async_engine(settings.database_url, echo=False)
async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


def parse_dt(val):
    if val is None:
        return None
    if isinstance(val, datetime):
        return val
    return datetime.fromisoformat(val)


def clean_jsonb(val):
    if val is None or val == "null" or val == "None":
        return None
    return val


class Base(DeclarativeBase):
    pass

class PipelineResult(Base):
    """
    One row per news item that entered the pipeline.

    This table is what the NextJS dashboard queries. Key columns:

        Filterable:
            - status:   "completed" | "skipped" | "failed"
            - country:  For geographic filtering
            - city:     For city-level drill down
            - sector:   For sector filtering (water, energy, etc.)
            - urgency_score / opportunity_score: For priorit sorting
        
        Full outputs (JSONB):
            - problems:         Array of problem objects from Agent 2
            - stakeholders:     Array of stakeholder objects from Agent 2
            - product_matches:  Array of feature-to-problem mappings from Agent 3
            - directive_matches:Array of EU directive alignments from Agent 3
        
        Summary:
            - business_case_summary: Full text pitch from Agent 4
    """
    __tablename__ = "pipeline_results"

    # ── Primary Key ──
    pipeline_id = Column(String, primary_key=True)

    # ── Original news ──
    title = Column(String, nullable=False)
    summary = Column(Text, nullable=True)
    source_url = Column(String, nullable=False)
    published_at = Column(DateTime(timezone=True))

    # ── Agent 1 output ──
    country = Column(String, index=True)
    city = Column(String, index=True)
    sector = Column(String, index=True)
    relevance_score = Column(Float)
    is_relevant = Column(Boolean)

    # ── Agent 2 output (JSONB for nested structures) ──
    problems = Column(JSONB)
    stakeholders = Column(JSONB)
    urgency_score = Column(Float, index=True)
    opportunity_score = Column(Float, index=True)

    # ── Agent 3 output ──
    product_matches = Column(JSONB)
    directive_matches = Column(JSONB)

    # ── Agent 4 output ──
    business_case_summary = Column(Text)

    # ── Metadata ──
    status = Column(String, index=True, default="pending")
    error = Column(Text)
    ingested_at = Column(DateTime(timezone=True))
    completed_at = Column(DateTime(timezone=True))
    retry_count = Column(Integer, default=0)


# Table 2: Centralized error log (new)

class PipelineError(Base):
    """
    Logs ALL errors from every component in the system. 

    Sources: webhook, worker, redis, agent_1, agent_2, agent_3,
             agent_4, database, qdrant, embedding, unknown
    """
    __tablename__ = "pipeline_errors"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime(timezone=True), nullable=True, index=True)
    source = Column(String, nullable=False, index=True)
    error_type = Column(String, nullable=False, index=True)
    error_message = Column(String, nullable=False)
    stack_track = Column(Text)
    pipeline_id = Column(String, index=True)
    context = Column(JSONB)
    severity = Column(String, nullable=False, index=True, default="error")


# Table 3: Claude API cost tracking

# Pricing per million tokens
PRICING = {
    "claude-sonnet-4-6": {"input": 3.00, "output": 15.00},
    "claude-haiku-4-5": {"input": 1.00, "output": 5.00},
}

# Default fallback pricing if model not in PRICING dict
DEFAULT_PRICING = {"input": 3.00, "output": 15.00}

class ApiUsage(Base):
    """
    Tracks every Claude API call with token counts and calculated costs. 
    Enables cost dashboards: daily spend, cost per agent, cost per opportunity.
    """
    __tablename__ = "api_usage"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime(timezone=True), nullable=False, index=True)
    pipeline_id = Column(String, index=True)
    agent = Column(String, nullable=False, index=True)
    model = Column(String, nullable=False)
    input_tokens = Column(Integer, nullable=False, default=0)
    output_tokens = Column(Integer, nullable=False, default=0)
    cache_read_tokens = Column(Integer, default=0)
    cache_write_tokens = Column(Integer, default=0)
    input_cost = Column(Float, nullable=False, default=0.0)
    output_cost = Column(Float, nullable=False, default=0.0)
    total_cost = Column(Float, nullable=False, default=0.0)
    response_time = Column(Float)
    success = Column(Boolean, nullable=False, default=True)


# Type Conversion functions

def parse_dt(val):
    if val is None:
        return None
    if isinstance(val, datetime):
        val
    return datetime.fromisoformat(val)

def clean_jsonb(val):
    if val is None or val == "null" or val == "None":
        return None
    return val


def calculate_cost(model: str, input_tokens: int, output_token: int) -> tuple[float, float, float]:
    """
    Calculate USD cost from token counts using the pricing table. 
    Returns (input_cost, output_cost, total_cost).
    """
    pricing = PRICING.get(model, DEFAULT_PRICING)
    input_cost = (input_tokens / 1_000_000) * pricing["input"]
    output_cost = (output_token / 1_000_000) * pricing["output"]
    return input_cost, output_cost, input_cost + output_cost



# Database operations
# ──────────────────────────────

async def init_db():
    """
    Create all tables if they don't exist
    Call this once at worker startup.
    """
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database tables ready (pipeline_results, pipeline_errors, api_usage)")

async def save_result_old(state: dict):
    """
    Save completed pipeline result to PostgreSQL.

    Takes the flat GraphState dict from LangGraph and writes it
    as a single row. The NextJS dashboard reads from this table.

    This is called by the worker after a successful pipeline run.
    """

    async with async_session() as session:
        result = PipelineResult(
            pipeline_id = state.get("pipeline_id"),
            title = state.get("title"),
            summary = state.get("summary"),
            source_url = str(state.get("source_url")),
            published_at = parse_dt(state.get("published_at")),
            country = state.get("country"),
            city = state.get("city"),
            sector = state.get("sector"),
            relevance_score = state.get("relevance_score"),
            is_relevant = state.get("is_relevant"),
            problems = clean_jsonb(state.get("problems")),
            stakeholders = state.get("stakeholders"),
            urgency_score = state.get("urgency_score"),
            opportunity_score = state.get("opportunity_score"),
            product_matches = state.get("product_matches"),
            directive_matches = state.get("directive_matches"),
            business_case_summary = state.get("business_case_summary"),
            status = state.get("status", "completed"),
            error = state.get("error"),
            ingested_at = parse_dt(state.get("ingested_at")),
            completed_at = datetime.now(timezone.utc),
            retry_count = state.get("retry_count", 0),
        )

        session.add(result)
        await session.commit()

        logger.info(
            "   Saved to DB: [%s] %s - %s (status=%s)",
            state.get("pipeline_id","?")[:8],
            state.get("country", "?"),
            state.get("title", "?")[:10],
            state.get("status", "?")
        )

async def save_result(state: dict):
    values = {
        "pipeline_id": state.get("pipeline_id"),
        "title": state.get("title"),
        "summary": state.get("summary"),
        "source_url": str(state.get("source_url" or "")),
        "published_at": parse_dt(state.get("published_at")),
        "country": state.get("country"),
        "city": state.get('city'),
        "sector": state.get("sector"),
        "relevance_score": state.get("relevance_score"),
        "is_relevant": state.get("is_relevant"),
        "problems": clean_jsonb(state.get("problems")),
        "stakeholders": clean_jsonb(state.get("stakeholders")),
        "urgency_score": state.get("urgency_score"),
        "opportunity_score": state.get("opportunity_score"),
        "product_matches": clean_jsonb(state.get("product_matches")),
        "directive_matches": clean_jsonb(state.get("directive_matches")),
        "business_case_summary": state.get("business_case_summary"),
        "status": state.get("status", "completed"),
        "error": state.get("error"),
        "ingested_at": parse_dt(state.get("ingested_at")),
        "completed_at": datetime.now(timezone.utc),
        "retry_count": state.get("retry_count", 0),
    }

    async with async_session() as session:
        stmt = pg_insert(PipelineResult).values(**values)
        stmt = stmt.on_conflict_do_update(
            index_elements=["pipeline_id"],
            set_=values,
        )
        await session.execute(stmt)
        await session.commit()

    logger.info(
        "   Saved to DB: [%s] %s - %s (status = %s)",
        state.get("pipeline_id", "?")[:8],
        state.get("country", "?"),
        state.get("title", "?")[:40],
        state.get("status", "?"),
    )


# Centralized error logging

async def log_error(
        source: str,
        error: Exception,
        pipeline_id: str | None = None,
        context: dict | None = None,
        severity: str = "error",
):
    """
    Log an error from ANY component to the pipeline_errors table.

    Args:
        source:     Where the error happened. Use these standard names:
                    webhook, worker, redis, agent_1, agent_2, agent_3,
                    agent_4, database, qdrant, embedding, unknown
        error:      The exception object
        pipeline_id:Which news item was being processed (None if not applicable)
        context:    Any extra info as a dict (news title, input data, etc.)
        severity:   "critical", "error", or "warning"

    Usage in any component:
        try:
            ...
        except Exception as e:
            await log_error("agent_2", e, pipeline_id="abc_123",
                            context={"title": "Munich water loss"})
    """
    try:
        # Serialize context safely
        safe_context = None
        if context:
            try:
                safe_context = json.loads(json.dumps(context, default=str))
            except Exception:
                safe_context = {"raw": str(context)[:1000]}
            
        async with async_session() as session:
            entry = PipelineError(
                timestamp = datetime.now(timezone.utc),
                source = source,
                error_type = type(error).__name__,
                error_message = str(error)[:2000],
                stack_trace = traceback.format_exc()[:5000],
                pipeline_id = pipeline_id,
                context = safe_context,
                severity = severity,
            )
            session.add(entry)
            await session.commit()
        
    except Exception as db_error:
        # If we can't even log the error, print to stderr as last resort
        logger.error(
            "FAILED TO LOG ERROR to database: %s - Original error: [%s] %s: %s",
            db_error, source, type(error).__name__, str(error)[:200]
        )


# API usage tracking

async def log_api_usage(
        agent: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        response_time: float,
        success: bool = True,
        pipeline_id: str | None = None,
        cache_read_tokens: int = 0,
        cache_write_tokens: int = 0,
):
    """
    Log a Claude API call with token counts and calculates costs.

    Called by claude_client.py after every API call (success or failure).

    Args:
        agent:              Which agent made the call (agent_1, agent_2, agent_3, agent_4)
        model:              Model string (e.g. "claude-sonnet-4-6")
        input_tokens:       Tokens in the request (prompt + system)
        output_tokens:      Tokens in the response
        response_time:      Seconds the call took
        success:            Did the call succeed
        pipeline_id:        Which news item triggered this
        cache_read_tokens:  Tokens read from prompt cache
        cache_write_tokens: Tokens written to prompt cache
    """
    try:
        input_cost, output_cost, total_cost = calculate_cost(
            model, input_tokens, output_tokens,
        )

        async with async_session() as session:
            entry = ApiUsage(
                timestamp = datetime.now(timezone.utc),
                pipeline_id = pipeline_id,
                agent = agent,
                model = model,
                input_tokens = input_tokens,
                output_tokens = output_tokens,
                cache_read_tokens = cache_read_tokens,
                cache_write_tokens = cache_write_tokens,
                input_cost = round(input_cost, 6),
                total_cost = round(output_cost, 6), 
                response_time = round(response_time, 2),
                success = success,
            )
            session.add(entry)
            await session.commit()

    except Exception as e:
        logger.error("Failed to log API usage: %s", e)