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
from datetime import datetime, timezone

from sqlalchemy import (
    Column, String, Float, Boolean, DateTime, Text, Integer,
    create_engine, text,
)
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, DeclarativeBase
from sqlalchemy.dialects.postgresql import JSONB

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


# Database operations
# ──────────────────────────────

async def init_db():
    """
    Create all tables if they don't exist
    Call this once at worker startup.
    """
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database tables ready")

async def save_result(state: dict):
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