"""
models.py - Data contracts fror the news ingestion pipeline.

NewsItem is the schema that n8n workflow must produce.
PipelineState is the evolving staste object that travels through
every agent in the LangGraph pipeline.
"""

from datetime import datetime
from pydantic import BaseModel, HttpUrl, Field


class NewsItem(BaseModel):
    """
    The JSON body n8n sends in POST /webook/news.

    Required Fields:
        - title:        Headline of the articles
        - summary:      2-3 sentence (n8n can generate this from RSS description)
        - source_url:   Original article URL (also used for dedup)
        - country:      ISO country or free-text (e.g. "Germany", "DE")
        - published_at: When the article was published
    
    Optional Fields:
        - city:         if n8n can extract it, great; otherwise Agent 1 will infer it
        - raw_content:  Full article test if available (improves Agent 2 analysis)
        - tags:         Any tags n8n alredy assigned (e.g. ["water", "infrastructure"]) 
    """

    title: str = Field(..., min_length=5, max_length=500)
    summary: str | None = None
    link: HttpUrl
    country: str | None = None
    isoDate: datetime
    city: str | None = None
    content: str = Field(..., min_length=10, max_length=2000)
    tags: list[str] | None = None


class tempNewsItem(BaseModel):
    title: str = Field(..., min_length=5, max_length=500)
    link: HttpUrl
    isoDate: datetime
    content: str = Field(..., min_length=10, max_length=2000)
    country: str | None = None
    city: str | None = None

class PipelineState(BaseModel):
    """
    The state object that flows through the LangGraph pipeline. 

    Each agent reads what it needs, add its output, and passes
    the enriched state to the next agent. This grows as it moves
    through the pipeline:

        Agent 1 (Classifier) adds: sector, relevance_score, is_relevant
        Agent 2 (Problem Analyzer) adds : problems, stakeholders, urgency_score
        Agent 3 (Solution Analyst) adds : product_mtches, directive_matches
        Agent 4 (Business Case) adds :business_case_sumary.
    """

    # Original news item (set at ingestion)
    news: tempNewsItem

    # Metadata (set by webhook)
    pipeline_id: str        # unique ID for this pipeline run
    ingested_at: datetime   # When the webhook received it
    retry_count: int = 0    # How many times this item has been retired

    # Agent 1: Classifier output
    sector: str | None = None
    relevance_score: float | None = None
    is_relevant: bool | None = None

    # Agent 2: Problem Analyzer output
    problems: list[dict] | None = None
    stakeholders: list[dict] | None = None
    urgency_score: float | None = None
    opportunity_score: float | None = None

    # Agent 3: Solution Analyst output
    product_matches: list[dict] | None = None
    directive_matches: list[dict] | None = None

    # Agent 4: Business Case Writer Output
    business_case_summary: str | None = None

    # Final Status
    status: str = "pending" # pending -> processing -> completed -> failed
    error: str | None = None