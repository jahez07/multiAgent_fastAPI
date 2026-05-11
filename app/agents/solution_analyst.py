"""
app/agents/solution_analyst.py - Agent 3: Solution Analyst (Claude + RAG)

The most complex and most valuable agent. It bridges:
    - Agent 2's problem analyst (what's wrong)
    - Your product catalog in Qdrant (what you sell)
    - EU directive docs in Qdrant (what regulations require)

Flow for each news item:
    1. Take the problems identified by Agent 2
    2. Build search queries from each problem description
    3. Query Qdrant product collection -> top matching features
    4. Query Qdrant directives collection -> top matching articles
    5. Pass everything to Claude in a structured prompt
    6. Claude explains: Feature X solves Problem Y, aligned with Directive Z

This agent produces the output the sales team actually uses:
"Munich is losing 30% of water -> our Water Clamp Sensor's leak detection
feature addresses this -> and it helps them comply with Article 4 of the
Water Framework Directive." 
"""

import logging
from typing import Any

from pydantic import BaseModel, Field

from app.agents.claude_client import claude_json
from knowledge_base.search import search_products, search_directives

logger = logging.getLogger("pipeline")

# System Prompt
SYSTEM_PROMPT = """You are a solution architect for a company that sells IoT monitoring devices and AI solutions based on the IoT data. Your job is to map specific product features city problems,
and align those solutions with EU regulatory requirements.

You will receive:
    1. PROBLEMS identified in a news article (from a previous analysis)
    2. MATCHING PRODUCT FEATURES retrieved from our product catalog
    3. RELEVANT EU DIRECTIVE ARTICLES retrieved from our requlatory knowledge base

Your task:
    - For each problem, idenfity which of our retrieved product features BEST addresses it.
    - Explain specifically HOW the feature solves the problem (not generic marketing - tie it to the specific situation)
    - Explain the regulatory alignment concretely ( cite the specific article requirement )

RULES:
    - Only match features that genuinely address the problem - don't force-fit
    - If no product feature
"""