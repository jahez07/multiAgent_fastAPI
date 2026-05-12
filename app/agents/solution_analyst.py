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
    - If no product feature matches a problem, skip it
    - If no directive matches a solution, set directive fields to null
    - Be specific: "Our Water Clamp Sensor's proactive leak detection identifies abnormal flow pattern in Munich's agin 30-year-old pipe sections" not "Our sensor detects leaks"
    - Estimate financial or operational impact where the data supports it
    - Maximum 5 product matches (most relevant only)
    - Maximum 3 directive matches (most relevant only)

Respond with ONLY JSON object:
{
    "product_matches": [
        {
            "product": "Product name",
            "feature": "Specific feature name",
            "problem_addressed": "Which problem this solve",
            "how_it_helps": "2-3 sentences explaining specifically how this feature addresses the city's problem",
            "estimated_impact": "Quantified impact if possible (cost savings, `%` improvements, etc.)"
        }
    ],
    "directive_matches":[
        {
            "directive": "Full directive name and number",
            "article": "Specific article number and title",
            "solution_aligned": "Which product/feature this relates to",
            "alignment": "2-3 sentences explaining how deploying our solution helps comply with this specific article"
        }
    ]
}
"""

# Output validation

class ProductMatch(BaseModel):
    product: str = ""
    feature: str = ""
    problem_addressed: str = ""
    how_it_helps: str = ""
    estimated_impact: str = ""

class DirectiveMatch(BaseModel):
    directive: str = ""
    article: str = ""
    solution_aligned: str = ""
    alignment: str = ""

class SolutionOutput(BaseModel):
    product_matches: list[ProductMatch] = Field(default_factory=list)
    directive_matches: list[DirectiveMatch] = Field(default_factory=list)


# RAG retrieval helpers

def build_search_queries(state: dict) -> list[str]:
    """
    Build search queries from the problems identified by Agent 2. 

    Strategy: use each problem's description + root cause as a query. 
    This gives Qdrant more semantic signal than just the news title. 
    Also include the news title as a fallback query. 
    """
    queries = []
    problems = state.get("problems", [])

    for problem in problems:
        if isinstance(problem, dict):
            # Combine problem descrption + root cause for richer query
            parts = []
            if problem.get("problem"):
                parts.append(problem["problem"])
            if problem.get("root_cause"):
                parts.append(problem["root_cause"])
            if parts:
                queries.append(" ".join(parts)[:300])
    
    # Fallback: use the news title if no problems
    if not queries:
        title = state.get("title", "")
        summary = state.get("summary", "")
        queries.append(f"{title} {summary}"[:300])

    return queries