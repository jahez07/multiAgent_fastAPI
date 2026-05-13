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

async def retrieve_products(queries: list[str], top_k_per_query: int = 5) -> list[dict]:
    """
    Search the products collection for each query and deduplicate results.

    Deduplication by feature name prevents the same feature from 
    appearing multiple time when different problems match the same product.
    """
    seen_features = set()
    all_results = []

    for query in queries:
        try:
            results = await search_products(query, top_k=top_k_per_query)
            for r in results:
                feature_key = r["payload"].get("feature", "")
                if feature_key not in seen_features:
                    seen_features.add(feature_key)
                    all_results.append(r)
        except Exception as e:
            logger.warning(
                "   [Agent 3] Product search failed for query: %s - %s",
                query[:50], str(e)[:100]
            )
    
    # Sort by relevance score descending, keep top 10
    all_results.sort(key=lambda x:x.get("score", 0), reverse=True)

    return all_results[:10]


async def retrieve_directives(queries: list[str], top_k_per_query: int = 3) -> list[dict]:
    """
    Search the directives collection for each query and deduplicate. 

    Deduplication by directives + article prevents the same article
    from appearing multiple times.
    """
    seen_articles = set()
    all_results = []

    for query in queries:
        try:
            results = await search_directives(query, top_k=top_k_per_query)
            for r in results:
                article_key = (
                    r["payload"].get("directive", "") + 
                    r["payload"].get("article", "")
                )
                if article_key not in seen_articles:
                    seen_articles.add(article_key)
                    all_results.append(r)
        except Exception as e:
            logger.warning(
                "   [Agent 3] Directive search failed for query: %s - %s", 
                query[:50], str(e)[:100]
            )
    all_results.sort(key=lambda x:x.get("score", 0), reverse=True)
    return all_results[:8]


def format_rag_context(
        problems: list[dict],
        product_results: list[dict],
        directive_results: list[dict],
) -> str:
    """
    Format retrieved results into a structured context block for Claude.

    This is what Claude sees alongside the problems - it's the
    "knowledge" part of RAG. Formatted compactly to minimize tokens.
    """
    sections = []

    # Problems section
    sections.append("=== PROBLEMS IDENTIFIED ===")
    for i, p in enumerate(problems, 1):
        if isinstance(p, dict):
            sections.append(
                f"{i}. {p.get('problem', 'Unknown')}\n"
                f"  Root cause: {p.get('root_cause', 'Unknown')}\n"
                f"  Scale: {p.get('scale', 'Unknown')}"
            )
    
    # Product features section
    sections.append("\n=== MATCHING PRODUCT FEATURES FROM OUR CATALOG ===")
    for i , r in enumerate(product_results, 1):
        p = r["payload"]
        score = r.get("score", 0)
        sections.append(
            f"{i}. [{p.get('product', '?')}] {p.get('feature', '?')}"
            f"(revlevance: {score:.2f})\n"
            f"  {p.get('description', '')}"
        )
    
    # Directive articles section
    sections.append("\n=== RELEVANT EU DIRECTIVE ARTICLES ===")
    for i, r in enumerate(directive_results, 1):
        p = r["payload"]
        score = r.get("score", 0)
        # Truncate long directive text to save tokens
        text = p.get("text", "")[:400]
        sections.append(
            f"{i}. [{p.get('directive', '?')}] {p.get('article', '?')}"
            f"(relevance: {score:.2f})\n"
            f"  {text}"
        )
    
    return "\n".join(sections)


# Agent function
async def solve(state: dict) -> dict[str, Any]:
    """
    Agent 3: Match products to problems and align with EU directives.

    Flow:
        1. Build search queries from problems given by Agent 2
        2. Search Qdrant products collection
        3. Search Qdrant directives collection
        4. Format RAG content
        5. Pass to Claude for reasoning
        6. Validate and return structured output
    """
    title = state.get("title", "")
    country = state.get("country", "Unknown")
    city = state.get("city", "Unknown")
    sector = state.get("sector", "general")
    problems = state.get("problems", [])

    logger.info(
        "   [Agent 3] Matching solutions: %s - %s",
        country, title[:50]
    )

    # -- Step 1: Build search queries --
    queries = build_search_queries(state=state)
    logger.info(
        "   [Agent 3] Built %d search queries from %d problems",
        len(queries), len(problems)
    )

    # Step 2 & 3: Search Qdrant
    product_results = await retrieve_products(queries, top_k_per_query=5)
    directive_results = await retrieve_directives(queries, top_k_per_query=5)

    logger.info(
        "   [Agent 3] Retrieved %d product features, %d directive articles",
        len(product_results), len(directive_results)
    )

    # Handle empty retrieval
    if not product_results and not directive_results:
        logger.warning("   [Agent 3] No results from Qdrant - returning empty matches")
        return{
            "product_matches": [],
            "directive_matches": [],
        }
    
    # Step 4: Format RAG Context
    rag_context = format_rag_context(problems, product_results, directive_results)

    # Step 5: Call Claude
    prompt = f"""Analyze this situation and map our products to the problems:

    NEWS: {title}
LOCATION: {city}, {country}
SECTOR: {sector}

{rag_context}

Based on the problems identified and the product features and directive articles retrieved above,
create the product-to-problem mappings and directive alignments.
Respond with the JSON object."""
    
    result = await claude_json(
        prompt=prompt,
        system=SYSTEM_PROMPT,
        max_tokens=2000,
        temperature=0.2,
        timeout = 60.0
    )

    # Handle Claude failure
    if result is None:
        logger.warning("    [Agent 3] Claude failed = returning raw retrieval results")
        # Fallback: return top product matches without Claude reasoning
        fallback_products = []
        for r in product_results[:3]:
            p = r["payload"]
            fallback_products.append({
                "product": p.get("product", ""),
                "feature": p.get("feature", ""),
                "problem_addresses": "Analysis unavailable - Claude API failed",
                "how_it_helps": p.get("description", ""),
                "estimated_impact": "Unknown",
            })
        
        return {
            "product_matches": fallback_products,
            "directive_matches": [],
        }
    

    # Step 6: Validate with Pydantic
    try:
        output = SolutionOutput.model_validate(result)
        product_matches = [m.model_dump() for m in output.product_matches]
        directive_matches = [m.model_dump() for m in output.directive_matches]
    except Exception as e:
        logger.warning("    [Agent 3] Pydantic validation failed: %s - using raw", e)
        product_matches = result.get("product_matches", [])
        directive_matches = result.get("directive_matches", [])

    logger.info(
        "   [Agent 3] Result: %d product matches, %d directive matches",
        len(product_matches), len(directive_matches),
    )

    return {
        "product_matches": product_matches,
        "directive_matches": directive_matches,
    }