"""
pipeline.py - LangGraph agent pipeline.

This define the directed graph that processes each news item through
4 agents sequentially, with a conditional gate after Agent 1.

How LangGraph works (key concepts):
──────────────────────────────────────
LangGraph models the pipeline as a state machine:

    - STATE:    A TypeDict that every node can read from and write to. 
                In our case, this mirrors the PipelineState - news, country,
                sector, problems, product_matches, etc. 
    
    - NODES:    Functions that take the state, do work (call an LLM, 
                query a databse, run logic), and return a partial state
                update. LangGraph merges the update into the full state.
    
    - EDGES:    Define the order nodes run in. Can be:
                * Unconditional:    A always run after B
                * Conditional:      After A, chech a field and branch

Our graph looks like this:

  START
    │
    ▼
  classify (Agent 1 — Ollama)
    │
    ├── is_relevant=True  → analyze (Agent 2 — Claude)
    │                           │
    │                           ▼
    │                       solve (Agent 3 — Claude + RAG)
    │                           │
    │                           ▼
    │                       write_case (Agent 4 — Claude)
    │                           │
    │                           ▼
    │                         END
    │
    └── is_relevant=False → END (skip all agents, mark as "skipped")

The worker calls `graph.ainvoke(state)` and gets back the
fully enriched state with all agent outputs filled in. 
"""

import logging
from datetime import datetime, timezone
from typing import Any

from langgraph.graph import StateGraph, END

logger =logging.getLogger("pipeline")


# State definition for LangGraph
# ──────────────────────────────────────────────────────────────
# LangGraph uses TypedDict for state. We keep it flat (not nested)
# so each node can easily update individual fields. 

from typing import TypedDict

class GraphState(TypedDict, total=False):
    """
    The state that flows through the graph. 

    This is a flat dict version of PipelineState. We convert
    PipelineState -> GraphState at the start, and GraphState -> 
    PipelineState at the end.

    'total=False' means all fields are optional - each node only
    needs to return the fields it wants to update, not the entire state.
    """

    # ────── Input (set before graph starts) ──────
    pipeline_id: str
    title: str
    summary: str
    source_url: str
    published_at: str
    raw_content: str
    ingested_at: str

    # ────── Agent 1: Classifier ──────
    country: str | None
    city: str | None
    sector: str | None
    relevance_score: float | None
    is_relevant: bool | None

    # ────── Agent 2: Problem Analyzer ──────
    problems: list[dict] | None
    stakeholders: list[dict] | None
    urgency_score: float | None
    opportunity_score: float | None

    # ────── Agent 3: Solution Analyst ──────
    product_matches: list[dict] | None
    directive_matches: list[dict] | None

    # ────── Agent 4: Business Case Writer ──────
    business_case_summary: str | None

    # ────── Pipeline metadata ──────
    status: str
    error: str | None



# Agent nodes (PLACEHOLDERS - replace with real LLM calls in Phase 2)
# ──────────────────────────────────────────────────────────────

from app.agents.classifier import classify

async def analyze(state: GraphState) -> dict[str, Any]:
    """
    Agent 2: Problem Analyzer (runs on Claude API)

    Responsibilities:
      - Identify the core problems the city faces
      - Analyze root causes behind the news
      - Map stakeholders (who decides, who buys, who's affected)
      - Score urgency and opportunity

    In Phase 2, this will call Claude API:
    
    ```python
    import anthropic
    client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
    response = await client.messages.create(
        model=settings.claude_model,
        max_tokens=2000,
        system = "You are an expert analyst ....",
        messages=[
            {
                "role": "user",
                "content": f"Analyze: {state['title']}..."
            }
        ]
    )
    ```
    """
    logger.info(
        "   [Agent 2] Analyzing problems: %s - %s",
        state.get("country", "Unknown"),
        state.get("title")[:50],
    )

    # ── PLACEHOLDER: Simulate problem analysis ──
    return {
        "problems": [
            {
                "problem": "Aging pipe infrastructure causing water loss",
                "root_cause": "Defered maintenance and lack of monitoring",
                "scale": "City-wide, affecting 30% \of water supply",
            }
        ],
        "stakeholders": [
            {"role": "Municipal water utility", "type": "decision_maker"},
            {"role": "City council finance committee", "type": "budget_approver"},
            {"role": "EU compliance office", "type": "regulator"},
        ],
        "urgency_score": 0.9,
        "opportunity_score": 0.85,
    }

async def solve(state: GraphState) -> dict[str, Any]:
    """
    Agent 3: Solution Analyst (runs on Claude API + RAG)

    Responsibilities:
      - Match problems to specific product features
      - Explain HOW each feature solves the problem
      - Align the solution with relevant EU directives
      - Cite specific directive articles and requirements

    In Phase 2, this will:
      1. Query Qdrant vector store for relevant product features
      2. Query Qdrant for relevant EU directives
      3. Pass both to Claude with the problem analysis 
    """
    logger.info(
        "   [Agent 3] Matching solutions: %s",
        state.get("title")[:50],
    )

    # ── PLACEHOLDER: Simulate product + directive matching ──
    return {
        "product_matches": [
            {
                "feature": "Water leak detection",
                "problem_solved": "Aging pipe infrastructure causing water loss",
                "how_it_helps": (
                    "Real-time acoustic sensors detect leaks within hours"
                    "instead of weeks, reducing water loss from 30% to under 5%"
                ),
                "estimated_savings": "EUR 12M annually in recovered water",
            },
            {
                "feature": "Anomaly detection",
                "problem_solved": "Lack of monitoring infrastructure",
                "how_it_helps": (
                    "ML-based pattern recognition identifies pressure anomalies "
                    "before they become visible leaks, enabling preventive repair."
                ),
                "estimated_savings": "EUR 3M annually in avoided emergency repairs",
            },
        ],
        "directive_matches": [
            {
                "directive": "EU Water Framework Directive",
                "article": "Article 4 - Envoronmental objectives",
                "alignment": (
                    "Member states must achieve good status for all water bodies."
                    "Reducing 30% water loss directly supports this requiremenet."
                ),
            },
            {
                "directive": "Drinking Water Directive (EU 2020/2184)",
                "article": "Article 4 - Quality standards",
                "alignment": (
                    "Requires member states to minimize water loss. Our solution"
                    "provides the monitoring infrastructure needed for compliance."
                ),
            },
        ],
    }


async def write_case(state: GraphState) -> dict[str, Any]:
    """
    Agent 4: Business Case Writer (runs on Claude API)

    Responsibilities:
      - Synthesize everything into a compelling business case
      - Include: problem, solution, regulatory alignment, ROI
      - Write in a style suitable for presenting to city officials

    In Phase 2, Claude will receieve the full enriched state
    and produce a structured pitch document.
    """
    logger.info(
        "   [Agent 4] Writing business case: %s",
        state.get("title")[:50],
    )

    # ── PLACEHOLDER: Simulate business case generation ──
    country = state.get("country", "Unknown")
    city = state.get(("city"), "Unknown")

    return {
        "business_case_summary":(
            f"OPPORTUNITY: {city}, {country}\n\n"
            f"PORBLEM: City faces significant water infrastructure challenges "
            f"with 30% of water loss due to aging pipes.\n\n"
            f"SOLUTION: Deploy water leak detection and anomaly detection "
            f"systems across the municipal water network.\n\n"
            f"REGULATORY FIT: Aligns with EU Water Framework Directive "
            f"(Article 4) and Drinking Water Directive (EU 2020/2184).\n\n"
            f"ROI: Estimated EUR 15M annual savings through reduced water "
            f"loss and avoided emergency repairs. \n\n"
            f"RECOMMENDED NEXT STEP: Schedule a technical demo with the "
            f"municipal water utility."
        ),
        "status":"completed"
    }


# Routing logic (conditional edge after Agent 1)
# ──────────────────────────────────────────────────────────────
def should_continue(state: GraphState) -> str:
    """
    After Agent 1 classifies the news, decide whether to continue.

    if is_relevant is True: -> proceed to Agent 2 ("analyze")
    if is_relevant is False: -> skip to END (doesn't waste API calls)

    This is the gte that saves you money irrelavant news gets filtered by
    the cheap local model (Ollama) and never hits Claude.
    """
    if state.get("is_relevant", True):
        return "analyze"
    else:
        logger.info(
            "  [Gate] Not relevant (score=%.2f) - skipping",
            state.get("relevance_score",0)
        )
        return "skip"
    
async def mark_skipped(state: GraphState) -> dict[str, Any]:
    """Mark items that Agent 1 deemed irrelevant"""
    return {"status":"skipped"}


# Build the graph
# ──────────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    """
    Construct and compile the LangGraph pipeline.

    The compiled graph is runnable object - call it with:
        result = await graph.ainvoke(initial_state)

    Returns the filly enriched state with all agent outputs.
    """

    graph = StateGraph(GraphState)

    # ── Add nodes (each is an async function) ──
    graph.add_node("classify", classify)
    graph.add_node("analyze", analyze)
    graph.add_node("solve", solve)
    graph.add_node("write_case", write_case)
    graph.add_node("skip", mark_skipped)

    # ── Define the flow ──

    # Start -> Agent 1 (always runs first)
    graph.set_entry_point("classify")

    # Agent 1 -> conditional branch
    graph.add_conditional_edges(
        "classify",
        should_continue,
        {
            "analyze":"analyze",
            "skip":"skip"
        },
    )

    # Agent 2 -> Agent 3 (unconditional)
    graph.add_edge("analyze", "solve")

    # Agent 3 -> Agent 4 (unconditional)
    graph.add_edge("solve","write_case")

    # Agent 4 -> END
    graph.add_edge("write_case", END)

    # ── Compile and return ──
    compiled = graph.compile()
    logger.info("Pipeline graph compiled: classify -> [analyze -> solve -> write_case] -> END")
    return compiled