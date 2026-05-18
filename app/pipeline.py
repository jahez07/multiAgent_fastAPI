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
    │                          END
    │
    └── is_relevant=False → END (skip all agents, mark as "skipped")

The worker calls `graph.ainvoke(state)` and gets back the
fully enriched state with all agent outputs filled in. 
"""

import logging
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

# Agent 1: Classify News - Ollama Llama3.1:8b
from app.agents.classifier import classify

# Agent 2: Analyze News - Claude Sonnet API
from app.agents.analyzer import analyze

# Agent 3: Real implementation using Claude Sonnet + Qdrant RAG
from app.agents.solution_analyst import solve

# Agent 4: Business Case writer
from app.agents.business_case import write_case


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