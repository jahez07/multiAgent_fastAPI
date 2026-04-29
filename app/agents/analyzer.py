"""
app/agents/analyzer.py - Agent 2: Problem Analyzer.

Runs on Claude Sonnet via the Anthropic API.

What it does:
    1. Analyzes the root causes behind the news
    2. Assesses the scale and impact of the problem
    3. Maps stakeholders (decision makers, budget approvers, regulators)
    4. Scores urgency (how pressing) and opportunity (how good a fit)

All four analyses happen in ONE Claude API call - the system prompt
asks for a single JSON response containing everything. 

Prompt design choices:
    - System prompt includes your product portfolio context so Claude
    frames its analysis around problems your products can solve.
    - Stakeholder types are pre-defined (decision_maker, budget_approver,
    regulator, technical_evaluator, end_user) to standardize output.
    - Urgency and opportunity scoring criteria are explicit (not just
    "score from 0 to 1" - that produces random numbers)
    - The prompt asks for "reasoning" on each score so you can audit
    why the model scored the way it did.
"""

import logging
from typing import Any

from pydantic import BaseModel, Field

from app.agents.claude_client import claude_json

logger = logging.getLogger("pipeline")


# System prompt
# ──────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a senior business analyst working for a company that sells IoT monitoring 
sensors for buildings and infrastructure.

Our products:
    - Water Clamp Sensor: non-intrusive water pipe monitoring (leak detection, consumption, stagnation, backflow, 
    drought restriction monitoring)
    - Gas Clamp Sensor: non-intrusive gas pipe monitoring (leak detection, consumption, methane, 
    appliance efficiency)
    - Electrical Panel Sensor: non-intrusive circuit-level electrcity monitoring (load detection, ghost loads,
    heat safety, consumption tracking)
    - SensePod: environmental condition sensor (flood detection, humidity, condensation, vibration, mold prevention)

Your job is to analyze a news article and produce a structured analysis identifying:
1. The specific PROBLEMS the city or organization faces
2. The ROOT CAUSES behind each problem
3. The STAKEHOLDERS who would be involved in purchasing a solution
4. How URGENT this is and how good an OPPORTUNITY it represents for our products


STAKEHOLDER TYPES (use exactly these):
    - decision_maker: Person/department that approves the purchase
    - budget_approver: Person/department that controls the buget
    - regulator: Regulatory body enforcing compliance
    - technical_evaluator: Person/team that evaluates technical solutions
    - end_user: The people who would use the monitoring system daily

URGENCY SCORING (0.0 to 1.0):
    0.9-1.0: Immediate crisis — active leaks, safety incidents, regulatory deadlines within 6 months
    0.7-0.8: High pressure — public complaints, budget overruns, regulatory deadlines within 1-2 years
    0.5-0.6: Growing concern — trend worsening, political pressure building, proactive planning stage
    0.3-0.4: Low urgency — long-term planning, no immediate pressure
    0.0-0.2: No urgency — hypothetical or resolved

OPPORTUNITY SCORING (0.0 to 1.0):
    0.9-1.0: Perfect fit — the problem maps directly to one or more of our products' core features
    0.7-0.8: Strong fit — our products address the core problem but may need complementary solutions
    0.5-0.6: Moderate fit — our products address part of the problem
    0.3-0.4: Weak fit — tangential connection to our products
    0.0-0.2: No fit — our products don't address this problem
 
Respond with ONLY a JSON object in this exact format:
{
    "problems": [
        {
            "problem": "Clear one-sentence description of the problem",
            "root_cause": "Why this pro exists - the underlying cause",
            "scale": "How big: number of people affected, financial impact, geographic scope",
            "affected_population": "Who is directly impacted"
        }
    ],
    "stakeholders": [
        {
            "role": "Specific title or department name",
            "type": "one of: decision_maker, budget_analyzer, regulator, technical_evaluator, end_user",
            "influence": "How much influence they have on purchasing: high, medium, or low"
        }
    ],
    "urgency_score": 0.0,
    "urgency_reasoning": "One sentence explaining the urgency score",
    "opportunity_score": 0.0,
    "opportunity_reasoning": "One sentence explaning the opportunity score"
}
"""

# Output validation

class Problem(BaseModel):
    problem: str
    root_cause: str
    scale: str = ""
    affected_population: str = ""

class Stakeholder(BaseModel):
    role: str
    type: str = "decision_maker"
    influence: str = "medium"

class AnalyzerOutput(BaseModel):
    problems: list[Problem] = Field(default_factory=list)
    stakeholders: list[Stakeholder] = Field(default_factory=list)
    urgency_score: float = Field(default=0.5, ge=0.0, le=1.0)
    urgency_reasoning: str = ""
    opportunity_score: float = Field(default=0.5, ge=0.0, le=1.0)
    opportunity_reasoning: str = ""

# Agent function

async def analyze(state: dict) -> dict[str, Any]:
    """
    Agent 2: Analyze the problem behind a classified news item.

    Receives state from Agent 1 (title, summary, country, city, sector)
    and uses Claude to perfom deep analysis.

    Returns:
        problems:           List of problem objects with root causes
        stakeholders:       List of stakeholder objects with roles and influence
        urgency_score:      0.0-1.0 how pressing this is
        opportunity_score:  0.0-1.0 how well our products fit
    """
    title = state.get("title", "")
    summary = state.get("summary", "")
    country = state.get("country", "Unknown")
    city = state.get("city", "Unknown")
    sector = state.get("sector", "general")

    logger.info("   [Agent 2] Analyzing: %s - %s", country, title[:50])

    # Build the prompt 
    prompt = f"""Analyze this news article for business opportunities:
    
    TITLE: {title}
    SUMMARY: {summary}
    CONTEXT:
        Country: {country}
        City: {city}
        Sector: {sector}
    
    Identify the problems, root causes, stakeholders, and score the urgency and opportunity.
    Respond with the JSON object.
"""
    
    # Call Claude
    result = await claude_json(
        prompt=prompt,
        system=SYSTEM_PROMPT,
        max_tokens=2000,
        temperature=0.2,
    )

    # Handle Claude failure
    if result is None:
        logger.warning("    [Agent 2] Claude failed - using minimal fallback")
        return {
            "problems":[
                {
                    "problem": f"Issue reported in {city}, {country} related to {sector}",
                    "root_cause": "Unable to analyze - Claude API unavailable",
                    "scale": "Unknown",
                    "affected_population": "Unknown",
                },
            ],
            "stakeholders":[
                {"role": "Municpal authority", "type": "decision_maker", "influence": "High"},
            ],
            "urgecy_score": 0.5,
            "opportunity_score": 0.5,
        }
    # Validate with Pydanitc
    try:
        ouput = AnalyzerOutput.model_validate(result)
    except Exception as e:
        logger.warning("    [Agent 2] Invalid output from Claude: %s", e)
        # Use the raw dict if Pydantic fails - Claude usually returns close-enough
        return {
            "problems": result.get("problems", []),
            "stakeholders": result.get("stakeholders", []),
            "urgency_score": min(max(float(result.get("urgency_score", 0.5)), 0.0), 1.0),
            "opportunity_score": min(max(float(result.get("opportunity_score", 0.5)), 0.0), 1.0),
        }
    
    # Convert to dicts for LangGraph state
    problems = [p.model_dump() for p in ouput.problems]
    stakeholders = [s.model_dump() for s in ouput.stakeholders]

    logger.info(
        "   [Agent 2] Found %d problems, %d stakeholders | urgenc=%.2f opportunity=%.2f",
        len(problems), len(stakeholders),
        ouput.urgency_score, ouput.opportunity_score,
    )

    if ouput.urgency_reasoning:
        logger.info("   [Agent 2] Urgency: %s", ouput.urgency_reasoning[:100])
    if ouput.opportunity_reasoning:
        logger.info("   [Agent 2] Opportunity: %s", ouput.opportunity_reasoning[:100])
    
    return {
        "problems": problems,
        "stakeholders": stakeholders,
        "urgency_score": ouput.urgency_score,
        "opportunity_score": ouput.opportunity_score,
    }
