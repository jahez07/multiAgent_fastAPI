"""
app/agents/classifier.py - Agent 1: News Classifier.

Runs on Ollama (Llama 3.1 8B) on local GPUs.

What it does:
    1. Extracts country and city from the news (if not already provided)
    2. Classifies the sector (water, gas, energy, budgets, etc.)
    3. Scores relevance to the product portfolio (0.0 to 1.0)
    4. Decides: should this go to Claude for deep analysis?

Why this matters:
    The n8n feed sends ~500 articles. Maybe 10-20% are relevant.
    Without this filter, every article triggers 3 Claude API calls.
    With this filter, only relevant news reaches Claude -> ~80% cost savings.

Prompt design choices:
    - System prompt defines the exact sector taxonomy matching the products.
    - Asks for JSON output with specific field names
    - Uses low temperature (0.1) for deterministic extraction
    - Includes relevance scoring criteria tied to the product capabilties
    - Fallback to safe defaults if Ollama fails (is_relevance = True so we 
    don't accidentally drop a real opportunity)
"""

import logging
from typing import Any

from pydantic import BaseModel, Field

from app.agents.ollama_client import ollama_generate

logger = logging.getLogger("pipeline")


# Sector taxonomy - maps to your product portfolio
# ──────────────────────────────────────────────────────────────
# Update this when a new product is added. Agent 1 will classify
# every news item into one of these sectors. 

SECTORS = {
    "water_infrastructure": (
        "Water supply, water pipes, water leaks, water loss, water quality, "
        "water treatment, water metering, water monitoring, water stagnation, "
        "Legionella risks, backflow contamination, water hammer, pipe freezing, "
        "muncipal water systems, water conservation, pipe maintenance, "
        "drought restrictions, water scarcity"
    ),
    "gas_infrastructure":(
        "Gas supply, gas pipes, gas leaks, methane detection, gas metering, "
        "gas monitoring, gas safety, gas distribution networks, natural gas, "
        "heating systems, boiler efficiency, gas appilance safety, "
        "gas consumption, inefficient gas appliances"
    ),
    "electrical_infrastructure":(
        "Electricity monitoring, electrical panel safety, circuit breaker issues, "
        "power consumption monitoring, electrical fire risks, overloaded circuits, "
        "ghost loads, phantom power, standby energy waste, smart grid, "
        "electricity cost optimization, appliance energy efficiency, "
        "building electrical safety inspections"
    ),
    "energy_efficiency":(
        "Building energy audits, energy consumption reduction, energy savings, "
        "HVAC efficiency, smart buildings, energy management, systems, "
        "renewable energy integrations, building renovation for energy performance, "
        "thermal insulation, energy performance certificate, "
        "EU Energy Efficiency Directive compliance"
    ),
    "muncipal_budgets":(
        "City budget planning, infrastructure spending, maintenance budgets, "
        "muncipal finance, cost overruns, deferred maintenance costs, "
        "utility cost management, public spending on infrastructure, "
        "smart city investments, infrastructure funding gaps"
    ),
    "environmental_compliance":(
        "EU environmental directives, regulatory compliance, environmental targets, "
        "emissions monitoring, sustainability reporting, carbon reduction targets, "
        "environmental impact assessments, green infrastructure mandates, "
        "water framework directive, drinking water directive, "
        "energy efficiency directive compliance"
    ),
    "not_relevant":(
        "Politics without infrastructure focus, sports, entertainment, crime, "
        "military conflicts, trade disputs, general enconmics without "
        "infrastrucutre angle, celebrity news, judicial proceedings"
    ),
}

# Build a readable sector list for the prompt
SECTOR_DESCRIPTIONS = "\n".join(
    f" -{name}: {desc}" for name, desc in SECTORS.items()
)

# System Prompt 
# ──────────────────────────────────────────────────────────────

SYSTEM_PROMPT = f"""You are a news classification agent for a company that sells IoT sensor products
for building and infrastructure monitoring. 

Our product portfolio:
    - Water Clamp Sensor: non-intrusive water pipe mointoring (leak detection, cosumption, freezing, stagnation, backflow)
    - Gas Clamp Sensor: non-intrusive gas pipe monitoring (leak detection, consumption, methane, appliance efficiency)
    - Electrical Panel Sensor: non-intrusive circuit-level electicity monitoring (load detection, ghost loads, safety consumption)
    - SensePod: environmental condition sensor (flood detection, humidity, condensation, vibration, mold prevention)

Your job is to analyze a news article and extract structured information.

SECTORS (classify into exactly one):
{SECTOR_DESCRIPTIONS}

RELEVANCE SCORING (0.0 to 1.0):
    0.9 - 1.0: Directly mentions water/gas/electricity infrastructure problems, leaks, monitoring needs, or building environmental issues
    0.7 - 0.8: Mentions energy efficiency, building audits, electrical safety, or municipal infrastructure spending
    0.5 - 0.6: Mentions EU environmental or energy directives, compliance requirements related to water/gas/energy/buildings
    0.3 - 0.4: Tangentially related - general infrastructure, muncipal governance, or smart city initiatives
    0.0 - 0.2: Not related - politics, sports, crime, military, entertainment, trade without infrastructure angle

RULES:
    - Extract the PRIMARY country and city mentioned in the article
    - If multiple countries are mentioned, pick the one most affected by the issue
    - If no specific city is mentioned, set city to null
    - If no specific country is mentioned, set country to null
    - Set is_relevant to true if relevance_score >= 0.4
    - Set is_relevant to false if relevance_score < 0.4

Respond with ONLY a JSON object in this exact format:
{{
    "country": "string or null",
    "city": "string or null",
    "sector": "one of the sector names above",
    "relevance_score": 0.0 to 1.0,
    "reasoning": "one sentence explaining your classification"
}}"""

# Output Validation

class ClassifierOutput(BaseModel):
    """Validates and normalizez the JSON output from Ollama."""
    country: str | None = None
    city: str | None = None
    sector: str = "not_relevant"
    relevance_score: float = Field(default=0.0, ge=0.0, le=1.0)
    is_relevant: bool =False
    reasoning: str = ""

# Agent Function

RELEVANCE_THRESHOLD = 0.4

async def classify(state: dict) -> dict[str, Any]:
    """
    Agent 1: Classify a news item using Ollama.
 
    Flow:
      1. Check if country/city already provided (skip extraction if so)
      2. Build prompt with title + summary
      3. Call Ollama for structured JSON extraction
      4. Validate output with Pydantic
      5. Fall back to safe defaults if anything fails
 
    The fallback strategy is deliberately optimistic:
      - If Ollama fails entirely → is_relevant=True (don't drop potential leads)
      - If Ollama returns but JSON is malformed → is_relevant=True
      - Only when Ollama explicitly scores below threshold → is_relevant=False
    """
    title = state.get("title", "")
    summary = state.get("summary", "")
    existing_country = state.get("country")
    existing_city = state.get("city")

    logger.info("   [Agent 1] Classifying: %s", title[:60])

    prompt = f"""Analyze this news article:
TITLE : {title}
SUMMARY: {summary}
Classify this article and respond with the JSON object.
    """

    # Call Ollama
    result = await ollama_generate(
        prompt=prompt,
        system=SYSTEM_PROMPT,
        temperature=0.1,
        timeout=60.0,
    )

    # Handle Ollama failure
    if result is None:
        logger.warning("    [Agent 1] Ollama failed - using optimistic fallback")
        return {
            "country": existing_city or "Unknown",
            "city": existing_city or "Unknown",
            "sector": "general",
            "relevance_score": 0.5,
            "is_relevant": True,
        }
    
    # Validate with Pydantic
    try:
        output = ClassifierOutput.model_validate(result)
    except Exception as e:
        logger.warning("    [Agent 1] Invalid output from Ollama: %s: - using fallback", e)
        return {
            "country": existing_city or "Unknown",
            "city": existing_city or "Unknown",
            "sector": "general",
            "relevance_score": 0.5,
            "is_relevant": True,
        }
    
    # Apply threshold  
    is_relevant = output.relevance_score >= RELEVANCE_THRESHOLD

    # Parse existing country/city if n8n provided them 
    country = existing_country or output.country or "Unknown"
    city = existing_city or output.city or "Unknown"

    logger.info(
        "   [Agent 1] Result: country=%s city=%s sector=%s score=%s relevant=%s",
        country, city, output.sector, output.relevance_score, is_relevant,
    )
    if output.reasoning:
        logger.info("   [Agent 1] Reasoning: %s", output.reasoning[:100])
    

    return {
        "country": country,
        "city": city,
        "sector": output.sector,
        "relevance_score": output.relevance_score,
        "is_relevant": is_relevant,
    }