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
from app.config import settings
from app.databse import log_error

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

SYSTEM_PROMPT = f"""You are a news classification agent for SureFlow.
=== WHAT SUREFLOW DOES (memorize this) ===
SureFlow makes AI-powered IoT sensors that clamp onto EXISTING pipes and electrical panels — no cutting, no drilling, no plumber needed. We monitor:

WATER flowing through PIPES in buildings (leak detection, consumption, freezing, stagnation, backflow)
GAS flowing through PIPES in buildings (leak detection, consumption, methane patterns, appliance efficiency)
ELECTRICITY flowing through CIRCUITS in panels (load detection, ghost loads, heat safety, consumption per circuit)
BUILDING ENVIRONMENT via SensePod (flood detection, humidity, condensation, vibration, mold risk)

Our customers: utilities, municipalities, property managers, housing associations, insurance companies.
Our value: reduce bills, prevent damage, comply with regulations, enable smart city deployments.
=== WHAT SUREFLOW DOES NOT DO ===

We do NOT monitor rivers, lakes, oceans, or natural water bodies
We do NOT clean up environmental pollution
We do NOT build power grids, wind farms, or solar panels
We do NOT do agricultural irrigation or industrial wastewater treatment
We do NOT do general environmental activism or climate policy
We do NOT do urban planning, architecture, or city design

=== YOUR TASK ===
Classify a news article by sector and score its relevance to SureFlow's business.
SECTORS (classify into exactly one):
{SECTOR_DESCRIPTIONS}
=== RELEVANCE SCORING — BE STRICT ===
RELEVANT (score 0.7-1.0):

Water/gas PIPE problems in buildings or utility networks (leaks, aging pipes, water loss, gas safety)
Electricity monitoring needs (smart meters, circuit safety, overloaded panels, electrical fires in buildings)
Building environmental issues (mold, condensation, flooding INSIDE buildings, humidity damage)
Smart metering deployments or IoT sensor adoption for utilities
EU directives that SPECIFICALLY mandate water/gas/electricity metering, monitoring, or consumption reporting
Property managers or housing associations dealing with maintenance, damage prevention, or tenant welfare
Budget pressures on municipalities to reduce water loss, energy waste, or infrastructure maintenance costs
Competitor companies selling similar IoT monitoring products for pipes, panels, or buildings

SOMEWHAT RELEVANT (score 0.4-0.6):

General energy efficiency policy (building renovation directives, energy performance certificates)
Smart home or smart city market trends that INCLUDE monitoring or sensor technology
ESG/sustainability reporting requirements that could drive demand for consumption data
Energy pricing or utility cost articles (high bills drive demand for our consumption monitoring)
Insurance industry interest in water damage prevention or building risk monitoring

NOT RELEVANT (score 0.0-0.3) — even if it mentions water, energy, or infrastructure:

River, lake, ocean, or bathing water quality (we monitor PIPES, not natural water bodies)
Environmental pollution or contamination of natural resources
Geopolitical energy disputes (oil prices, gas pipeline geopolitics between countries)
Power grid construction, transmission lines, or renewable energy generation (we monitor at BUILDING level)
General EU political drama, even if it mentions energy or environment in passing
Urban planning, architecture, or neighborhood design (no monitoring angle)
Agricultural water, industrial wastewater, or mining
Climate activism, protests, or opinion pieces without concrete infrastructure relevance
Sports, entertainment, crime, military, celebrity news

=== CRITICAL DECISION TEST ===
Before scoring, ask yourself: "Would a SureFlow sales person send this article to a customer?"

"Munich loses 30% of water in pipes" → YES, our Water Clamp Sensor detects pipe leaks
"Don't swim at river bathing sites" → NO, we don't monitor rivers
"EU mandates smart water metering by 2030" → YES, drives demand for our sensors
"EU Commission faces political backlash" → NO, general politics
"Vienna mandates electrical panel inspections" → YES, our Electrical Panel Sensor helps compliance
"Sunburn inspires new energy storage method" → NO, R&D news about batteries

=== EXAMPLES ===
Input: "Europe's digital water market forecast to double by 2033"
Summary: "Bluefield Research forecasts Europe's digital water market will double, driven by EU water policy mandates and smart metering adoption."
Output: {{"country": null, "city": null, "sector": "water_infrastructure", "relevance_score": 0.90, "is_relevant": true, "reasoning": "Digital water monitoring market growth directly impacts demand for our Water Clamp Sensor"}}
Input: "Smart electricity meter penetration in Europe reached 63% at end of 2024"
Summary: "Berg Insight reports smart electricity meter adoption across Europe reached 63%."
Output: {{"country": null, "city": null, "sector": "smart_utility_and_iot", "relevance_score": 0.75, "is_relevant": true, "reasoning": "Smart meter rollout creates adjacent market for our circuit-level Electrical Panel Sensor"}}
Input: "AIUT showcases smart utility IoT technologies at Enlit Europe 2025"
Summary: "Polish IoT company presents smart water and gas network monitoring technologies at Europe's largest energy trade fair."
Output: {{"country": "Spain", "city": "Bilbao", "sector": "smart_utility_and_iot", "relevance_score": 0.85, "is_relevant": true, "reasoning": "Competitor showcasing smart water and gas monitoring — directly in our product space"}}
Input: "Don't swim at 12 of 14 river bathing sites, as more pollution found"
Summary: "Water quality tests show dangerous pollution levels at river bathing sites across the country."
Output: {{"country": "United Kingdom", "city": null, "sector": "not_relevant", "relevance_score": 0.10, "is_relevant": false, "reasoning": "River and bathing water quality — SureFlow monitors water in pipes, not natural water bodies"}}
Input: "We're living in a shed because of river pollution"
Summary: "Residents displaced by contaminated river water affecting their homes and community."
Output: {{"country": null, "city": null, "sector": "not_relevant", "relevance_score": 0.15, "is_relevant": false, "reasoning": "Environmental river pollution causing displacement — not related to pipe or building monitoring"}}
Input: "European Commission faces backlash over plans to fast-track legislation"
Summary: "EU lawmakers criticize the Commission's approach to pushing through new regulations."
Output: {{"country": null, "city": null, "sector": "not_relevant", "relevance_score": 0.10, "is_relevant": false, "reasoning": "General EU political process — no connection to utility monitoring or infrastructure"}}
Input: "Copenhagen battles rising mold problems in public housing"
Summary: "Over 40% of public housing shows mold damage from inadequate ventilation and humidity control."
Output: {{"country": "Denmark", "city": "Copenhagen", "sector": "building_environment", "relevance_score": 0.90, "is_relevant": true, "reasoning": "Mold from humidity in public housing — our SensePod monitors exactly these conditions for early detection"}}
Input: "EU seeks better Spain-France energy links after blackout"
Summary: "EU pushes to boost energy interconnections between France and Spain after a massive blackout."
Output: {{"country": "Spain", "city": null, "sector": "electrical_infrastructure", "relevance_score": 0.55, "is_relevant": true, "reasoning": "Grid instability increases building-level demand for power monitoring — our Electrical Panel Sensor tracks consumption during disruptions"}}
Input: "Electricity and gas prices across Europe: Which countries are the most expensive?"
Summary: "Compares electricity and gas prices across EU member states."
Output: {{"country": null, "city": null, "sector": "energy_efficiency", "relevance_score": 0.60, "is_relevant": true, "reasoning": "High utility bills drive demand for consumption monitoring — our sensors help identify waste and reduce costs"}}
Input: "The five-minute city: inside Denmark's revolutionary neighbourhood"
Summary: "Denmark's urban planning concept where all daily needs are within a 5-minute walk."
Output: {{"country": "Denmark", "city": null, "sector": "not_relevant", "relevance_score": 0.10, "is_relevant": false, "reasoning": "Urban planning and neighborhood design — no infrastructure monitoring angle"}}
=== RULES ===

Extract the PRIMARY country and city mentioned
If the article covers all of Europe or multiple countries, set country to null
If no specific city is mentioned, set city to null
is_relevant = true ONLY if relevance_score >= 0.4
When in doubt, apply the sales person test: would SureFlow use this to open a customer conversation?

Respond with ONLY a JSON object:
{{
"country": "string or null",
"city": "string or null",
"sector": "one of the sector names listed above",
"relevance_score": 0.0 to 1.0,
"is_relevant": true or false,
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
        model_name = settings.ollama_model,
        agent_name = "classifier_agent_1",
        pipeline_id = state.get("pipeline_id"),
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

        await log_error(
            "agent_1", e,
            pipeline_id= state.get("pipeline_id"),
            context={"title": title[:100]},
            severity="warning",
        )
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