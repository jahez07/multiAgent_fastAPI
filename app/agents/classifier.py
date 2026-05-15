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
from app.agents.claude_client import claude_json
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

SYSTEM_PROMPT = f"""You are a news classification agent for a company that sells IoT sensor products for building and infrastructure monitoring.

Statement About Sureflow:
SureFlow is an R&D company specializing in AI-powered IoT sensors for smart utility monitoring. 
We help utilities, municipalities, and property managers monitor and manage water, energy, and gas resources in real time using non-intrusive, 
retrofit technology that integrates with existing infrastructure and requires no specialized installation. Our AI-driven platform turns raw 
consumption data into actionable intelligence, supporting sustainability goals and measurable cost savings across the public and private sectors.

Our product portfolio:
  - Water Clamp Sensor: non-intrusive water pipe monitoring (leak detection, consumption, freezing, stagnation, backflow)
  - Gas Clamp Sensor: non-intrusive gas pipe monitoring (leak detection, consumption, methane, appliance efficiency)
  - Electrical Panel Sensor: non-intrusive circuit-level electricity monitoring (load detection, ghost loads, safety, consumption)
  - SensePod: environmental condition sensor (flood detection, humidity, condensation, vibration, mold prevention)

Your job is to analyze a news article and extract structured information.

SECTORS (classify into exactly one):
{SECTOR_DESCRIPTIONS}

RELEVANCE SCORING (0.0 to 1.0):
  0.9-1.0: Directly mentions water/gas/electricity infrastructure problems, leaks, pipe monitoring, meter deployment, or building environmental hazards
  0.7-0.8: Mentions energy management systems, smart metering rollouts, utility digitization, building energy audits, or infrastructure monitoring market growth
  0.5-0.6: Mentions EU energy/water/environmental directives, ESG compliance, energy pricing impacts, or smart home/smart city initiatives that include monitoring
  0.3-0.4: Tangentially related — general infrastructure policy, municipal governance, broad smart city concepts without monitoring specifics
  0.0-0.2: Not related — urban planning without infrastructure monitoring, politics, sports, crime, military, entertainment, trade without infrastructure angle

EXAMPLES (learn from these):

Input: "Europe's digital water market forecast to double by 2033 as policy and technology drive transformation"
Summary: "Bluefield Research forecasts Europe's digital water market will double by 2033, driven by EU water policy mandates and smart metering/monitoring technology adoption."
Output: {{"country": null, "city": null, "sector": "water_infrastructure", "relevance_score": 0.90, "is_relevant": true, "reasoning": "Directly about digital water monitoring market growth in Europe, core to our Water Clamp Sensor market"}}

Input: "Smart electricity meter penetration rate in Europe reached 63% at end of 2024"
Summary: "Berg Insight reports smart electricity meter adoption across Europe reached 63%, driven by EU mandates and utility digitization."
Output: {{"country": null, "city": null, "sector": "electrical_infrastructure", "relevance_score": 0.75, "is_relevant": true, "reasoning": "Smart meter rollout creates adjacent market for our Electrical Panel Sensor's circuit-level monitoring"}}

Input: "Home Energy Management Systems in Europe reached 4.5 million in 2024"
Summary: "4.5M households now have HEMS, growing due to EV and heat pump adoption. Governments offering subsidies for residential energy management."
Output: {{"country": null, "city": null, "sector": "energy_efficiency", "relevance_score": 0.80, "is_relevant": true, "reasoning": "HEMS growth signals demand for granular energy monitoring — our Electrical Panel Sensor provides circuit-level consumption data that HEMS platforms need"}}

Input: "AIUT showcases smart utility IoT technologies at Enlit Europe 2025"
Summary: "Polish IoT company presents smart water and gas network monitoring technologies at Europe's largest energy trade fair."
Output: {{"country": "Spain", "city": "Bilbao", "sector": "water_infrastructure", "relevance_score": 0.85, "is_relevant": true, "reasoning": "Competitor showcasing smart water and gas monitoring solutions at major EU trade fair — directly relevant to our product space"}}

Input: "EU seeks better Spain-France energy links after blackout"
Summary: "EU pushes to boost energy interconnections between France and Spain after a massive blackout hit the Iberian Peninsula."
Output: {{"country": "Spain", "city": null, "sector": "electrical_infrastructure", "relevance_score": 0.60, "is_relevant": true, "reasoning": "Grid instability increases demand for building-level monitoring; our Electrical Panel Sensor helps track power quality and consumption during supply disruptions"}}

Input: "ESG in 2025: Developments and regulatory context in the EU"
Summary: "Covers EU ESG regulatory developments including CSRD reporting requirements, ECB sustainability initiatives, and ESG rating reforms."
Output: {{"country": null, "city": null, "sector": "environmental_compliance", "relevance_score": 0.50, "is_relevant": true, "reasoning": "ESG reporting requirements may drive demand for energy and water consumption monitoring data that our sensors provide"}}

Input: "The five-minute city: inside Denmark's revolutionary neighbourhood"
Summary: "Explores Denmark's urban planning concept where all daily needs are within a 5-minute walk, focusing on community design and livability."
Output: {{"country": "Denmark", "city": null, "sector": "not_relevant", "relevance_score": 0.15, "is_relevant": false, "reasoning": "Urban planning concept focused on walkability and community design, not infrastructure monitoring"}}

Input: "Electricity and gas prices across Europe: Which countries are the most expensive?"
Summary: "Compares electricity and gas prices across EU member states, highlighting price shifts since the energy crisis."
Output: {{"country": null, "city": null, "sector": "energy_efficiency", "relevance_score": 0.65, "is_relevant": true, "reasoning": "High energy costs drive demand for consumption monitoring — our Gas Clamp and Electrical Panel sensors help identify waste and reduce bills"}}

Input: "Europe Smart Home Market Forecasts 2024-2031"
Summary: "Market report on European smart home growth across lighting, speakers, security monitoring, and smart HVAC control."
Output: {{"country": null, "city": null, "sector": "energy_efficiency", "relevance_score": 0.55, "is_relevant": true, "reasoning": "Smart home market growth including HVAC control aligns with our SensePod environmental monitoring and Electrical Panel Sensor's consumption tracking"}}

RULES:
  - Extract the PRIMARY country and city mentioned in the article
  - If the article covers all of Europe or multiple countries, set country to null
  - If no specific city is mentioned, set city to null
  - Set is_relevant to true if relevance_score >= 0.4
  - Set is_relevant to false if relevance_score < 0.4
  - Consider market reports, competitor news, and technology adoption trends as RELEVANT — they signal where our products fit
  - Consider EU regulations and directives as RELEVANT when they touch energy, water, gas, or building efficiency
  - Consider general politics, urban design, social policy, sports, and entertainment as NOT RELEVANT

Respond with ONLY a JSON object in this exact format:
{{
  "country": "string or null",
  "city": "string or null",
  "sector": "one of the sector names above",
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
    result = await claude_json(
        prompt=prompt,
        system=SYSTEM_PROMPT,
        temperature=0.1,
        timeout=60.0,
        model_name=settings.claude_model,
        agent_name="classifier_agent_1",
        pipeline_id=state.get("pipeline_id")
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