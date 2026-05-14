"""
Agent 4 — Business Case Writer
Claude Haiku (cheaper model — writing, not deep reasoning)

Design note on validation:
  We deliberately skip strict Pydantic validation here. The output is stored
  as a JSON blob (TEXT column). Light validation + repair only.
"""

import json
import logging

from app.agents.claude_client import claude_json
from app.config import settings

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are a senior B2B sales strategist writing business cases for an IoT sensor company targeting European municipalities, utilities, and building managers.

Your writing style:
- Specific, not generic. Always mention the city, the numbers, the regulation.
- Outcome-focused. Decision makers care about cost savings, compliance deadlines, and political risk.
- Concise. Executive summaries are 2-3 sentences max.
- Action-oriented. Every business case ends with clear next steps.

CRITICAL: Output ONLY a valid JSON object. No preamble. No explanation. No markdown fences.
The JSON keys must match EXACTLY the schema provided — do not rename any field."""


def _urgency_label(score):
    s = score or 0.5
    if s >= 0.85: return "critical"
    if s >= 0.65: return "high"
    if s >= 0.40: return "medium"
    return "low"


def _fmt_problems(problems):
    if not problems: return "No specific problems identified."
    lines = []
    for i, p in enumerate((problems or [])[:3], 1):
        if isinstance(p, dict):
            lines.append(f"{i}. {p.get('problem','N/A')} | Root cause: {p.get('root_cause','N/A')} | Scale: {p.get('scale','N/A')}")
    return "\n".join(lines)


def _fmt_stakeholders(stakeholders):
    if not stakeholders: return "No stakeholders identified."
    lines = []
    for s in (stakeholders or [])[:5]:
        if isinstance(s, dict):
            lines.append(f"- {s.get('role','?')} [{s.get('type','?')}]")
    return "\n".join(lines)


def _fmt_products(matches):
    if not matches: return "No product matches found."
    lines = []
    for m in (matches or [])[:5]:
        if isinstance(m, dict):
            lines.append(f"- [{m.get('product','?')}] {m.get('feature','?')}: {m.get('how_it_helps','')} (Impact: {m.get('estimated_impact','N/A')})")
    return "\n".join(lines)


def _fmt_directives(matches):
    if not matches: return "No directive alignments found."
    lines = []
    for m in (matches or [])[:3]:
        if isinstance(m, dict):
            lines.append(f"- {m.get('directive','?')} {m.get('article','')}: {m.get('alignment','')}")
    return "\n".join(lines)


def build_prompt(state):
    city              = state.get("city") or state.get("country") or "the city"
    country           = state.get("country") or "Europe"
    sector            = state.get("sector") or "infrastructure"
    title             = state.get("title") or "News event"
    summary           = (state.get("summary") or "")[:500]
    urgency_score     = state.get("urgency_score") or 0.5
    urgency_label     = _urgency_label(urgency_score)

    return f"""
=== INTELLIGENCE BRIEF ===
News: {title}
Location: {city}, {country}
Sector: {sector}
Summary: {summary}

Urgency score: {urgency_score:.2f}/1.00 ({urgency_label})
Opportunity score: {(state.get("opportunity_score") or 0.5):.2f}/1.00

Problems:
{_fmt_problems(state.get("problems"))}

Stakeholders:
{_fmt_stakeholders(state.get("stakeholders"))}

Matched product features:
{_fmt_products(state.get("product_matches"))}

EU directive alignments:
{_fmt_directives(state.get("directive_matches"))}

=== REQUIRED JSON OUTPUT ===

Return EXACTLY this JSON structure. Do not rename any key.

{{
  "opportunity_title": "Short punchy title, max 10 words",
  "executive_summary": "2-3 sentences: what happened, why it matters, what we offer",
  "problem": {{
    "headline": "One sentence: the core problem",
    "details": "2-3 sentences of context",
    "scale": "Quantified scope (people affected, cost, volume)",
    "urgency_level": "{urgency_label}"
  }},
  "solution": {{
    "headline": "One sentence: what we deploy and why",
    "recommended_products": ["product name as string", "another product name"],
    "key_capabilities": ["capability 1", "capability 2", "capability 3"],
    "deployment_approach": "1-2 sentences on how we deploy in this specific context"
  }},
  "regulatory_context": {{
    "aligned_directives": ["Directive name + Article, as a string"],
    "compliance_benefit": "1-2 sentences on how our solution meets the requirement"
  }},
  "roi_estimate": {{
    "quantified_benefits": ["benefit with number", "benefit with number", "benefit with number"],
    "estimated_payback_period": "e.g. 12-18 months",
    "risk_of_inaction": "1-2 sentences: cost of doing nothing"
  }},
  "next_steps": [
    {{"step": 1, "action": "Concrete action to take (this is a string describing what to do)", "stakeholder": "Role or person who does it (string)"}},
    {{"step": 2, "action": "Concrete action to take", "stakeholder": "Role or person who does it"}},
    {{"step": 3, "action": "Concrete action to take", "stakeholder": "Role or person who does it"}}
  ],
  "key_contacts": ["Stakeholder role 1", "Stakeholder role 2"]
}}

FIELD TYPES — strictly follow these:
- next_steps[].step: INTEGER (1, 2, 3)
- next_steps[].action: STRING — what to do (e.g. "Schedule a discovery call to present the ROI model")
- next_steps[].stakeholder: STRING — who does it (e.g. "Head of Munich Stadtwerke Water Division")
- All list fields contain strings, not objects

Be specific to {city}. Mention actual numbers and directives from the brief above.
Output ONLY the JSON object, nothing else.
"""


def _validate_and_repair(result, state):
    """Light validation — repair missing/misnamed keys. No Pydantic."""
    city   = state.get("city") or state.get("country") or "city"
    sector = state.get("sector") or "infrastructure"

    if "opportunity_title" not in result:
        result["opportunity_title"] = f"{sector.replace('_',' ').title()} opportunity in {city}"

    if "executive_summary" not in result:
        result["executive_summary"] = state.get("title", "See source news.")

    if not isinstance(result.get("problem"), dict):
        result["problem"] = {
            "headline":      state.get("title", "See source news"),
            "details":       (state.get("summary") or "")[:300],
            "scale":         "See Agent 2 analysis",
            "urgency_level": _urgency_label(state.get("urgency_score")),
        }
    else:
        if result["problem"].get("urgency_level") not in {"critical","high","medium","low"}:
            result["problem"]["urgency_level"] = _urgency_label(state.get("urgency_score"))

    if not isinstance(result.get("solution"), dict):
        result["solution"] = {
            "headline": "IoT sensor deployment pending full analysis",
            "recommended_products": [], "key_capabilities": [], "deployment_approach": "TBD",
        }

    if not isinstance(result.get("regulatory_context"), dict):
        result["regulatory_context"] = {"aligned_directives": [], "compliance_benefit": "TBD"}

    # Haiku sometimes calls this "roi_section" — rename it
    if "roi_estimate" not in result:
        result["roi_estimate"] = result.pop("roi_section", {
            "quantified_benefits": [], "estimated_payback_period": "TBD", "risk_of_inaction": "TBD"
        })

    # Repair next_steps: ensure step=int, action=str, stakeholder=str
    if not isinstance(result.get("next_steps"), list):
        result["next_steps"] = []
    else:
        repaired = []
        for i, ns in enumerate(result["next_steps"], 1):
            if not isinstance(ns, dict):
                continue
            repaired.append({
                "step":        i,
                "action":      str(ns.get("action") or ""),
                "stakeholder": str(ns.get("stakeholder") or ""),
            })
        result["next_steps"] = repaired

    if not isinstance(result.get("key_contacts"), list):
        result["key_contacts"] = []

    return result


def _minimal_fallback(state):
    city   = state.get("city") or state.get("country") or "city"
    sector = state.get("sector") or "infrastructure"
    return {
        "opportunity_title":   f"{sector.replace('_',' ').title()} opportunity in {city}",
        "executive_summary":   "Analysis incomplete — Claude unavailable during writing step.",
        "problem": {
            "headline":      state.get("title", "See source news"),
            "details":       (state.get("summary") or "")[:300],
            "scale":         "See Agent 2 analysis",
            "urgency_level": _urgency_label(state.get("urgency_score")),
        },
        "solution": {
            "headline": "IoT sensor deployment pending full analysis",
            "recommended_products": [], "key_capabilities": [], "deployment_approach": "TBD",
        },
        "regulatory_context": {"aligned_directives": [], "compliance_benefit": "TBD"},
        "roi_estimate": {
            "quantified_benefits": [], "estimated_payback_period": "TBD", "risk_of_inaction": "TBD"
        },
        "next_steps":   [],
        "key_contacts": [],
    }


async def write_case(state: dict) -> dict:
    """
    Agent 4 — Business Case Writer.
    Uses Claude Haiku. Always returns with business_case_summary set (never None).
    """
    logger.info("Agent 4 starting for pipeline_id=%s", state.get("pipeline_id"))

    business_case = {}

    try:
        result = await claude_json(
            prompt=build_prompt(state),
            system=SYSTEM_PROMPT,
            max_tokens=1500,
            temperature=0.4,
            timeout=60.0,
            model_name=settings.claude_haiku,
        )

        if not isinstance(result, dict):
            logger.warning("Agent 4: Claude returned None or non-dict — using fallback")
            business_case = _minimal_fallback(state)
        else:
            business_case = _validate_and_repair(result, state)
            logger.info(
                "Agent 4 complete: '%s' | urgency=%s | next_steps=%d",
                business_case.get("opportunity_title", ""),
                business_case.get("problem", {}).get("urgency_level", "?"),
                len(business_case.get("next_steps", [])),
            )

    except Exception as e:
        logger.error("Agent 4 unexpected error: %s", e)
        business_case = _minimal_fallback(state)

    try:
        bc_json = json.dumps(business_case, ensure_ascii=False)
    except Exception as e:
        logger.error("Agent 4 json.dumps failed: %s", e)
        bc_json = json.dumps(_minimal_fallback(state), ensure_ascii=False)

    return {**state, "business_case_summary": bc_json}