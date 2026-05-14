"""
Agent 4 — Business Case Writer
Claude Haiku (cheaper model — writing, not deep reasoning)

Receives the fully enriched pipeline state from Agents 1–3 and synthesizes
a structured JSON business case ready for the NextJS dashboard.

Output schema:
{
  "opportunity_title": str,
  "executive_summary": str,
  "problem": {
    "headline": str,
    "details": str,
    "scale": str,
    "urgency_level": "critical" | "high" | "medium" | "low"
  },
  "solution": {
    "headline": str,
    "recommended_products": [str],
    "key_capabilities": [str],
    "deployment_approach": str
  },
  "regulatory_context": {
    "aligned_directives": [str],
    "compliance_benefit": str
  },
  "roi_estimate": {
    "quantified_benefits": [str],
    "estimated_payback_period": str,
    "risk_of_inaction": str
  },
  "next_steps": [
    {"step": int, "action": str, "stakeholder": str}
  ],
  "key_contacts": [str]
}
"""

import json
import logging
from pydantic import BaseModel, ValidationError

from app.agents.claude_client import claude_json
from app.config import settings

logger = logging.getLogger(__name__)


# ── Pydantic output models ────────────────────────────────────────────────────

class ProblemSection(BaseModel):
    headline: str
    details: str
    scale: str
    urgency_level: str                  # critical | high | medium | low


class SolutionSection(BaseModel):
    headline: str
    recommended_products: list[str]
    key_capabilities: list[str]
    deployment_approach: str


class RegulatorySection(BaseModel):
    aligned_directives: list[str]
    compliance_benefit: str


class RoiSection(BaseModel):
    quantified_benefits: list[str]
    estimated_payback_period: str
    risk_of_inaction: str


class NextStep(BaseModel):
    step: int
    action: str
    stakeholder: str


class BusinessCase(BaseModel):
    opportunity_title: str
    executive_summary: str
    problem: ProblemSection
    solution: SolutionSection
    regulatory_context: RegulatorySection
    roi_estimate: RoiSection
    next_steps: list[NextStep]
    key_contacts: list[str]


# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a senior B2B sales strategist writing business cases for an IoT sensor company targeting European municipalities, utilities, and building managers.

Your writing style is:
- Specific, not generic. Always mention the city, the numbers, the regulation.
- Outcome-focused. Decision makers care about cost savings, compliance deadlines, and political risk.
- Concise. Executive summaries are 2–3 sentences max.
- Action-oriented. Every business case ends with clear next steps assigned to real stakeholder roles.

You will receive structured intelligence gathered by three previous AI agents:
- Classified news with country/city/sector context
- Root cause analysis, stakeholder map, urgency and opportunity scores
- Matched product features and aligned EU directive articles

Your job: synthesize all of this into a compelling, accurate business case in JSON format.

Output ONLY valid JSON. No preamble, no explanation, no markdown fences."""


# ── Input formatting ──────────────────────────────────────────────────────────

def _urgency_label(score: float | None) -> str:
    """Convert numeric urgency score to label."""
    s = score or 0.5
    if s >= 0.85:
        return "critical"
    if s >= 0.65:
        return "high"
    if s >= 0.40:
        return "medium"
    return "low"


def _summarize_problems(problems: list | None) -> str:
    if not problems:
        return "No specific problems identified."
    lines = []
    for i, p in enumerate(problems[:3], 1):
        if isinstance(p, dict):
            lines.append(
                f"{i}. {p.get('problem', 'N/A')} "
                f"(Root cause: {p.get('root_cause', 'N/A')}; "
                f"Scale: {p.get('scale', 'N/A')})"
            )
    return "\n".join(lines)


def _summarize_stakeholders(stakeholders: list | None) -> str:
    if not stakeholders:
        return "No stakeholders identified."
    lines = []
    for s in stakeholders[:5]:
        if isinstance(s, dict):
            lines.append(
                f"- {s.get('role', '?')} [{s.get('type', '?')}] "
                f"(influence: {s.get('influence', '?')})"
            )
    return "\n".join(lines)


def _summarize_product_matches(matches: list | None) -> str:
    if not matches:
        return "No product matches found."
    lines = []
    for m in matches[:5]:
        if isinstance(m, dict):
            lines.append(
                f"- [{m.get('product', '?')}] {m.get('feature', '?')}: "
                f"{m.get('how_it_helps', '')} "
                f"(Impact: {m.get('estimated_impact', 'N/A')})"
            )
    return "\n".join(lines)


def _summarize_directive_matches(matches: list | None) -> str:
    if not matches:
        return "No directive alignments found."
    lines = []
    for m in matches[:3]:
        if isinstance(m, dict):
            lines.append(
                f"- {m.get('directive', '?')} {m.get('article', '')}: "
                f"{m.get('alignment', '')}"
            )
    return "\n".join(lines)


def build_prompt(state: dict) -> str:
    """Assemble the full prompt from pipeline state."""
    city                    = state.get("city")                     or state.get("country") or "the city"
    country                 = state.get("country")                  or "Europe"
    sector                  = state.get("sector")                   or "infrastructure"
    title                   = state.get("title")                    or "News event"
    summary                 = (state.get("summary")                 or "")[:500]
    urgency_score           = state.get("urgency_score")            or 0.5
    opportunity_score       = state.get("opportunity_score")        or 0.5
    urgency_reasoning       = state.get("urgency_reasoning")        or ""
    opportunity_reasoning   = state.get("opportunity_reasoning")    or ""

    problems                = state.get("problems")                 or []
    stakeholders            = state.get("stakeholders")             or []
    product_matches         = state.get("product_matches")          or []
    directive_matches       = state.get("directive_matches")        or []

    return f"""
Write a business case for the following sales opportunity.

=== NEWS EVENT ===
Title: {title}
City: {city}, {country}
Sector: {sector}
Summary: {summary}

=== AGENT INTELLIGENCE ===
Urgency score: {urgency_score:.2f}/1.00 — {urgency_reasoning}
Opportunity score: {opportunity_score:.2f}/1.00 — {opportunity_reasoning}

Problems identified:
{_summarize_problems(problems)}

Key stakeholders:
{_summarize_stakeholders(stakeholders)}

Matched product features:
{_summarize_product_matches(product_matches)}

EU directive alignments:
{_summarize_directive_matches(directive_matches)}

=== REQUIRED OUTPUT ===
Produce a JSON object with this exact schema:

{{
  "opportunity_title": "Short punchy title for this opportunity (max 10 words)",
  "executive_summary": "2-3 sentence summary. What happened, why it matters to us, what we offer.",
  "problem": {{
    "headline": "One sentence: the core problem",
    "details": "2-3 sentences of context: causes, consequences, timeline pressure",
    "scale": "Quantified scope if available (people affected, cost, volume)",
    "urgency_level": "one of: critical | high | medium | low"
  }},
  "solution": {{
    "headline": "One sentence: what we deploy and why",
    "recommended_products": ["product names only"],
    "key_capabilities": ["3-5 specific feature bullets relevant to this situation"],
    "deployment_approach": "1-2 sentences: how we would deploy in this specific context"
  }},
  "regulatory_context": {{
    "aligned_directives": ["Directive name + Article reference"],
    "compliance_benefit": "1-2 sentences: how our solution helps meet the requirement and by when"
  }},
  "roi_estimate": {{
    "quantified_benefits": ["3-4 specific benefit statements with numbers where possible"],
    "estimated_payback_period": "e.g. 12-18 months",
    "risk_of_inaction": "1-2 sentences: cost of doing nothing — financial, regulatory, political"
  }},
  "next_steps": [
    {{"step": 1, "action": "concrete action", "stakeholder": "role who does it"}},
    {{"step": 2, "action": "concrete action", "stakeholder": "role who does it"}},
    {{"step": 3, "action": "concrete action", "stakeholder": "role who does it"}}
  ],
  "key_contacts": ["List of stakeholder roles to target first, from the stakeholder map above"]
}}

Be specific to {city}. Mention the actual numbers and directives identified above.
Output ONLY the JSON object.
"""


# ── Main agent function ───────────────────────────────────────────────────────

async def write_case(state: dict) -> dict:
    """
    Agent 4 — Business Case Writer.
    Uses Claude Haiku (cheaper) since this is a writing task, not a reasoning task.
    """
    logger.info("Agent 4 starting for pipeline_id=%s", state.get("pipeline_id"))

    prompt = build_prompt(state)

    result = await claude_json(
        prompt=prompt,
        system=SYSTEM_PROMPT,
        max_tokens=1500,
        temperature=0.4,    # Slightly higher than other agents — writing benefits from a little variety
        timeout=60.0,
        model_name=settings.claude_haiku,        # ← Key difference: Haiku, not Sonnet
    )

    if result is None:
        logger.warning("Agent 4: Claude returned None — storing minimal fallback")
        business_case = _minimal_fallback(state)
    else:
        try:
            output        = BusinessCase.model_validate(result)
            business_case = output.model_dump()
            logger.info(
                "Agent 4 complete: '%s' with %d next steps",
                business_case.get("opportunity_title", ""),
                len(business_case.get("next_steps", [])),
            )
        except (ValidationError, Exception) as e:
            logger.error("Agent 4 Pydantic validation failed: %s", e)
            # Store whatever Claude returned even if it doesn't fully validate
            business_case = result if isinstance(result, dict) else _minimal_fallback(state)

    # Serialize to JSON string for storage in the TEXT column
    return {
        **state,
        "business_case_summary": json.dumps(business_case, ensure_ascii=False),
    }


def _minimal_fallback(state: dict) -> dict:
    """Return a minimal valid business case structure when Claude fails."""
    city   = state.get("city")   or state.get("country") or "city"
    sector = state.get("sector") or "infrastructure"
    return {
        "opportunity_title":  f"{sector.replace('_', ' ').title()} opportunity in {city}",
        "executive_summary":  "Analysis incomplete — Claude unavailable during writing step.",
        "problem": {
            "headline":      state.get("title", "See source news"),
            "details":       state.get("summary", "")[:300],
            "scale":         "See Agent 2 analysis",
            "urgency_level": _urgency_label(state.get("urgency_score")),
        },
        "solution": {
            "headline":              "IoT sensor deployment pending full analysis",
            "recommended_products":  [],
            "key_capabilities":      [],
            "deployment_approach":   "To be determined",
        },
        "regulatory_context": {
            "aligned_directives": [],
            "compliance_benefit": "To be determined",
        },
        "roi_estimate": {
            "quantified_benefits":      [],
            "estimated_payback_period": "To be determined",
            "risk_of_inaction":         "To be determined",
        },
        "next_steps":   [],
        "key_contacts": [],
    }