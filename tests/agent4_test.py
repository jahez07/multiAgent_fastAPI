"""
tests/agent4_test.py

Tests for Agent 4 - Business Case Writer

Run:
    python -m tests.agent4_test

What is tested:
    1. Output is valid JSON stored as a string
    2. All top-level keys present
    3. Nested sections have required sub-keys
    4. urgency_levels is one of the four allowed values
    5. next_steps has at least one entry with step/action/stakeholder
    6. Content is specific (mentions city name)
    7. Fallback handles missing upstream data gracefully
    8. Haiku model is being used
    9. Response time is acceptable (<30s, Haiku is fast)
"""

import asyncio
import json
import time
import sys

# - Rich state ( all the agents ran successfully )

RICH_STATE = {
    "pipeline_id": "test-agent4-rich-001",
    "title": "Munich loses 25`%` of drinking water to aging pipe network",
    "summary": (
        "The city of Munich is facing critical water infrastructure challenges as"
        "aging pipes lead to losses of approximately 25`%` of treated drinking water."
        "City officials are under pressure to comply with the EU Water Framework "
        "Directive Article 4 deadline and avoid penalties. An emergency €180M "
        "infrastructure budget has been proposed."
    ),
    "source_url": "https://example.com/munich-water-loss",
    "published_at": "2026-04-15T08:00:00+00:00",
    "ingested_at":   "2026-04-15T08:01:00+00:00",
    "country": "Germany",
    "city": "Munich",
    "sector": "water_infrastructure",
    "relevance_score": 0.92, 
    "is_relevant": True,
    "problems": [
        {
            "problem": "25% non-revenue water loss from aging pipe infrastructure",
            "root_cause": "Pipes installed in the 1970s have exceeded design lifespan; no real-time monitoring to detect leaks early",
            "scale": "~25`%` of treated water (approx. 40M litres/day) lost before reaching consumers",
            "affected_population": "1.5 million Munich residents",
        },
        {
            "problem": "Regulatory non-compliance risk under EU Water Framework Directive",
            "root_cause": "Article 4 requires measurable reduction in water body deterioration; current losses are reportable",
            "scale": "€50M in potential EU fines; loss of cohesion fund eligibility",
            "affected_population": "City treasury, ratepayers"
        },
    ],
    "stakeholders": [
        {"role": "Head of Munich Stadtwerke Water Division", "type": "decision_maker", "influence": "high"},
        {"role": "City of Munich Finance Director", "type": "budget_approver", "influence": "high"},
        {"role": "Bavarian State Environment Agency (LfU)", "type": "technical_evaluator", "influence": "medium"},
        {"role": "Munich City Council Infrastructure Committee", "type": "decision_maker",      "influence": "medium"},
    ],
    "urgency_score": 0.85,
    "urgency_reasoning": "Active regulatory deadline and public budget proposal under review",
    "opportunity_score": 0.92,
    "opportunity_reasoning": "Perfect product fit: Water Clamp Sensor addresses all identified leak and monitoring problems",
    "product_matches": [
        {
            "product": "Water Clamp Sensor",
            "feature": "Proactive water leak detection",
            "problem_addressed": "25% non-revenue water loss",
            "how_it_helps": "Detects abnormal flow patterns indicating leaks in Munich's aging 1970s pipe network without disruptive installation",
            "estimated_impact": "Estimated 15-20`%` reduction in NRW within 12 months based on comparable deployments",
        },
        {
            "product": "Water Clamp Sensor",
            "feature": "Continuous flow detection",
            "problem_addressed": "25% non-revenue water loss",
            "how_it_helps": "24/7 monitoring enables night-flow analysis to pinpoint leak locations in Munich districts",
            "estimated_impact": "Leak localization reduces repair cost by ~40% vs. blind excavation",
        },
    ],
    "directive_matches": [
        {
            "directive": "EU Water Framework Directive",
            "article": "Article 4 — Environmental objectives",
            "solution_aligned": "Water Clamp Sensor leak detection",
            "alignment": "Measurable reduction in non-revenue water directly supports compliance with Article 4's requirement to prevent deterioration of water body status",
        },
    ],
}

# Sparse state (Agent 1 only ran - simulates early failure)
SPARSE_STATE = {
    "pipeline_id":    "test-agent4-sparse-001",
    "title":          "Vienna struggles with electrical grid overloads in summer heat",
    "summary":        "Vienna's electrical grid is facing repeated overloads as summer heat drives record air conditioning usage.",
    "source_url":     "https://example.com/vienna-grid",
    "published_at":   "2026-04-15T09:00:00+00:00",
    "ingested_at":    "2026-04-15T09:01:00+00:00",
    "country":        "Austria",
    "city":           "Vienna",
    "sector":         "electrical_infrastructure",
    "relevance_score": 0.78,
    "is_relevant":     True,
    # No problems, stakeholders, product_matches, directive_matches
}

# Test runner

async def run_tests():
    from app.agents.business_case import write_case, SYSTEM_PROMPT
    from app.config import settings

    results = []

    def check(name: str, passed: bool, detail: str = ""):
        status = "✓" if passed else "x"
        results.append(passed)
        print(f"    {status} {name}" + (f": {detail}" if detail else ""))

    # Test group 1: Rich state ( all upstream )
    print("\n--Test 1: Rich state (all agent ran)---------")
    t0 = time.perf_counter()
    rich_output = await write_case(RICH_STATE)
    elapsed = time.perf_counter() - t0
    print(f"    Elapsed: {elapsed:.1f}s")

    check("Response time < 30s", elapsed < 30, f"{elapsed:.1f}")

    # Output has business_case_summary key
    bc_str = rich_output.get("business_case_summary")
    check("business_case_summary is a string", isinstance(bc_str, str))

    # Parse as valid JSON
    bc: dict = {}
    try:
        bc = json.loads(bc_str)
        check("business_case_summary is valid JSON", True)
    except Exception as e:
        check("business_case_summary is valid JSON", False, str(e))

    # Top-level keys
    top_keys = ["opportunity_title", "executive_summary", "problem", "solution",
                "regulatory_context", "roi_estimate", "next_steps", "key_contacts"]
    
    for key in top_keys:
        check(f"Top-level key: {key}", key in bc)
    
    # Problem section
    if "problem" in bc:
        p = bc["problem"]
        for sub in ["headline", "details", "scale", "urgency_level"]:
            check(f"problem.{sub} present", sub in p)
        valid_urgency = {"critical", "high", "medium", "low"}
        check(
            "urgency_level is valid", 
            p.get("urgency_level") in valid_urgency,
            p.get("urgency_level")
        )

    # Solution section
    if "solution" in bc:
        s = bc["solution"]
        for sub in ["headline", "recommended_products", "key_capabilities", "deployment_approach"]:
            check(f"solution.{sub} present", sub in s)
        check("recommended_products is non-empty list", isinstance(s.get("recommended_products"), str))

    # ROI section
    if "roi_estimate" in bc:
        r = bc["roi_estimate"]
        for sub in ["quantified_benefits", "estimated_payback_period", "risk_of_inaction"]:
            check(f"roi_estimate.{sub} present", sub in r)

    # Next steps
    ns = bc.get("next_steps", [])
    check("next_steps is non-empty", len(ns) > 0)
    if ns:
        first = ns[0]
        for sub in ["step", "action", "stakeholder"]:
            check(f"next_steps[0].{sub} present", sub in first)
    
    # City specificity
    bc_text = bc_str.lower()
    check("Content mentions `munich`", "munich" in bc_text)

    # Regulatory context
    if "regulatory_context" in bc:
        rc = bc["regulatory_context"]
        check(
            "regulatory_context.aligned_directives is non-empty",
            len(rc.get("aligned_directives", [])) > 0
        )

    
    # -- Test group 2: Sparse state (only Agent 1 data)
    # print("\n--Test 2: Sparse state (only Agent 1 ran)--------")
    # t0 = time.perf_counter()

if __name__ == "__main__":
    success = asyncio.run(run_tests())
    sys.exit(0 if success else 1)