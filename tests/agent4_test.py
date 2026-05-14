"""
tests/test_agent4_writer.py — Agent 4 Business Case Writer tests

Run:
    python -m tests.test_agent4_writer
"""

import asyncio, json, time, sys

RICH_STATE = {
    "pipeline_id":    "test-agent4-rich-001",
    "title":          "Munich loses 25% of drinking water to aging pipe network",
    "summary": (
        "The city of Munich is facing critical water infrastructure challenges as "
        "aging pipes lead to losses of approximately 25% of treated drinking water. "
        "City officials are under pressure to comply with the EU Water Framework "
        "Directive Article 4 deadline. An emergency EUR 180M infrastructure budget has been proposed."
    ),
    "source_url":     "https://example.com/munich-water-loss",
    "published_at":   "2026-04-15T08:00:00+00:00",
    "ingested_at":    "2026-04-15T08:01:00+00:00",
    "country":        "Germany",
    "city":           "Munich",
    "sector":         "water_infrastructure",
    "relevance_score":   0.92,
    "is_relevant":       True,
    "urgency_score":     0.85,
    "urgency_reasoning": "Active regulatory deadline and public budget proposal under review",
    "opportunity_score": 0.92,
    "opportunity_reasoning": "Perfect product fit: Water Clamp Sensor addresses all identified problems",
    "problems": [
        {
            "problem":    "25% non-revenue water loss from aging pipe infrastructure",
            "root_cause": "Pipes installed in the 1970s have exceeded design lifespan; no real-time monitoring",
            "scale":      "~25% of treated water (approx. 40M litres/day) lost before reaching consumers",
            "affected_population": "1.5 million Munich residents",
        },
        {
            "problem":    "Regulatory non-compliance risk under EU Water Framework Directive",
            "root_cause": "Article 4 requires measurable reduction in water body deterioration",
            "scale":      "EUR 50M in potential EU fines; loss of cohesion fund eligibility",
            "affected_population": "City treasury, ratepayers",
        },
    ],
    "stakeholders": [
        {"role": "Head of Munich Stadtwerke Water Division",     "type": "decision_maker",      "influence": "high"},
        {"role": "City of Munich Finance Director",              "type": "budget_approver",     "influence": "high"},
        {"role": "Bavarian State Environment Agency (LfU)",      "type": "regulator",           "influence": "high"},
        {"role": "Munich Infrastructure Technical Lead",         "type": "technical_evaluator", "influence": "medium"},
    ],
    "product_matches": [
        {
            "product":           "Water Clamp Sensor",
            "feature":           "Proactive water leak detection",
            "problem_addressed": "25% non-revenue water loss",
            "how_it_helps":      "Detects abnormal flow patterns in Munich's aging 1970s network without disruptive installation",
            "estimated_impact":  "Estimated 15-20% NRW reduction within 12 months",
        },
    ],
    "directive_matches": [
        {
            "directive":       "EU Water Framework Directive",
            "article":         "Article 4 — Environmental objectives",
            "solution_aligned": "Water Clamp Sensor leak detection",
            "alignment":       "Measurable NRW reduction directly supports Article 4 compliance",
        },
    ],
}

SPARSE_STATE = {
    "pipeline_id":    "test-agent4-sparse-001",
    "title":          "Vienna struggles with electrical grid overloads in summer heat",
    "summary":        "Vienna's electrical grid faces repeated overloads as summer heat drives record AC usage.",
    "source_url":     "https://example.com/vienna-grid",
    "published_at":   "2026-04-15T09:00:00+00:00",
    "ingested_at":    "2026-04-15T09:01:00+00:00",
    "country":        "Austria",
    "city":           "Vienna",
    "sector":         "electrical_infrastructure",
    "relevance_score": 0.78,
    "is_relevant":     True,
    # No problems/stakeholders/product_matches/directive_matches
}


async def run_tests():
    from app.agents.business_case import write_case
    from app.config import settings

    results = []

    def check(name, passed, detail=""):
        status = "✓" if passed else "✗"
        results.append(passed)
        print(f"  {status} {name}" + (f": {detail}" if detail else ""))

    # ── Test 1: Rich state ────────────────────────────────────────────────────
    print("\n── Test 1: Rich state (all agents ran) ──────────────────────────────")
    t0 = time.perf_counter()
    rich_output = await write_case(RICH_STATE)
    elapsed = time.perf_counter() - t0
    print(f"  Elapsed: {elapsed:.1f}s")

    check("Response time < 30s", elapsed < 30, f"{elapsed:.1f}s")

    bc_str = rich_output.get("business_case_summary")
    check("business_case_summary is a string", isinstance(bc_str, str), type(bc_str).__name__)

    bc = {}
    if isinstance(bc_str, str):
        try:
            bc = json.loads(bc_str)
            check("business_case_summary is valid JSON", True)
        except Exception as e:
            check("business_case_summary is valid JSON", False, str(e))
    else:
        check("business_case_summary is valid JSON", False, "bc_str is None")

    # Top-level keys
    top_keys = ["opportunity_title","executive_summary","problem","solution",
                "regulatory_context","roi_estimate","next_steps","key_contacts"]
    for key in top_keys:
        check(f"Top-level key: {key}", key in bc)

    # Problem section
    problem = bc.get("problem", {})
    for sub in ["headline","details","scale","urgency_level"]:
        check(f"problem.{sub} present", sub in problem)
    check("urgency_level is valid",
          problem.get("urgency_level") in {"critical","high","medium","low"},
          problem.get("urgency_level"))

    # Solution section
    solution = bc.get("solution", {})
    check("solution.recommended_products is non-empty list",
          isinstance(solution.get("recommended_products"), list) and len(solution.get("recommended_products",[])) > 0)
    check("solution.key_capabilities is non-empty list",
          isinstance(solution.get("key_capabilities"), list) and len(solution.get("key_capabilities",[])) > 0)

    # ROI section
    roi = bc.get("roi_estimate", {})
    for sub in ["quantified_benefits","estimated_payback_period","risk_of_inaction"]:
        check(f"roi_estimate.{sub} present", sub in roi)

    # Next steps
    ns = bc.get("next_steps", [])
    check("next_steps is non-empty", len(ns) > 0)
    if ns:
        first = ns[0]
        check("next_steps[0].step is int", isinstance(first.get("step"), int), str(first.get("step")))
        check("next_steps[0].action is str", isinstance(first.get("action"), str))
        check("next_steps[0].stakeholder is str", isinstance(first.get("stakeholder"), str))

    # City specificity
    if isinstance(bc_str, str):
        check("Content mentions 'munich'", "munich" in bc_str.lower())

    # Regulatory context
    rc = bc.get("regulatory_context", {})
    check("regulatory_context.aligned_directives is non-empty",
          len(rc.get("aligned_directives", [])) > 0)

    # ── Test 2: Sparse state ──────────────────────────────────────────────────
    print("\n── Test 2: Sparse state (only Agent 1 ran) ──────────────────────────")
    t0 = time.perf_counter()
    sparse_output = await write_case(SPARSE_STATE)
    elapsed = time.perf_counter() - t0
    print(f"  Elapsed: {elapsed:.1f}s")

    sp_str = sparse_output.get("business_case_summary")
    check("Sparse: business_case_summary is a string", isinstance(sp_str, str))

    if isinstance(sp_str, str):
        try:
            sp = json.loads(sp_str)
            check("Sparse: valid JSON", True)
            check("Sparse: opportunity_title present", bool(sp.get("opportunity_title")))
            check("Sparse: urgency_level present", bool(sp.get("problem", {}).get("urgency_level")))
            check("Sparse: mentions 'vienna'", "vienna" in sp_str.lower())
        except Exception as e:
            check("Sparse: valid JSON", False, str(e))
    else:
        check("Sparse: valid JSON", False, "sp_str is None")

    # ── Test 3: State passthrough ─────────────────────────────────────────────
    print("\n── Test 3: State passthrough ────────────────────────────────────────")
    check("pipeline_id preserved", rich_output.get("pipeline_id") == "test-agent4-rich-001")
    check("city preserved", rich_output.get("city") == "Munich")
    check("sector preserved", rich_output.get("sector") == "water_infrastructure")

    # ── Test 4: Model ─────────────────────────────────────────────────────────
    print("\n── Test 4: Model ────────────────────────────────────────────────────")
    check("HAIKU constant is set", bool(settings.claude_haiku))
    check("HAIKU contains 'haiku'", "haiku" in settings.claude_haiku.lower(), settings.claude_haiku)
    print(f"  Using model: {settings.claude_haiku}")

    # ── Summary ───────────────────────────────────────────────────────────────
    passed = sum(results)
    total  = len(results)
    print(f"\n{'='*60}")
    print(f"Agent 4 results: {passed}/{total} passed")
    if passed == total:
        print("Agent 4 is COMPLETE ✓")
    else:
        print(f"Failed: {total - passed} checks")
    print(f"{'='*60}")
    return passed == total


if __name__ == "__main__":
    success = asyncio.run(run_tests())
    sys.exit(0 if success else 1)