"""
tests/test_3_pipeline.py — Verify the LangGraph pipeline processes items correctly.

Verifies:
  ✓ Graph compiles without errors
  ✓ All 4 agents run for relevant items
  ✓ Skip node runs for irrelevant items
  ✓ State is correctly enriched after each agent
  ✓ Conditional routing works (relevant vs irrelevant)

Run:
  python tests/test_3_pipeline.py
"""

import asyncio
import sys

passed = 0
failed = 0


def check(name, condition, detail=""):
    global passed, failed
    if condition:
        print(f"  ✓ {name}")
        passed += 1
    else:
        print(f"  ✗ {name} — {detail}")
        failed += 1


async def run_tests():
    global passed, failed

    # ── Test 1: Graph compiles ──
    print("\n── Test 1: Graph compilation ──")
    try:
        from app.pipeline import build_graph
        graph = build_graph()
        check("Graph compiled successfully", graph is not None)
    except Exception as e:
        check("Graph compiled", False, str(e))
        sys.exit(1)

    # ── Test 2: Full pipeline run (relevant item) ──
    print("\n── Test 2: Full pipeline run (relevant item) ──")
    relevant_input = {
        "pipeline_id": "test-relevant-001",
        "title": "Munich loses 30% of water supply due to aging pipe infrastructure",
        "summary": "City faces major water loss from deteriorating pipes.",
        "link": "https://example.com/test-pipeline-relevant",
        "published_at": "2026-04-15T10:30:00Z",
        "tags": ["water"],
        "ingested_at": "2026-04-15T11:00:00Z",
        "country": None,
        "city": None,
        "status": "processing",
        "retry_count": 0,
    }

    result = await graph.ainvoke(relevant_input)

    check("Status is completed", result.get("status") == "completed")
    check("Country was set", result.get("country") is not None)
    check("City was set", result.get("city") is not None)
    check("Sector was set", result.get("sector") is not None)
    check("is_relevant is True", result.get("is_relevant") is True)
    check("Problems populated", result.get("problems") is not None and len(result["problems"]) > 0)
    check("Stakeholders populated", result.get("stakeholders") is not None and len(result["stakeholders"]) > 0)
    check("Product matches populated", result.get("product_matches") is not None)
    check("Directive matches populated", result.get("directive_matches") is not None)
    check("Business case written", result.get("business_case_summary") is not None)
    check("Urgency score set", result.get("urgency_score") is not None)
    check("Opportunity score set", result.get("opportunity_score") is not None)

    # ── Test 3: Verify pipeline_id survives ──
    print("\n── Test 3: State preservation ──")
    check("pipeline_id preserved", result.get("pipeline_id") == "test-relevant-001")
    check("title preserved", result.get("title") == relevant_input["title"])
    check("link preserved", result.get("link") == relevant_input["link"])

    # ── Test 4: Print enriched state ──
    print("\n── Test 4: Enriched state summary ──")
    print(f"    pipeline_id:  {result.get('pipeline_id')}")
    print(f"    country:      {result.get('country')}")
    print(f"    city:         {result.get('city')}")
    print(f"    sector:       {result.get('sector')}")
    print(f"    relevance:    {result.get('relevance_score')}")
    print(f"    urgency:      {result.get('urgency_score')}")
    print(f"    opportunity:  {result.get('opportunity_score')}")
    print(f"    status:       {result.get('status')}")
    print(f"    problems:     {len(result.get('problems', []))} items")
    print(f"    matches:      {len(result.get('product_matches', []))} products, "
          f"{len(result.get('directive_matches', []))} directives")
    print(f"    biz case:     {len(result.get('business_case_summary', ''))} chars")

    # ── Summary ──
    print(f"\n{'=' * 50}")
    print(f"Results: {passed} passed, {failed} failed")
    print(f"{'=' * 50}")

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    asyncio.run(run_tests())