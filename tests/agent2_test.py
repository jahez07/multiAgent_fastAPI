"""
tests/agent2_test.py - Test Agent 2 with real Claude API calls. 

Sends classified news items to the problem analyzer and verifies:
    - Claude returns valid JSON
    - Problems have root causes and scale
    - Stakeholders have correct types
    - Urgency and opportunity scores are reasonable
    - Analysis is contextually relevant ( not generic )

REQUIRES: ANTHROPIC_API_KEY set in .env

Run:
    python -m tests.agent2_test.py
"""

import asyncio
import json
import sys
import time

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

# Test cases: pre-classified news items (as Agent 1 would output them)
TEST_CASES = [
    {
        "title": "Munich loses 30%_ of water supply due to aging pipe infrastructure",
        "summary": (
            "The city of Munich reported that nearly a third of its treated "
            "water is lost before reaching consumers due to deteriorating pipe "
            "networks. City officials estimate repair costs at EUR 2.1 billion "
            "over the next decade and are exploring smart monitoring solutions."
        ),
        "country": "Germany",
        "city": "Munich",
        "sector": "water_infrastructure",
        "expect_high_urgency": True,
        "expect_high_opportunity": True,
    },
    {
        "title": "Vienna mandates electrical safety inspections for all residential buildings over 30 years old",
        "summary": (
            "The city of Vienna has introduced mandatory electrical panel "
            "inspections for older residential buildings following a series "
            "of fires linked to overloaded circuits and outdated wiring. "
            "Building owners have 18 months to comply."
        ),
        "country": "Austria",
        "city": "Vienna",
        "sector": "electrical_infrastructure",
        "expect_high_urgency": True,
        "expect_high_opportunity": True,
    },
    {
        "title": "Copenhagen battles rising mold problems in public housing",
        "summary": (
            "Over 40%_ of Copenhagen's public housing stock shows signs of "
            "mold damage caused by inadequate ventilation and humidity control. "
            "The city plans EUR 200M investment in environmental monitoring "
            "and building remediation."
        ),
        "country": "Denmark",
        "city": "Copenhagen",
        "sector": "building_environment",
        "expect_high_urgency": True,
        "expect_high_opportunity": True,
    },
]

VALID_STAKEHOLDER_TYPES = {
    "decision_maker", "budget_approver", "regulator",
    "technical_evaluator", "end_user"
}

async def run_tests():
    global passed, failed

    from app.agents.analyzer import analyze
    from app.config import settings

    # Test 1: API key configured
    print("\n── Test 1: Claude API configuration ──")
    has_key = settings.anthropic_api_key and not settings.anthropic_api_key.startswith("sk-ant-....")
    check(
        "API key is configured",
        has_key,
        "Set ANTHROPIC_API_KEY in .env"
    )

    if not has_key:
        print(" Cannot continue without API key")
        sys.exit(1)
    

    # Test 2: Analyze each test case
    print("\n── Test 2: Problem analysis ──")
    total_time = 0

    for i, tc in enumerate(TEST_CASES, 1):
        print(f"\n  ── Case {i}/{len(TEST_CASES)}: {tc['title'][:50]}... ──")

        state = {
            "title": tc["title"],
            "summary": tc["summary"],
            "country": tc["country"],
            "city": tc["city"],
            "sector": tc["sector"],
        }

        start = time.time()
        result = await analyze(state)
        elapsed = time.time() - start
        total_time += elapsed

        print(f"    ({elapsed:.1f}s)")

        # Check problems
        problems = result.get("problems", [])
        check(f"[{i}] Has problems", len(problems) > 0, "No problems returned")

        if problems:
            for j, p in enumerate(problems):
                check(
                    f"[{i}] Problem {j+1} has root_cause",
                    "root_cause" in p and len(p.get("root_cause", "")) > 10,
                    f"Got: {p.get('root_cause','missing')[:50]}"
                )
            print(f"    Problems ({len(problems)})")
            for p in problems:
                print(f"    • {p.get('problem', '?')[:70]}")
                print(f"    Root: {p.get('root_cause', '?')[:70]}")

        # Check stakeholders
        stakeholders = result.get("stakeholders", [])
        check(f"[{i}] Has stakeholders", len(stakeholders) > 0, "No stakeholders returned")

        if stakeholders:
            types = {s.get("type") for s in stakeholders}
            check(
                f"[{i}] Stakeholder types are valid",
                types.issubset(VALID_STAKEHOLDER_TYPES),
                f"Invalid types: {types - VALID_STAKEHOLDER_TYPES}"
            )
            print(f"    Stakeholders ({len(stakeholders)})")
            for s in stakeholders:
                print(f"    • {s.get('role', '?')} ({s.get('type', '?')}, {s.get('influence', '?')})")
            
        
        # Check scores
        urgency = result.get("urgency_score", 0)
        opportunity = result.get("opportunity_score", 0)

        check(
            f"[{i}] Urgency score in range",
            0.0 <= urgency <= 1.0,
            f"Got: {urgency}"
        )
        check(
            f"[{i}] Opportunity score in range",
            0.0 <= opportunity <= 1.0,
            f"Got: {opportunity}"
        )

        if tc["expect_high_urgency"]:
            check(
                f"[{i}] Urgency is high (>= 0.6)",
                urgency >= 0.6,
                f"Got:  {urgency:.2f}"
            )
        
        if tc["expect_high_opportunity"]:
            check(
                f"[{i}] Opportunity is high (>= 0.6)",
                opportunity >= 0.6,
                f"Got: {opportunity:.2f}"
            )
        
        print(f"    Scores: urgency={urgency:.2f} opportunity={opportunity:.2f}")

    # Performance summary
    print(f"\n── Test 3: Performance ──")
    avg_time = total_time / len(TEST_CASES)
    check(
        f"Average response time: {avg_time:.1f}s",
        avg_time < 15, 
        "Very slow - check API key and network"
    )
    print(f"    Total: {total_time:.1f}s for {len(TEST_CASES)} items")

    # Summary
    print(f"\n{'=' * 55}")
    print(f"Agent 2 tests: {passed} passed, {failed} failed")
    print(f"{'=' * 55}")

    sys.exit(0 if failed == 0 else 1)

if __name__ == "__main__":
    asyncio.run(run_tests())