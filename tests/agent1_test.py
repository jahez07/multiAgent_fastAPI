"""
tests/agent1_test.py - Test Agent 1 with real Ollama calls.

Sends a mix of relevant and irrelevant news items to the classifier
and verifies it correctly identifies sectors, extracts locations, 
and filters irrelevant articles.

REQUIRES: Ollama running with llama3.1:8b pulled
    ollama pull lamma3.1:8b

Run:
    python -m tests.agent1_test
"""

import asyncio
import sys
import time

passed = 0
failed = 0

def check(name, condition, detail= ""):
    global passed, failed
    if condition:
        print(f"    ✓ {name}")
        passed += 1
    else:
        print(f"    ✗ {name} — {detail}")
        failed += 1
    
# Test cases: a mix of relevant and irrelevant news
TEST_CASES = [
    # ── Should be RELEVANT ──
    {
        "title": "European Commission faces backlash over plans to fast-track polluting projects",
        "summary": "According to the Corporate Europe Observatory report, proposed EU legislation would speed up approval processes for projects labelled as “strategic” or of “overriding public interest,” which could in some cases allow them to bypass standard environmental review requirements.",
        "expect_relevant": False,
        "expect_sector": "not_relevant",
        "expect_country": "none",
    },
    {
        "title": "Barcelona mandates energy audits for all public buildings by 2027",
        "summary": "The Barcelona city council has passed a regulation requiring comprehensive energy audits for all municipal buildings, aligning with EU Energy Efficiency Directive targets.",
        "expect_relevant": True,
        "expect_sector": "energy_efficiency",
        "expect_country": "Spain",
    },
    {
        "title": "Lyon faces EUR 400M budget shortfall in infrastructure maintenance",
        "summary": "Lyon's municipal government disclosed a significant gap in its infrastructure maintenance budget, with deferred maintenance on water, energy, and transport systems creating growing risks.",
        "expect_relevant": True,
        "expect_sector": None,  # Could be water, municipal_budgets, etc.
        "expect_country": "France",
    },
    {
        "title": "Dutch gas network operator warns of methane leak risks in aging residential pipes",
        "summary": "The Netherlands' gas distribution network faces increasing safety risks as aging pipe infrastructure in residential areas shows signs of methane leakage.",
        "expect_relevant": True,
        "expect_sector": "gas_infrastructure",
        "expect_country": "Netherlands",
    },
    {
        "title": "Vienna mandates electrical safety inspections for all residential buildings over 30 years old",
        "summary": "The city of Vienna has introduced mandatory electrical panel inspections for older residential buildings following a series of fires linked to overloaded circuits and outdated wiring.",
        "expect_relevant": True,
        "expect_sector": "electrical_infrastructure",
        "expect_country": "Austria",
    },
    {
        "title": "Copenhagen battles rising mold problems in public housing due to poor ventilation",
        "summary": "Over 40% of Copenhagen's public housing stock shows signs of mold damage caused by inadequate ventilation and humidity control, prompting the city to invest in environmental monitoring.",
        "expect_relevant": True,
        "expect_sector": "building_environment",
        "expect_country": "Denmark",
    },
 
    # ── Should be NOT RELEVANT ──
    {
        "title": "How sunburn inspired a new way to store energy",
        "summary": "A former presidential tableware steward was found guilty of stealing pieces, his partner of helping sell them online.",
        "expect_relevant": False,
        "expect_sector": "not_relevant",
        "expect_country": None,  # Don't care
    },
    {
        "title": "'Don't swim' at 12 of 14 river bathing sites, as more locations announced",
        "summary": "Too much bacteria linked to faeces found at almost all England's designated river bathing sites",
        "expect_relevant": False,
        "expect_sector": "not_relevant",
        "expect_country": None,
    },
    {
        "title": "'We're living in a shed because of river pollution'",
        "summary": "Jane and Tony Coyle spent seven years waiting for planning permission due to River Lugg pollution.",
        "expect_relevant": False,
        "expect_sector": None,
        "expect_country": None,
    },
]


async def run_tests():
    global passed, failed

    from app.agents.classifier import classify

    # Test 1: Ollama connectivity
    print("\n Test 1: Ollama connectivity")
    try:
        import httpx
        from app.config import settings
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(f"{settings.ollama_base_url}/api/tags")
            models = [m["name"] for m in resp.json().get("models", [])]
            check("Ollama is reachable", resp.status_code == 200)
            has_model = any(settings.ollama_model in m for m in models)
            check(f"Model  '{settings.ollama_model}' is avaiable",
                  has_model,
                  f"Available: {models}")
    except Exception as e:
        print(f"    ✗ Cannot connect to Ollama: {e}")
        print(f"    Make sure Ollama is running: ollama serve")
        sys.exit(1)

    
    # Test 2: Classify each test case
    print("\n── Test 2: Classification accuracy ──")
    relevant_correct = 0
    relevant_total = 0
    total_time = 0

    for i, tc in enumerate(TEST_CASES, 1):
        state = {
            "title": tc["title"],
            "summary": tc["summary"],
            "country": None,
            "city": None,
        }

        start = time.time()
        result = await classify(state)
        elapsed = time.time() - start
        total_time += elapsed

        is_relevant = result.get('is_relevant', False)
        sector = result.get("sector", "?")
        country = result.get("country","?")
        score = result.get("relevance_score", 0)

        # Check relevance prediction
        expected_relevant = tc["expect_relevant"]
        relevance_match = is_relevant == expected_relevant
        if relevance_match:
            relevant_correct += 1
        relevant_total += 1

        # Print result
        icon = "✓" if relevance_match else "✗"
        print(f"\n  {icon} [{i}/{len(TEST_CASES)}] {tc['title'][:55]}...")
        print(f"      sector={sector}  score={score:.2f}  relevant={is_relevant}  ({elapsed:.1f}s)")
        print(f"      country={country}  city={result.get('city', '?')}")

        if not relevance_match:
            print(f"      EXPECTED relevant={expected_relevant}, GOT relevant={is_relevant}")
        
        # Check sector if we have an expectation
        if tc["expect_sector"] and tc["expect_sector"] != sector:
            print(f"      NOTE: expected sector={tc['expect_sector']}, got sector={sector}")
 
        # Check country if we have an expectation
        if tc["expect_country"] and tc["expect_country"] not in str(country):
            print(f"      NOTE: expected country containing '{tc['expect_country']}', got '{country}'")

    
    # ── Test 3: Performance summary ──
    print(f"\n── Test 3: Performance ──")
    accuracy = relevant_correct / relevant_total * 100
    avg_time = total_time / len(TEST_CASES)

    check(
        f"Relevancy accuracy: {accuracy:.0f}% ({relevant_correct}/{relevant_total})",
        accuracy >= 70,
        f"Below 70% - prompt may need tuning"
    )
    check(
        f"Average response time: {avg_time:.1f}s",
        avg_time < 10,
        f"Slow - model may stil be loading"
    )
    print(f"    Total time: {total_time:.1f}s for {len(TEST_CASES)} items")

    # ── Summary ──
    print(f"\n{'=' * 55}")
    print(f"Agent 1 tests: {passed} passed, {failed} failed")
    print(f"Relevance accuracy: {accuracy:.0f}%")
    if accuracy < 70:
        print("TIP: If accuracy is low, try:")
        print("  - Llama 3.1 70B instead of 8B (better reasoning)")
        print("  - Adjusting RELEVANCE_THRESHOLD in classifier.py")
        print("  - Refining sector descriptions in SECTORS dict")
    print(f"{'=' * 55}")
 
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    asyncio.run(run_tests())