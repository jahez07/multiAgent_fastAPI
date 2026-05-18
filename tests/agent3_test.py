"""
tests/agent3_test.py - Test Agent 3 with real Qdrant + Claude.

Tests the full RAD pipeline:
    - Search queries are built correctly from problems
    - Qdrant returns relevant product features
    - Qdrant returns relevant directive articles
    - Claude maps features to problems with specific reasoning
    - Claude aligns solutions with directive articles

REQUIRES:
    - Qdrant running with products and directives ingested
    - ANTHRPIC_API_KEY set in .env
    - Ollama running with nomic-embed-text

Run:
    python -m tests.agent3_test
"""

import asyncio
import sys
import time

passed = 0
failed = 0

def check(name, condition, detail=""):
    global passed, failed
    if condition:
        print(f" ✓ {name}")
        passed += 1
    else:
        print(f" x {name} - {detail}")
        failed += 1


# Simulate state as it would arrive from Agent 1 + 2
TEST_CASES = [
    {
        "name": "Munich water loss",
        "state": {
            "title": "Munich loses 30`%` of water supply due to aging pipe infrastructure",
            "summary": "The city of Munich reported that nearly a third of its treated water is lost before reaching consumers.",
            "country": "Germany",
            "city": "Munich",
            "sector": "water_infrastructure",
            "problems": [
                {
                    "problem": "Munich is losing 30`%` of treated water due to deteriorating pipe networks",
                    "root_cause": "Aging infrastructure exceeding operational lifespan with no real-time monitoring",
                    "scale": "City-wide, EU 2.1 billion estimated repair costs",
                },
                {
                    "problem": "Lack of real-time visibility into pipe network performance",
                    "root_cause": "Absence of smart monitoring infrastructure",
                    "scale": "Entire municipal water distribution network",
                },
            ],
        },
        "expect_products": ["Water Clamp Sensor"],
        "expect_features": ["leak detection", "consumption", "flow"],
    },
    {
        "name": "Vienna electrical safety",
        "state":{
            "title": "Vienna mandates electrical safety inspections for all residential buildings over 30 years old",
            "summary": "Mandatory electrical panel inspections following fires linked to overloaded circuits.",
            "country": "Austria",
            "city": "Vienna",
            "sector": "electrical_infrastructure",
            "problems":[
                {
                    "problem": "Dangerous electrical panels in buildings over 30 years old",
                    "root_cause": "Aging electrical infrastructure not updated to modern safety standards",
                    "scale": "All residential buildings over 30 years old in Vienna",
                },
                {
                    "problem": "Fire incidents linked to overloaded circuits",
                    "root_cause": "Absence of continuous circuit-level monitoring",
                    "scale": "Multiple fire incidents across residential districts",
                },
            ],
        },
        "expect_products": ["Electrical Panel Sensor"],
        "expect_features": ["load", "circuit", "heat safety", "ghost"],
    },
    {
        "name": "Copenhagen mold",
        "state": {
            "title": "Copenhagen battles rising mold problems in public housing",
            "summary": "Over 40`%` of public housing shows mold damage from inadequate ventilation and humidity control.",
            "country": "Denmark",
            "city": "Copenhagen",
            "sector": "building_environment",
            "problems": [
                {
                    "problem": "Over 40`%` of public housing affected by mold damage",
                    "root_cause": "Insufficient environmental monitoring and ventilation",
                    "scale": "40`%` of public housing stock, EUR 200M planned investment",
                },
            ],
        },
        "expect_products": ["SensePod"],
        "expect_features": ["humidity", "dew point", "mold", "flood", "environmental"],
    }
]

async def run_tests():
    global passes, failed

    # Test 1: Qdrant Connectivity
    print("\nTest 1: Qdrant connectivity")
    try:
        from qdrant_client import QdrantClient
        from app.config import settings
        client = QdrantClient(url=settings.qdrant_url)
        collections = [c.name for c in client.get_collections().collections]
        check("Qdrant is reachable", True)
        check(
            "Product collection exists",
            settings.qdrant_products_collection in collections,
            f"Found: {collections}"
        )
        check(
            "Directives collection exists",
            settings.qdrant_directives_collection in collections,
            f"Found: {collections}"
        )
    except Exception as e:
        print(f"    x Cannot connect to Qdrant: {e}")
        print("     Make sure Qdrant is running and collections are ingested")
        sys.exit(1)

    # Test 2: RAG retrieval quality
    print("\nTest 2: RAG retrieval quality")
    from app.agents.solution_analyst import build_search_queries, retrieve_products, retrieve_directives

    for tc in TEST_CASES:
        print(f"\n -- {tc['name']} --")
        queries = build_search_queries(tc['state'])
        check(f"Built {len(queries)} queries", len(queries) > 0)

        products = await retrieve_products(queries, top_k_per_query=5)
        check(f"Retrieved {len(products)} product features", len(products) > 0)

        if products:
            product_names = {r["payload"].get("product", "") for r in products}
            expected = tc["expect_products"]
            has_expected = any(exp in product_names for exp in expected)
            check(
                f"Found expected product ({expected})",
                has_expected,
                f"Got: {product_names}"
            )

            print(f"    Top 3 products")
            for r in products:
                p = r["payload"]
                print(f"    [{r['score']:.2f}] {p.get('product')} - {p.get('feature')}")

        directives = await retrieve_directives(queries, top_k_per_query=3)
        check(f"Retrieved {len(directives)} directive articles", len(directives) > 0)

        if directives:
            print(f"    Top 2 directives")
            for r in directives[:2]:
                p = r["payload"]
                print(f"    [{r['score']}] {p.get('directive')} - {p.get('article')}")
            
    # Test 3: Full Agent 3 with Claude
    from app.agents.solution_analyst import solve

    print("\nTest 3: Full Agent 3 with Claude")

    total_time = 0

    for tc in TEST_CASES:
        print(f"\n -- {tc['name']} --")

        start = time.time()
        result = await solve(tc["state"])
        elapsed = time.time() - start
        total_time += elapsed
        print(f"    ({elapsed:.1f}s)")

        # Check product matches
        pm = result.get("product_matches", [])
        check(f"Has product matches", len(pm) > 0, "No matches returned")

        if pm:
            print(f"    Product matches ({len(pm)}):")
            for m in pm:
                product = m.get("product", "?")
                feature = m.get("feature", "?")
                how = m.get("how_it_helps", "?")[:80]
                print(f"    * [{product}] {feature}")
                print(f"      {how}...")

                # Check specificity - should mention the city or problem
                check(
                    f"Match is specific (not generic)",
                    len(m.get("how_it_helps", "")) > 50,
                    "Too short - may be generic"
                )

        # Check directive matches
        dm = result.get("directive_matches", [])
        check(f"Has directive matches", len(dm) > 0, "No directive matches")

        if dm:
            print(f"    Directive matches ({len(dm)}):")
            for m in dm:
                directive = m.get("directive", "?")
                article = m.get("article", "?")
                alignment = m.get("alignment", "?")[:80]
                print(f"    * {directive} - {article}")
                print(f"      {alignment}")

    # Performance
    print(f"\n Test 4: Performance")
    avg = total_time / len(TEST_CASES)
    print(f"    Agerage: {avg:.1f}s per item ({total_time:.1f}s total)")
    check("Average under 30s", avg < 30, f"Got {avg:.1f}")

    # Summary
    print(f"\n{'=' * 55}")
    print(f"Agent 3 tests: {passed} passed, {failed} failed")
    print(f"\n{'=' * 55}")

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    asyncio.run(run_tests())