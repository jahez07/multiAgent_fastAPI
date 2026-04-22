"""
tests/test_1_webhook.py — Test the FastAPI webhook.

Verifies:
  ✓ Health check endpoint works
  ✓ News items are accepted and queued
  ✓ Duplicate URLs are rejected
  ✓ Invalid API keys are rejected
  ✓ Malformed payloads return 422

Run:
  python tests/test_1_webhook.py
"""

import httpx
import json
import sys

WEBHOOK_URL = "http://192.168.2.185:8000/webhook/news"
API_KEY = "change-me-to-a-real-secret"  # Must match your .env

SAMPLE_NEWS = [
    # Item 1: Has country + city
    {
        "title": "Munich loses 30% of water supply due to aging pipe infrastructure",
        "content": (
            "The city of Munich reported that nearly a third of its treated "
            "water is lost before reaching consumers due to deteriorating pipe "
            "networks. City officials estimate repair costs at EUR 2.1 billion."
        ),
        "link": "https://example.com/test-munich-water-2026",
        "country": "Germany",
        "city": "Munich",
        "isoDate": "2026-04-15T10:30:00Z",
    },
    # Item 2: No country/city (Agent 1 will extract)
    {
        "title": "Barcelona mandates energy audits for all public buildings by 2027",
        "content": (
            "The Barcelona city council has passed a regulation requiring "
            "comprehensive energy audits for all municipal buildings."
        ),
        "link": "https://example.com/test-barcelona-energy-2026",
        "isoDate": "2026-04-14T08:15:00Z",
    },
    # Item 3: Bare minimum from RSS
    {
        "title": "Lyon faces EUR 400M budget shortfall in infrastructure maintenance",
        "content": (
            "Lyon's municipal government disclosed a significant gap in its "
            "infrastructure maintenance budget."
        ),
        "link": "https://example.com/test-lyon-budget-2026",
        "isoDate": "2026-04-13T14:45:00Z",
    },
]

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


def main():
    global passed, failed
    client = httpx.Client(timeout=10)

    # ── Test 1: Health check ──
    print("\n── Test 1: Health check ──")
    try:
        resp = client.get("http://192.168.2.185:8000/health")
        check("Returns 200", resp.status_code == 200)
        check("Redis connected", resp.json().get("redis") == "connected")
    except httpx.ConnectError:
        print("  ✗ Cannot connect to http://192.168.2.185:8000")
        print("    Make sure FastAPI is running: uvicorn app.main:app --port 8000")
        sys.exit(1)

    # ── Test 2: Send valid news items ──
    print("\n── Test 2: Send valid news items ──")
    pipeline_ids = []
    for i, news in enumerate(SAMPLE_NEWS, 1):
        resp = client.post(WEBHOOK_URL, json=news, headers={"X-API-Key": API_KEY})
        check(f"Item {i} accepted (202)", resp.status_code == 202)
        result = resp.json()
        check(f"Item {i} status=accepted", result.get("status") == "accepted")
        if "pipeline_id" in result:
            pipeline_ids.append(result["pipeline_id"])

    # ── Test 3: Duplicate rejection ──
    print("\n── Test 3: Duplicate rejection ──")
    resp = client.post(WEBHOOK_URL, json=SAMPLE_NEWS[0], headers={"X-API-Key": API_KEY})
    check("Returns 200 (not 202)", resp.status_code == 200)
    check("Status is duplicate", resp.json().get("status") == "duplicate")

    # ── Test 4: Invalid API key ──
    print("\n── Test 4: Invalid API key ──")
    resp = client.post(WEBHOOK_URL, json=SAMPLE_NEWS[0], headers={"X-API-Key": "wrong"})
    check("Returns 403", resp.status_code == 403)

    # ── Test 5: Malformed payload ──
    print("\n── Test 5: Malformed payload ──")
    resp = client.post(
        WEBHOOK_URL,
        json={"title": "Too short"},
        headers={"X-API-Key": API_KEY},
    )
    check("Returns 422", resp.status_code == 422)

    # ── Test 6: Queue stats ──
    print("\n── Test 6: Queue stats ──")
    resp = client.get("http://192.168.2.185:8000/queue/stats")
    stats = resp.json()
    check("Stream has items", stats.get("stream_length", 0) > 0)
    check("URLs tracked in dedup set", stats.get("unique_urls_seen", 0) > 0)

    # ── Summary ──
    print(f"\n{'=' * 50}")
    print(f"Results: {passed} passed, {failed} failed")
    if pipeline_ids:
        print(f"Pipeline IDs: {pipeline_ids}")
    print(f"{'=' * 50}")

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()