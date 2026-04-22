"""
test_webhook.py — Simulate n8n sending news items to the webhook.

Run the FastAPI app first:
  uvicorn app.main:app --port 8000 --reload

Then in another terminal:
  python test_webhook.py

This sends 3 sample news items that represent real-world scenarios
your pipeline will encounter.
"""

import httpx
import json

WEBHOOK_URL = "http://localhost:8000/webhook/news"
API_KEY = "change-me-to-a-real-secret"  # Must match your .env

# ──────────────────────────────────────────────────────────────
# Sample news items — these simulate what your n8n RSS feed would produce
# ──────────────────────────────────────────────────────────────

SAMPLE_NEWS = [
    {
        "title": "Munich loses 30% of water supply due to aging pipe infrastructure",
        "summary": (
            "The city of Munich reported that nearly a third of its treated "
            "water is lost before reaching consumers due to deteriorating pipe "
            "networks. City officials estimate repair costs at EUR 2.1 billion "
            "over the next decade and are exploring smart monitoring solutions."
        ),
        "source_url": "https://example.com/munich-water-loss-2026",
        "country": "Germany",
        "city": "Munich",
        "published_at": "2026-04-15T10:30:00Z",
        "tags": ["water", "infrastructure", "leak"],
    },
    {
        "title": "Barcelona mandates energy audits for all public buildings by 2027",
        "summary": (
            "The Barcelona city council has passed a regulation requiring "
            "comprehensive energy audits for all municipal buildings. The move "
            "aligns with EU Energy Efficiency Directive targets and is expected "
            "to reduce energy consumption by 20% across city-owned properties."
        ),
        "source_url": "https://example.com/barcelona-energy-audits-2026",
        "country": "Spain",
        "city": "Barcelona",
        "published_at": "2026-04-14T08:15:00Z",
        "tags": ["energy", "regulation", "buildings"],
    },
    {
        "title": "Lyon faces EUR 400M budget shortfall in infrastructure maintenance",
        "summary": (
            "Lyon's municipal government disclosed a significant gap in its "
            "infrastructure maintenance budget, with deferred maintenance on "
            "water, energy, and transport systems creating growing risks. "
            "The city is seeking technology partners to reduce costs through "
            "predictive analytics and anomaly detection."
        ),
        "source_url": "https://example.com/lyon-budget-shortfall-2026",
        "country": "France",
        "city": "Lyon",
        "published_at": "2026-04-13T14:45:00Z",
        "tags": ["budget", "infrastructure", "analytics"],
    },
]


def main():
    print("=" * 60)
    print("Testing webhook with sample news items")
    print("=" * 60)

    client = httpx.Client(timeout=10)

    # First check health
    try:
        resp = client.get("http://localhost/health")
        print(f"\nHealth check: {resp.json()}")
    except httpx.ConnectError:
        print("\nERROR: Cannot connect to http://localhost:8000")
        print("Make sure the FastAPI app is running:")
        print("  uvicorn app.main:app --port 8000 --reload")
        return

    # Send each news item
    for i, news in enumerate(SAMPLE_NEWS, 1):
        print(f"\n{'─' * 60}")
        print(f"Sending item {i}/{len(SAMPLE_NEWS)}: {news['title'][:50]}...")

        resp = client.post(
            WEBHOOK_URL,
            json=news,
            headers={"X-API-Key": API_KEY},
        )

        result = resp.json()
        print(f"  Status: {resp.status_code}")
        print(f"  Response: {json.dumps(result, indent=2)}")

    # Send one duplicate to test dedup
    print(f"\n{'─' * 60}")
    print("Sending duplicate of item 1 (should be rejected)...")
    resp = client.post(
        WEBHOOK_URL,
        json=SAMPLE_NEWS[0],
        headers={"X-API-Key": API_KEY},
    )
    print(f"  Status: {resp.status_code}")
    print(f"  Response: {json.dumps(resp.json(), indent=2)}")

    # Test invalid API key
    print(f"\n{'─' * 60}")
    print("Sending with wrong API key (should get 403)...")
    resp = client.post(
        WEBHOOK_URL,
        json=SAMPLE_NEWS[0],
        headers={"X-API-Key": "wrong-key"},
    )
    print(f"  Status: {resp.status_code}")

    # Check queue stats
    print(f"\n{'─' * 60}")
    resp = client.get("http://localhost:8000/queue/stats")
    print(f"Queue stats: {json.dumps(resp.json(), indent=2)}")

    print(f"\n{'=' * 60}")
    print("Done! Check the FastAPI logs to see the items being queued.")
    print("Start the worker to process them:")
    print("  python -m app.worker")
    print("=" * 60)


if __name__ == "__main__":
    main()