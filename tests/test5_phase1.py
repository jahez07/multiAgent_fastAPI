"""
tests/test5_phase1.py — Full end-to-end test.

Sends a fresh news item to the webhook, waits for the worker
to process it, then verifies it appears in PostgreSQL with
all fields populated.

REQUIRES: FastAPI + Worker + Redis + PostgreSQL all running.

Run:
  python tests/test_5_end_to_end.py
"""

import asyncio
import sys
from uuid import uuid4

import httpx
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

from app.config import settings

WEBHOOK_URL = "http://192.168.2.185:8000/webhook/news"
API_KEY = "change-me-to-a-real-secret"

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


async def run_test():
    global passed, failed

    # Generate a unique URL so dedup doesn't block us
    unique_id = str(uuid4())[:8]
    test_url = f"https://example.com/e2e-test-{unique_id}"

    test_news = {
        "title": f"E2E Test: Helsinki water infrastructure needs EUR 500M upgrade ({unique_id})",
        "content": (
            "Helsinki's municipal water authority announced that the city's "
            "aging pipe network requires a comprehensive upgrade. Officials "
            "estimate the cost at EUR 500 million over the next decade."
        ),
        "link": test_url,
        "country": "Finland",
        "city": "Helsinki",
        "isoDate": "2026-04-20T09:00:00Z",
        "tags": ["water", "infrastructure", "e2e-test"],
    }

    # ── Step 1: Send to webhook ──
    print(f"\n── Step 1: Send test item ({unique_id}) ──")
    client = httpx.Client(timeout=10)
    try:
        resp = client.post(WEBHOOK_URL, json=test_news, headers={"X-API-Key": API_KEY})
        check("Webhook accepted (202)", resp.status_code == 202)
        result = resp.json()
        pipeline_id = result.get("pipeline_id")
        check("Got pipeline_id", pipeline_id is not None)
        print(f"    pipeline_id: {pipeline_id}")
    except httpx.ConnectError:
        print("  ✗ Cannot connect to webhook. Is FastAPI running?")
        sys.exit(1)

    if not pipeline_id:
        print("  ✗ No pipeline_id — cannot continue")
        sys.exit(1)

    # ── Step 2: Wait for worker to process ──
    print("\n── Step 2: Wait for worker to process ──")
    engine = create_async_engine(settings.database_url)
    session_factory = sessionmaker(engine, class_=AsyncSession)

    max_wait = 30  # seconds
    poll_interval = 2
    elapsed = 0
    row = None

    while elapsed < max_wait:
        async with session_factory() as session:
            result = await session.execute(
                text("SELECT status, country, sector, business_case_summary "
                     "FROM pipeline_results WHERE pipeline_id = :pid"),
                {"pid": pipeline_id},
            )
            row = result.fetchone()

        if row and row[0] in ("completed", "skipped", "failed"):
            print(f"    Found after {elapsed}s (status={row[0]})")
            break

        print(f"    Waiting... ({elapsed}s)")
        await asyncio.sleep(poll_interval)
        elapsed += poll_interval

    check("Item appeared in PostgreSQL", row is not None,
          f"Not found after {max_wait}s — is the worker running?")

    if not row:
        await engine.dispose()
        sys.exit(1)

    status, country, sector, biz_case = row

    # ── Step 3: Verify data completeness ──
    print("\n── Step 3: Verify data ──")
    check("Status is completed", status == "completed", f"Got: {status}")
    check("Country is Finland", country == "Finland", f"Got: {country}")
    check("Sector was set", sector is not None, "Sector is None")
    check("Business case written", biz_case is not None and len(biz_case) > 50,
          f"Length: {len(biz_case) if biz_case else 0}")

    # ── Step 4: Full row dump ──
    print("\n── Step 4: Full result ──")
    async with session_factory() as session:
        result = await session.execute(
            text("SELECT * FROM pipeline_results WHERE pipeline_id = :pid"),
            {"pid": pipeline_id},
        )
        columns = result.keys()
        row = result.fetchone()
        if row:
            for col, val in zip(columns, row):
                val_str = str(val)[:80] if val is not None else "NULL"
                print(f"    {col:30s} {val_str}")

    await engine.dispose()

    # ── Summary ──
    print(f"\n{'=' * 50}")
    print(f"End-to-end test: {passed} passed, {failed} failed")
    if failed == 0:
        print("Phase 1 is COMPLETE — pipeline works end-to-end!")
    print(f"{'=' * 50}")

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    asyncio.run(run_test())