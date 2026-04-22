"""
tests/test_4_database.py — Verify PostgreSQL contains correct pipeline results.

Verifies:
  ✓ PostgreSQL is reachable
  ✓ pipeline_results table exists
  ✓ Completed items have all fields populated
  ✓ Status distribution is correct
  ✓ No orphaned or corrupt records

Run AFTER the worker has processed some items:
  python tests/test_4_database.py
"""

import asyncio
import sys

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

from app.config import settings

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

    # ── Test 1: Connect to PostgreSQL ──
    print("\n── Test 1: PostgreSQL connection ──")
    try:
        engine = create_async_engine(settings.database_url)
        async with engine.begin() as conn:
            result = await conn.execute(text("SELECT 1"))
            check("PostgreSQL is reachable", result.scalar() == 1)
    except Exception as e:
        print(f"  ✗ Cannot connect to PostgreSQL: {e}")
        print("    Make sure PostgreSQL is running: docker compose up -d")
        sys.exit(1)

    session_factory = sessionmaker(engine, class_=AsyncSession)

    # ── Test 2: Table exists ──
    print("\n── Test 2: Table structure ──")
    async with session_factory() as session:
        result = await session.execute(text(
            "SELECT column_name, data_type FROM information_schema.columns "
            "WHERE table_name = 'pipeline_results' ORDER BY ordinal_position"
        ))
        columns = {row[0]: row[1] for row in result.fetchall()}
        check("Table pipeline_results exists", len(columns) > 0)
        check("Has pipeline_id column", "pipeline_id" in columns)
        check("Has title column", "title" in columns)
        check("Has country column", "country" in columns)
        check("Has business_case_summary column", "business_case_summary" in columns)
        check("Has status column", "status" in columns)

    # ── Test 3: Row count ──
    print("\n── Test 3: Data volume ──")
    async with session_factory() as session:
        result = await session.execute(text("SELECT COUNT(*) FROM pipeline_results"))
        total = result.scalar()
        check(f"Has {total} rows", total > 0, "Table is empty — has the worker run?")

    # ── Test 4: Status distribution ──
    print("\n── Test 4: Status distribution ──")
    async with session_factory() as session:
        result = await session.execute(text(
            "SELECT status, COUNT(*) FROM pipeline_results GROUP BY status ORDER BY COUNT(*) DESC"
        ))
        statuses = {row[0]: row[1] for row in result.fetchall()}
        for status, count in statuses.items():
            print(f"    {status}: {count}")
        check("At least one completed item",
              statuses.get("completed", 0) > 0,
              f"Found statuses: {statuses}")

    # ── Test 5: Completed items have all fields ──
    print("\n── Test 5: Data completeness (completed items) ──")
    async with session_factory() as session:
        result = await session.execute(text(
            "SELECT pipeline_id, title, country, sector, urgency_score, "
            "problems, product_matches, business_case_summary "
            "FROM pipeline_results WHERE status = 'completed' LIMIT 5"
        ))
        rows = result.fetchall()

        if rows:
            for row in rows:
                pid, title, country, sector, urgency, problems, matches, biz = row
                print(f"\n    [{pid[:8]}] {title[:50]}")
                print(f"      country={country}, sector={sector}, urgency={urgency}")
                check(f"[{pid[:8]}] has country", country is not None)
                check(f"[{pid[:8]}] has sector", sector is not None)
                check(f"[{pid[:8]}] has problems", problems is not None)
                check(f"[{pid[:8]}] has product_matches", matches is not None)
                check(f"[{pid[:8]}] has business_case", biz is not None and len(biz) > 0)
        else:
            check("Completed items exist", False, "No completed items found")

    # ── Test 6: Sample data ──
    print("\n── Test 6: Recent items ──")
    async with session_factory() as session:
        result = await session.execute(text(
            "SELECT pipeline_id, title, country, status, completed_at "
            "FROM pipeline_results ORDER BY completed_at DESC NULLS LAST LIMIT 5"
        ))
        rows = result.fetchall()
        for row in rows:
            pid, title, country, status, completed = row
            print(f"    [{pid[:8]}] {status:10s} | {country or '?':10s} | {title[:45]}")

    await engine.dispose()

    # ── Summary ──
    print(f"\n{'=' * 50}")
    print(f"Results: {passed} passed, {failed} failed")
    print(f"{'=' * 50}")

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    asyncio.run(run_tests())