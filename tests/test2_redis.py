"""
tests/test_2_redis.py — Verify Redis streams are working correctly.

Verifies:
  ✓ Redis is reachable
  ✓ news:incoming stream exists and has items
  ✓ Consumer group exists with correct name
  ✓ Dedup set is tracking URLs
  ✓ Stream messages contain valid PipelineState JSON

Run:
  python tests/test_2_redis.py
"""

import json
import sys

import redis

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


def main():
    global passed, failed

    # ── Connect ──
    print("\n── Test 1: Redis connection ──")
    try:
        r = redis.from_url(settings.redis_url, decode_responses=True)
        r.ping()
        check("Redis is reachable", True)
    except Exception as e:
        print(f"  ✗ Cannot connect to Redis: {e}")
        sys.exit(1)

    # ── Check streams exist ──
    print("\n── Test 2: Stream existence ──")
    keys = r.keys("news:*")
    check("news:incoming exists", "news:incoming" in keys)
    check("news:seen_urls exists", "news:seen_urls" in keys)

    # ── Check stream length ──
    print("\n── Test 3: Stream content ──")
    stream_len = r.xlen("news:incoming")
    check(f"Stream has {stream_len} items", stream_len > 0, "Stream is empty")

    # ── Check consumer group ──
    print("\n── Test 4: Consumer group ──")
    try:
        groups = r.xinfo_groups("news:incoming")
        group_names = [g["name"] for g in groups]
        check(f"Consumer group exists: {group_names}",
              settings.consumer_group in group_names,
              f"Expected '{settings.consumer_group}', found {group_names}")

        if settings.consumer_group in group_names:
            group = next(g for g in groups if g["name"] == settings.consumer_group)
            print(f"    Entries read: {group.get('entries-read', '?')}")
            print(f"    Lag: {group.get('lag', '?')}")
            print(f"    Pending: {group.get('pending', '?')}")
    except Exception as e:
        check("Consumer group info", False, str(e))

    # ── Check dedup set ──
    print("\n── Test 5: Dedup set ──")
    seen_count = r.scard("news:seen_urls")
    check(f"Dedup set has {seen_count} URL hashes", seen_count > 0)

    # ── Validate message format ──
    print("\n── Test 6: Message format validation ──")
    entries = r.xrange("news:incoming", "-", "+", count=1)
    if entries:
        msg_id, fields = entries[0]
        check("Message has 'data' field", "data" in fields)

        if "data" in fields:
            try:
                state = json.loads(fields["data"])
                check("Data is valid JSON", True)
                check("Has pipeline_id", "pipeline_id" in state)
                check("Has news.title", "news" in state and "title" in state["news"])
                check("Has news.link", "news" in state and "link" in state["news"])
                print(f"    Sample: [{state.get('pipeline_id', '?')[:8]}] "
                      f"{state.get('news', {}).get('title', '?')[:50]}")
            except json.JSONDecodeError:
                check("Data is valid JSON", False, "Cannot parse JSON")
    else:
        check("At least one message exists", False, "Stream is empty")

    # ── Check dead letter queue ──
    print("\n── Test 7: Dead letter queue ──")
    dl_len = r.xlen("news:dead_letter")
    print(f"    Dead letter items: {dl_len}")
    if dl_len > 0:
        dl_entries = r.xrange("news:dead_letter", "-", "+", count=3)
        for msg_id, fields in dl_entries:
            error = fields.get("error", "unknown")[:80]
            print(f"    - {msg_id}: {error}")

    # ── Summary ──
    print(f"\n{'=' * 50}")
    print(f"Results: {passed} passed, {failed} failed")
    print(f"{'=' * 50}")

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()