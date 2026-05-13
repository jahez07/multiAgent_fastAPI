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
import json
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
        "expected_products": ["Electrical Panel Sensor"],
        "expect_features": ["load", "circuit", "heat safety", "ghost"],
    },
    {
        "name": "Copenhagen mold",
        "state": {
            "title": "Cop"
        }
    }
]