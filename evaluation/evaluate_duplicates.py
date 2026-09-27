"""Evaluation of Cross-Article Deduplication Module.

Evaluates exact duplicate detection, near-duplicate detection (fuzzy SequenceMatcher),
and false positive rejection on a curated test bank.
"""
from __future__ import annotations
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.deduplication.deduplicate import is_duplicate


def evaluate_duplicates():
    # Test cases: (new_text, existing_pool, expected_duplicate_bool)
    test_cases = [
        ("Major flood closes Port of Rotterdam indefinitely", ["Major flood closes Port of Rotterdam indefinitely"], True),
        ("Major flood closes Port of Rotterdam indefinitely", ["Port of Rotterdam closed indefinitely due to major flooding"], True),
        ("Strike at factory halts automotive parts production", ["Strike at factory halts automotive parts production"], True),
        ("New semiconductor fab opens in Dresden Germany", ["Major flood closes Port of Rotterdam indefinitely"], False),
        ("Cyberattack targets logistics provider Maersk", ["Earthquake strikes warehouse in Tokyo"], False),
        ("Slight delay in cargo shipment due to customs check", ["Slight delay in cargo shipment due to customs check"], True),
        ("Workers strike at chemical manufacturing plant", ["Workers strike at chemical manufacturing facility"], True),
        ("Unrelated company announces quarterly profit earnings", ["Workers strike at chemical manufacturing plant"], False),
    ]

    correct = 0
    for text, pool, expected in test_cases:
        pred = is_duplicate(text, pool, threshold=0.70)
        if pred == expected:
            correct += 1

    accuracy = correct / len(test_cases)
    print("========================================")
    print("      DEDUPLICATION EVALUATION          ")
    print(f"  Test cases: {len(test_cases)}         ")
    print(f"  Deduplication Accuracy: {accuracy*100:.1f}%")
    print("========================================")
    return {"accuracy": accuracy, "test_cases": len(test_cases)}


if __name__ == "__main__":
    evaluate_duplicates()
