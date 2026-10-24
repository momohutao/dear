#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter

from io_utils import iter_jsonl, passage_id, passage_text, text_hash


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate hard-negative training JSONL.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--expected-negatives", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    rows = 0
    sources = Counter()
    for row in iter_jsonl(args.input):
        positives = row.get("positive_passages", [])
        negatives = row.get("negative_passages", [])
        if not positives:
            raise ValueError(f"row {rows}: no positive passages")
        if len(negatives) != args.expected_negatives:
            raise ValueError(f"row {rows}: expected {args.expected_negatives} negatives, got {len(negatives)}")
        positive_ids = {passage_id(item) for item in positives}
        negative_ids = [passage_id(item) for item in negatives]
        if positive_ids.intersection(negative_ids):
            raise ValueError(f"row {rows}: positive/negative docid overlap")
        if len(set(negative_ids)) != len(negative_ids):
            raise ValueError(f"row {rows}: duplicate negative docids")
        hashes = [text_hash(passage_text(item)) for item in positives + negatives]
        if len(set(hashes)) != len(hashes):
            raise ValueError(f"row {rows}: duplicate positive/negative text")
        sources.update(item.get("source", "missing") for item in negatives)
        rows += 1
        if args.limit and rows >= args.limit:
            break
    print(f"PASS rows={rows} sources={dict(sources)}")


if __name__ == "__main__":
    main()

