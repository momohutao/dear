#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import random

from io_utils import (
    atomic_jsonl_writer, iter_jsonl, passage_id, passage_text, text_hash, write_jsonl_row,
)


def stable_rng(seed: int, query_id: str) -> random.Random:
    digest = hashlib.sha256(f"{seed}:{query_id}".encode()).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def choose_negatives(row: dict, mined: list[dict], mined_count: int, total: int, seed: int):
    positive_ids = {passage_id(item) for item in row.get("positive_passages", [])}
    positive_hashes = {text_hash(passage_text(item)) for item in row.get("positive_passages", [])}
    selected, seen_ids, seen_hashes = [], set(positive_ids), set(positive_hashes)

    def add(item: dict, source: str) -> bool:
        docid = passage_id(item)
        digest = text_hash(passage_text(item))
        if not docid or docid in seen_ids or digest in seen_hashes:
            return False
        copy = dict(item)
        copy["docid"] = docid
        copy.pop("id", None)
        copy["source"] = source
        selected.append(copy)
        seen_ids.add(docid); seen_hashes.add(digest)
        return True

    for item in mined:
        if sum(x["source"] == "dense_mined" for x in selected) >= mined_count:
            break
        add(item, "dense_mined")

    originals = list(row.get("negative_passages", []))
    stable_rng(seed, str(row["query_id"])).shuffle(originals)
    for item in originals:
        if len(selected) >= total:
            break
        add(item, "original")
    # If originals are scarce, use additional teacher-filtered mined negatives.
    for item in mined:
        if len(selected) >= total:
            break
        add(item, "dense_mined")
    if len(selected) < total:
        raise ValueError(
            f"query {row.get('query_id')} has only {len(selected)} unique safe negatives; need {total}"
        )
    stable_rng(seed + 1, str(row["query_id"])).shuffle(selected)
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description="Create an isolated 1-positive + 4-negative MS MARCO file.")
    parser.add_argument("--original-train", required=True)
    parser.add_argument("--mined", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mined-count", type=int, default=2)
    parser.add_argument("--total-negatives", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--only-mined-queries", action="store_true",
        help="Stop when the mined file ends; useful for a 100k-query pilot.",
    )
    args = parser.parse_args()
    if not 0 <= args.mined_count <= args.total_negatives:
        parser.error("mined-count must be between 0 and total-negatives")

    original_iter = iter_jsonl(args.original_train)
    mined_iter = iter_jsonl(args.mined)
    current_mined = next(mined_iter, None)
    written, with_mined = 0, 0
    with atomic_jsonl_writer(args.output) as writer:
        for row in original_iter:
            if current_mined is None and args.only_mined_queries:
                break
            query_id = str(row["query_id"])
            mined = []
            if current_mined is not None:
                mined_id = str(current_mined["query_id"])
                if mined_id != query_id:
                    raise ValueError(
                        f"Input order mismatch: original query {query_id}, mined query {mined_id}"
                    )
                mined = current_mined.get("mined_negatives", [])
                current_mined = next(mined_iter, None)
            negatives = choose_negatives(
                row, mined, args.mined_count, args.total_negatives, args.seed
            )
            if any(item.get("source") == "dense_mined" for item in negatives):
                with_mined += 1
            write_jsonl_row(writer, {
                "query_id": row["query_id"],
                "query": row["query"],
                "positive_passages": row["positive_passages"],
                "negative_passages": negatives,
            })
            written += 1
            if written % 10000 == 0:
                print(f"merged={written} queries_with_mined={with_mined}", flush=True)
    if current_mined is not None:
        raise ValueError("Mined file contains more rows than the selected original-train prefix")
    print(f"done: rows={written} rows_with_mined={with_mined} -> {args.output}")


if __name__ == "__main__":
    main()

