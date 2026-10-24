#!/usr/bin/env python3
from __future__ import annotations

import argparse

from io_utils import (
    CorpusStore, atomic_jsonl_writer, batched, iter_jsonl, passage_text, text_hash, write_jsonl_row,
)
from ranker_scorer import RankerScorer


def prepare_record(row: dict, store: CorpusStore, dense_keep: int) -> dict:
    candidates = row.get("candidates", [])
    docs = store.fetch_many([item["docid"] for item in candidates])
    positive_hashes = {text_hash(passage_text(item)) for item in row.get("positive_passages", [])}
    seen_ids, seen_hashes, prepared = set(), set(), []
    for item in candidates:
        docid = str(item["docid"])
        document = docs.get(docid)
        if document is None:
            continue
        digest = document["text_hash"]
        if docid in seen_ids or digest in seen_hashes or digest in positive_hashes:
            continue
        seen_ids.add(docid); seen_hashes.add(digest)
        prepared.append({
            "docid": docid,
            "title": "",
            "text": document["text"],
            "dense_score": float(item["dense_score"]),
            "dense_rank": int(item["dense_rank"]),
        })
        if len(prepared) == dense_keep:
            break
    return {**row, "candidates": prepared}


def main() -> None:
    parser = argparse.ArgumentParser(description="Select dense negatives that the frozen student scores highly.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--corpus-db", required=True)
    parser.add_argument("--student-model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="float16")
    parser.add_argument("--dense-keep", type=int, default=50)
    parser.add_argument("--student-keep", type=int, default=12)
    parser.add_argument("--query-buffer", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    scorer = RankerScorer(args.student_model, args.device, args.dtype)
    store = CorpusStore(args.corpus_db)
    count = 0
    try:
        with atomic_jsonl_writer(args.output) as writer:
            for rows in batched(iter_jsonl(args.input), args.query_buffer):
                rows = [prepare_record(row, store, args.dense_keep) for row in rows]
                pairs, owners = [], []
                for row_index, row in enumerate(rows):
                    for candidate_index, candidate in enumerate(row["candidates"]):
                        pairs.append((row["query"], candidate))
                        owners.append((row_index, candidate_index))
                scores = scorer.score(pairs, args.batch_size)
                for (row_index, candidate_index), score in zip(owners, scores):
                    rows[row_index]["candidates"][candidate_index]["student_score"] = score
                for row in rows:
                    row["candidates"] = sorted(
                        row["candidates"], key=lambda item: item["student_score"], reverse=True
                    )[: args.student_keep]
                    write_jsonl_row(writer, row)
                    count += 1
                print(f"student_reranked_queries={count}", flush=True)
    finally:
        store.close()
    print(f"done: {count} queries -> {args.output}")


if __name__ == "__main__":
    main()

