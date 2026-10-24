#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import faiss

from dense_encoder import DenseEncoder
from io_utils import atomic_jsonl_writer, batched, iter_jsonl, passage_id, write_jsonl_row


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieve dense candidates for MS MARCO train queries.")
    parser.add_argument("--train", required=True)
    parser.add_argument("--index", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="float16")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--search-extra", type=int, default=32)
    parser.add_argument("--nprobe", type=int, default=32)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    meta_path = Path(args.index).with_suffix(Path(args.index).suffix + ".meta.json")
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    encoder = DenseEncoder(
        metadata["model"], args.device, args.dtype, metadata["pooling"], int(metadata["max_length"])
    )
    if encoder.dimension != int(metadata["dimension"]):
        raise ValueError("Query encoder dimension does not match index")
    index = faiss.read_index(args.index)
    if hasattr(index, "nprobe"):
        index.nprobe = args.nprobe

    rows = iter_jsonl(args.train)
    if args.limit:
        rows = (row for number, row in enumerate(rows) if number < args.limit)
    written = 0
    with atomic_jsonl_writer(args.output) as writer:
        for group in batched(rows, args.batch_size):
            queries = [metadata["query_prefix"] + str(row["query"]) for row in group]
            vectors = encoder.encode(queries).numpy()
            scores, ids = index.search(vectors, args.top_k + args.search_extra)
            for row, row_scores, row_ids in zip(group, scores, ids):
                positives = row.get("positive_passages", [])
                positive_ids = {passage_id(item) for item in positives if passage_id(item)}
                candidates = []
                seen = set(positive_ids)
                for dense_rank, (docid, score) in enumerate(zip(row_ids, row_scores), 1):
                    if int(docid) < 0:
                        continue
                    key = str(int(docid))
                    if key in seen:
                        continue
                    seen.add(key)
                    candidates.append({
                        "docid": key,
                        "dense_score": float(score),
                        "dense_rank": dense_rank,
                    })
                    if len(candidates) == args.top_k:
                        break
                write_jsonl_row(writer, {
                    "query_id": str(row["query_id"]),
                    "query": row["query"],
                    "positive_passages": positives,
                    "candidates": candidates,
                })
                written += 1
            print(f"retrieved_queries={written}", flush=True)
    print(f"done: {written} queries -> {args.output}")


if __name__ == "__main__":
    main()

