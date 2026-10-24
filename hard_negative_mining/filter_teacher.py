#!/usr/bin/env python3
from __future__ import annotations

import argparse

from io_utils import atomic_jsonl_writer, batched, iter_jsonl, write_jsonl_row
from ranker_scorer import RankerScorer


def main() -> None:
    parser = argparse.ArgumentParser(description="Use a frozen teacher to remove likely false negatives.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--teacher-model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="float16")
    parser.add_argument("--min-teacher-margin", type=float, default=0.0)
    parser.add_argument("--mined-pool-size", type=int, default=8)
    parser.add_argument("--query-buffer", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    scorer = RankerScorer(args.teacher_model, args.device, args.dtype)
    count, accepted, rejected = 0, 0, 0
    with atomic_jsonl_writer(args.output) as writer:
        for rows in batched(iter_jsonl(args.input), args.query_buffer):
            pairs, layouts = [], []
            for row in rows:
                positives = row.get("positive_passages", [])
                candidates = row.get("candidates", [])
                if not positives:
                    raise ValueError(f"query {row.get('query_id')} has no positive passages")
                start = len(pairs)
                pairs.extend((row["query"], passage) for passage in positives)
                positive_end = len(pairs)
                pairs.extend((row["query"], passage) for passage in candidates)
                layouts.append((start, positive_end, len(pairs)))
            scores = scorer.score(pairs, args.batch_size)
            for row, (start, positive_end, end) in zip(rows, layouts):
                positive_scores = scores[start:positive_end]
                candidate_scores = scores[positive_end:end]
                reference = min(positive_scores)
                mined = []
                for candidate, teacher_score in zip(row.get("candidates", []), candidate_scores):
                    margin = float(reference - teacher_score)
                    candidate = dict(candidate)
                    candidate.update({
                        "teacher_score": float(teacher_score),
                        "teacher_positive_reference": float(reference),
                        "teacher_margin": margin,
                        "source": "dense_mined",
                    })
                    if margin >= args.min_teacher_margin:
                        mined.append(candidate); accepted += 1
                    else:
                        rejected += 1
                # Student score defines hardness; teacher is a safety filter only.
                mined.sort(key=lambda item: item["student_score"], reverse=True)
                write_jsonl_row(writer, {
                    "query_id": str(row["query_id"]),
                    "query": row["query"],
                    "positive_passages": row["positive_passages"],
                    "mined_negatives": mined[: args.mined_pool_size],
                })
                count += 1
            print(f"teacher_filtered_queries={count} accepted={accepted} rejected={rejected}", flush=True)
    print(f"done: queries={count} accepted={accepted} rejected={rejected} -> {args.output}")


if __name__ == "__main__":
    main()

