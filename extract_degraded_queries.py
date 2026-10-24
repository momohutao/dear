#!/usr/bin/env python3
"""Extract DL19/DL20 queries whose nDCG@10 drops from DeAR to DeAR+KK.

The output is one JSON file containing query text, per-query scores, and both
systems' top-10 documents (including qrels and document text).
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Tuple


RunEntry = Tuple[int, str, float]


def read_run(path: Path) -> Dict[str, List[RunEntry]]:
    """Read a six-column TREC run or a three-column qid/docid/score file."""
    by_query: MutableMapping[str, List[RunEntry]] = defaultdict(list)
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, 1):
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split()
            try:
                if len(parts) >= 6:
                    qid, docid = parts[0], parts[2]
                    rank, score = int(parts[3]), float(parts[4])
                elif len(parts) == 3:
                    qid, docid, score_text = parts
                    rank = len(by_query[qid]) + 1
                    score = float(score_text)
                else:
                    raise ValueError("expected 3 or at least 6 columns")
            except ValueError as exc:
                raise ValueError(f"{path}:{line_number}: malformed run line: {line}") from exc
            by_query[qid].append((rank, docid, score))

    result: Dict[str, List[RunEntry]] = {}
    for qid, entries in by_query.items():
        # Keep the first occurrence of a document after sorting by the declared rank.
        seen = set()
        unique = []
        for rank, docid, score in sorted(entries, key=lambda item: (item[0], -item[2])):
            if docid in seen:
                continue
            seen.add(docid)
            unique.append((rank, docid, score))
        result[qid] = unique
    return result


def read_qrels(path: Path) -> Dict[str, Dict[str, float]]:
    qrels: MutableMapping[str, Dict[str, float]] = defaultdict(dict)
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, 1):
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 4:
                raise ValueError(f"{path}:{line_number}: malformed qrels line: {line}")
            qid, docid = parts[0], parts[2]
            try:
                relevance = float(parts[3])
            except ValueError as exc:
                raise ValueError(f"{path}:{line_number}: invalid relevance: {line}") from exc
            qrels[qid][docid] = relevance
    return dict(qrels)


def read_candidates(
    path: Path,
) -> Tuple[Dict[str, str], Dict[Tuple[str, str], Dict[str, str]]]:
    queries: Dict[str, str] = {}
    documents: Dict[Tuple[str, str], Dict[str, str]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, 1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                qid = str(row["query_id"])
                docid = str(row["docid"])
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                raise ValueError(f"{path}:{line_number}: malformed candidate JSON") from exc
            queries.setdefault(qid, str(row.get("query", "")))
            documents[(qid, docid)] = {
                "title": str(row.get("title", "")),
                "text": str(row.get("text", "")),
            }
    return queries, documents


def dcg(relevances: Iterable[float], cutoff: int = 10) -> float:
    """Match trec_eval ndcg_cut: gain defaults to the qrels relevance value."""
    total = 0.0
    for rank, relevance in enumerate(relevances, 1):
        if rank > cutoff:
            break
        total += relevance / math.log2(rank + 1.0)
    return total


def ndcg_at_k(run: List[RunEntry], qrels: Mapping[str, float], cutoff: int = 10) -> float:
    actual = [qrels.get(docid, 0.0) for _, docid, _ in run[:cutoff]]
    ideal = sorted(qrels.values(), reverse=True)[:cutoff]
    ideal_dcg = dcg(ideal, cutoff)
    return dcg(actual, cutoff) / ideal_dcg if ideal_dcg > 0.0 else 0.0


def top_documents(
    run: List[RunEntry],
    qrels: Mapping[str, float],
    documents: Mapping[Tuple[str, str], Mapping[str, str]],
    qid: str,
    cutoff: int,
) -> List[dict]:
    output = []
    for output_rank, (_, docid, score) in enumerate(run[:cutoff], 1):
        doc = documents.get((qid, docid), {})
        output.append(
            {
                "rank": output_rank,
                "docid": docid,
                "score": score,
                "relevance": qrels.get(docid, 0.0),
                "title": doc.get("title", ""),
                "text": doc.get("text", ""),
            }
        )
    return output


def analyze_dataset(
    name: str,
    dear_path: Path,
    kk_path: Path,
    qrels_path: Path,
    candidates_path: Path,
    cutoff: int,
    epsilon: float,
) -> dict:
    dear_run = read_run(dear_path)
    kk_run = read_run(kk_path)
    qrels = read_qrels(qrels_path)
    queries, documents = read_candidates(candidates_path)

    comparable_qids = sorted(set(dear_run) & set(kk_run) & set(qrels))
    degraded = []
    improved = 0
    tied = 0

    for qid in comparable_qids:
        dear_ndcg = ndcg_at_k(dear_run[qid], qrels[qid], cutoff)
        kk_ndcg = ndcg_at_k(kk_run[qid], qrels[qid], cutoff)
        delta = kk_ndcg - dear_ndcg
        if delta < -epsilon:
            degraded.append(
                {
                    "dataset": name,
                    "qid": qid,
                    "query": queries.get(qid, ""),
                    "dear_ndcg_at_10": round(dear_ndcg, 8),
                    "kk_ndcg_at_10": round(kk_ndcg, 8),
                    "delta_kk_minus_dear": round(delta, 8),
                    "dear_top10": top_documents(
                        dear_run[qid], qrels[qid], documents, qid, cutoff
                    ),
                    "kk_top10": top_documents(
                        kk_run[qid], qrels[qid], documents, qid, cutoff
                    ),
                }
            )
        elif delta > epsilon:
            improved += 1
        else:
            tied += 1

    degraded.sort(key=lambda row: row["delta_kk_minus_dear"])
    return {
        "dataset": name,
        "summary": {
            "comparable_queries": len(comparable_qids),
            "degraded_queries": len(degraded),
            "improved_queries": improved,
            "tied_queries": tied,
            "missing_query_texts_in_degraded": sum(not row["query"] for row in degraded),
        },
        "degraded_queries": degraded,
    }


def existing_file(value: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"file does not exist: {path}")
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract DL19/DL20 queries where DeAR+KK loses to DeAR."
    )
    for dataset in ("dl19", "dl20"):
        parser.add_argument(f"--{dataset}-dear", type=existing_file, required=True)
        parser.add_argument(f"--{dataset}-kk", type=existing_file, required=True)
        parser.add_argument(f"--{dataset}-qrels", type=existing_file, required=True)
        parser.add_argument(f"--{dataset}-candidates", type=existing_file, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cutoff", type=int, default=10)
    parser.add_argument(
        "--epsilon",
        type=float,
        default=1e-12,
        help="Treat smaller absolute differences as ties.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    analyses = []
    for dataset in ("dl19", "dl20"):
        analyses.append(
            analyze_dataset(
                dataset.upper(),
                getattr(args, f"{dataset}_dear"),
                getattr(args, f"{dataset}_kk"),
                getattr(args, f"{dataset}_qrels"),
                getattr(args, f"{dataset}_candidates"),
                args.cutoff,
                args.epsilon,
            )
        )

    all_degraded = [
        row for analysis in analyses for row in analysis["degraded_queries"]
    ]
    result = {
        "metric": f"nDCG@{args.cutoff}",
        "comparison": "DeAR+KK minus DeAR",
        "datasets": analyses,
        "all_degraded_queries": sorted(
            all_degraded,
            key=lambda row: (row["dataset"], row["delta_kk_minus_dear"]),
        ),
    }

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)

    print(f"Saved: {output}")
    for analysis in analyses:
        summary = analysis["summary"]
        print(
            f"{analysis['dataset']}: comparable={summary['comparable_queries']}, "
            f"degraded={summary['degraded_queries']}, "
            f"improved={summary['improved_queries']}, tied={summary['tied_queries']}"
        )


if __name__ == "__main__":
    main()
