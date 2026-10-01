#!/usr/bin/env python
"""Post-hoc DL19 oracle analysis; never imported by inference code."""

import argparse
import json
import math
from collections import defaultdict


def read_qrels(path):
    qrels = defaultdict(dict)
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            qid, _, docid, relevance = line.split()
            qrels[qid][docid] = int(relevance)
    return qrels


def read_run(path):
    rows = defaultdict(list)
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            qid, _, docid, rank, score, *_ = line.split()
            # jtreceval orders by the score field; use rank only as a stable
            # tie breaker so oracle analysis matches the evaluated run.
            rows[qid].append((-float(score), int(rank), docid))
    return {qid: [docid for _, _, docid in sorted(values)] for qid, values in rows.items()}


def ndcg_at_10(ranking, labels):
    # This repository evaluates with jtreceval's graded nDCG convention,
    # whose gain is the qrels relevance itself (not 2**relevance-1).
    dcg = sum(labels.get(docid, 0) / math.log2(rank + 2) for rank, docid in enumerate(ranking[:10]))
    ideal = sorted(labels.values(), reverse=True)[:10]
    idcg = sum(relevance / math.log2(rank + 2) for rank, relevance in enumerate(ideal))
    return 0.0 if idcg == 0.0 else dcg / idcg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qrels", required=True)
    parser.add_argument("--base", required=True)
    parser.add_argument("--adapted", required=True)
    parser.add_argument("--hard", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    labels, base, adapted, hard = read_qrels(args.qrels), read_run(args.base), read_run(args.adapted), read_run(args.hard)
    qids = sorted(set(labels) & set(base) & set(adapted) & set(hard))
    rows, wins = [], {"adapted": 0, "base": 0, "tie": 0}
    for qid in qids:
        s0, s1, sh = (ndcg_at_10(run[qid], labels[qid]) for run in (base, adapted, hard))
        winner = "adapted" if s1 > s0 else "base" if s0 > s1 else "tie"
        wins[winner] += 1
        rows.append({"qid": qid, "base_ndcg@10": s0, "adapted_ndcg@10": s1, "hard_ndcg@10": sh, "oracle_ndcg@10": max(s0, s1), "s1_vs_s0": winner})
    summary = {
        "label": "Oracle upper bound / diagnostic only; not used by inference or routing.",
        "queries": len(rows), "s1_wins": wins["adapted"], "s0_wins": wins["base"], "ties": wins["tie"],
        "mean_base_ndcg@10": sum(row["base_ndcg@10"] for row in rows) / len(rows),
        "mean_adapted_ndcg@10": sum(row["adapted_ndcg@10"] for row in rows) / len(rows),
        "mean_hard_ndcg@10": sum(row["hard_ndcg@10"] for row in rows) / len(rows),
        "mean_oracle_ndcg@10": sum(row["oracle_ndcg@10"] for row in rows) / len(rows),
        "rows": rows,
    }
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps({key: value for key, value in summary.items() if key != "rows"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
