#!/usr/bin/env python3
"""Quantify ranking regressions for pairs that a teacher orders correctly.

A regression pair has different qrels, is ordered correctly by the ``before``
student, incorrectly by the ``after`` student, and correctly by the teacher.
Scores are z-normalized within each query before margins are compared.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path


def read_scores(path: Path) -> dict[str, dict[str, float]]:
    rows: dict[str, dict[str, float]] = defaultdict(dict)
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            parts = line.split()
            try:
                if len(parts) == 3:
                    qid, docid, score = parts
                elif len(parts) >= 6:
                    qid, docid, score = parts[0], parts[2], parts[4]
                else:
                    raise ValueError
                rows[qid][docid] = float(score)
            except ValueError as exc:
                raise ValueError(f"Malformed run row at {path}:{line_number}") from exc
    return dict(rows)


def read_qrels(path: Path) -> dict[str, dict[str, int]]:
    rows: dict[str, dict[str, int]] = defaultdict(dict)
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            qid, _, docid, relevance = line.split()[:4]
            rows[qid][docid] = int(relevance)
    return dict(rows)


def read_candidates(path: Path) -> tuple[dict[str, str], dict[tuple[str, str], str]]:
    queries: dict[str, str] = {}
    texts: dict[tuple[str, str], str] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            qid, docid = str(row["query_id"]), str(row["docid"])
            queries.setdefault(qid, str(row.get("query", "")))
            texts[(qid, docid)] = " ".join(
                part for part in (str(row.get("title", "")), str(row.get("text", ""))) if part
            )
    return queries, texts


def read_query_metrics(path: Path | None) -> dict[str, float]:
    if path is None:
        return {}
    metrics: dict[str, float] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            parts = line.split()
            if len(parts) == 3 and parts[0] == "ndcg_cut_10" and parts[1] != "all":
                metrics[parts[1]] = float(parts[2])
    return metrics


def ranked(scores: dict[str, float]) -> tuple[list[str], dict[str, int]]:
    # The inference files are already score-sorted.  Preserve that input rank
    # as the stable tie breaker, matching the evaluated TREC run.
    order = sorted(scores, key=lambda docid: -scores[docid])
    return order, {docid: index + 1 for index, docid in enumerate(order)}


def zscores(scores: dict[str, float]) -> dict[str, float]:
    values = list(scores.values())
    mean = statistics.fmean(values)
    std = math.sqrt(statistics.fmean((value - mean) ** 2 for value in values))
    return {key: (value - mean) / (std + 1e-8) for key, value in scores.items()}


def ndcg(order: list[str], labels: dict[str, int], cutoff: int = 10) -> float:
    actual = sum(labels.get(docid, 0) / math.log2(rank + 1) for rank, docid in enumerate(order[:cutoff], 1))
    ideal_labels = sorted(labels.values(), reverse=True)[:cutoff]
    ideal = sum(label / math.log2(rank + 1) for rank, label in enumerate(ideal_labels, 1))
    return actual / ideal if ideal else 0.0


def quantiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "median": None, "q25": None, "q75": None}
    ordered = sorted(values)
    def at(fraction: float) -> float:
        position = (len(ordered) - 1) * fraction
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return ordered[lower]
        return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)
    return {
        "mean": statistics.fmean(ordered),
        "median": statistics.median(ordered),
        "q25": at(0.25),
        "q75": at(0.75),
    }


def fmt(value: float | None, digits: int = 3) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--qrels", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--before-query-metrics", type=Path)
    parser.add_argument("--after-query-metrics", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--before-name", default="before")
    parser.add_argument("--after-name", default="after")
    args = parser.parse_args()

    before, after, teacher = (read_scores(path) for path in (args.before, args.after, args.teacher))
    qrels = read_qrels(args.qrels)
    queries, texts = read_candidates(args.candidates)
    before_query_metrics = read_query_metrics(args.before_query_metrics)
    after_query_metrics = read_query_metrics(args.after_query_metrics)
    qids = sorted(set(before) & set(after) & set(teacher) & set(qrels))
    pair_rows: list[dict[str, object]] = []
    nn_rows: list[dict[str, object]] = []
    query_rows: list[dict[str, object]] = []

    for qid in qids:
        common = set(before[qid]) & set(after[qid]) & set(teacher[qid])
        labels = {docid: relevance for docid, relevance in qrels[qid].items() if docid in common}
        before_order, before_rank = ranked(before[qid])
        after_order, after_rank = ranked(after[qid])
        _, teacher_rank = ranked(teacher[qid])
        before_z, after_z, teacher_z = zscores(before[qid]), zscores(after[qid]), zscores(teacher[qid])
        before_ndcg = before_query_metrics.get(qid, ndcg(before_order, qrels[qid]))
        after_ndcg = after_query_metrics.get(qid, ndcg(after_order, qrels[qid]))
        degraded = after_ndcg < before_ndcg - 1e-12
        counts = Counter()

        label_items = sorted(labels.items())
        for left in range(len(label_items)):
            doc_a, rel_a = label_items[left]
            for right in range(left + 1, len(label_items)):
                doc_b, rel_b = label_items[right]
                if rel_a == rel_b:
                    continue
                high_doc, high_rel, low_doc, low_rel = (
                    (doc_a, rel_a, doc_b, rel_b) if rel_a > rel_b else (doc_b, rel_b, doc_a, rel_a)
                )
                before_margin = before_z[high_doc] - before_z[low_doc]
                after_margin = after_z[high_doc] - after_z[low_doc]
                teacher_margin = teacher_z[high_doc] - teacher_z[low_doc]
                teacher_correct = teacher_margin > 0
                after_wrong = after_margin < 0
                before_correct = before_margin > 0
                if teacher_correct and after_wrong:
                    counts["teacher_correct_after_wrong"] += 1
                if not (teacher_correct and before_correct and after_wrong):
                    continue
                pair_type = "PN" if low_rel == 0 else "graded-positive"
                counts["new_regression_pairs"] += 1
                counts[f"{pair_type}_pairs"] += 1
                boundary = (
                    (before_rank[high_doc] <= 10) != (before_rank[low_doc] <= 10)
                    or (after_rank[high_doc] <= 10) != (after_rank[low_doc] <= 10)
                )
                if boundary:
                    counts["top10_boundary_pairs"] += 1
                    counts[f"{pair_type}_boundary_pairs"] += 1
                pair_rows.append({
                    "qid": qid,
                    "query": queries.get(qid, ""),
                    "query_degraded": degraded,
                    "high_docid": high_doc,
                    "low_docid": low_doc,
                    "high_qrel": high_rel,
                    "low_qrel": low_rel,
                    "qrel_pair": f"{high_rel}>{low_rel}",
                    "pair_type": pair_type,
                    "qrel_gap": high_rel - low_rel,
                    "before_high_rank": before_rank[high_doc],
                    "before_low_rank": before_rank[low_doc],
                    "after_high_rank": after_rank[high_doc],
                    "after_low_rank": after_rank[low_doc],
                    "teacher_high_rank": teacher_rank[high_doc],
                    "teacher_low_rank": teacher_rank[low_doc],
                    "before_margin_z": before_margin,
                    "after_wrong_margin_z": -after_margin,
                    "teacher_margin_z": teacher_margin,
                    "touches_top10_boundary": boundary,
                    "high_doc_words": len(texts.get((qid, high_doc), "").split()),
                    "low_doc_words": len(texts.get((qid, low_doc), "").split()),
                })

        zero_docs = [docid for docid, relevance in label_items if relevance == 0]
        nn_total_pairs = len(zero_docs) * (len(zero_docs) - 1) // 2
        nn_disagreements = 0
        nn_boundary_disagreements = 0
        for left in range(len(zero_docs)):
            for right in range(left + 1, len(zero_docs)):
                doc_a, doc_b = zero_docs[left], zero_docs[right]
                teacher_margin = teacher_z[doc_a] - teacher_z[doc_b]
                student_margin = after_z[doc_a] - after_z[doc_b]
                if teacher_margin == 0.0 or student_margin == 0.0 or teacher_margin * student_margin > 0.0:
                    continue
                nn_disagreements += 1
                boundary = (
                    (teacher_rank[doc_a] <= 10) != (teacher_rank[doc_b] <= 10)
                    or (after_rank[doc_a] <= 10) != (after_rank[doc_b] <= 10)
                )
                if boundary:
                    nn_boundary_disagreements += 1
                nn_rows.append({
                    "qid": qid,
                    "query": queries.get(qid, ""),
                    "query_degraded": degraded,
                    "doc_a": doc_a,
                    "doc_b": doc_b,
                    "teacher_a_rank": teacher_rank[doc_a],
                    "teacher_b_rank": teacher_rank[doc_b],
                    "student_a_rank": after_rank[doc_a],
                    "student_b_rank": after_rank[doc_b],
                    "teacher_abs_margin_z": abs(teacher_margin),
                    "student_opposite_margin_z": abs(student_margin),
                    "touches_top10_boundary": boundary,
                })

        all_truth_pairs = sum(
            1 for i, (_, a) in enumerate(label_items) for _, b in label_items[i + 1:] if a != b
        )
        query_rows.append({
            "qid": qid,
            "query": queries.get(qid, ""),
            "query_words": len(queries.get(qid, "").split()),
            "before_ndcg@10": before_ndcg,
            "after_ndcg@10": after_ndcg,
            "delta_after_minus_before": after_ndcg - before_ndcg,
            "degraded": degraded,
            "judged_candidates": len(labels),
            "truth_pairs": all_truth_pairs,
            "teacher_correct_after_wrong_pairs": counts["teacher_correct_after_wrong"],
            "teacher_correct_after_wrong_rate": counts["teacher_correct_after_wrong"] / all_truth_pairs if all_truth_pairs else 0.0,
            "new_regression_pairs": counts["new_regression_pairs"],
            "pn_regression_pairs": counts["PN_pairs"],
            "graded_positive_regression_pairs": counts["graded-positive_pairs"],
            "pn_regression_share": counts["PN_pairs"] / counts["new_regression_pairs"] if counts["new_regression_pairs"] else 0.0,
            "new_regression_rate": counts["new_regression_pairs"] / all_truth_pairs if all_truth_pairs else 0.0,
            "top10_boundary_regression_pairs": counts["top10_boundary_pairs"],
            "pn_top10_boundary_pairs": counts["PN_boundary_pairs"],
            "graded_positive_top10_boundary_pairs": counts["graded-positive_boundary_pairs"],
            "explicit_nn_pairs": nn_total_pairs,
            "nn_teacher_student_disagreements": nn_disagreements,
            "nn_disagreement_rate": nn_disagreements / nn_total_pairs if nn_total_pairs else 0.0,
            "nn_top10_boundary_disagreements": nn_boundary_disagreements,
        })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for filename, rows in (
        ("queries.csv", query_rows),
        ("pairs.csv", pair_rows),
        ("nn_disagreements.csv", nn_rows),
    ):
        with (args.output_dir / filename).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [])
            if rows:
                writer.writeheader()
                writer.writerows(rows)

    degraded_qids = {str(row["qid"]) for row in query_rows if row["degraded"]}
    degraded_pairs = [row for row in pair_rows if str(row["qid"]) in degraded_qids]
    stable_pairs = [row for row in pair_rows if str(row["qid"]) not in degraded_qids]
    degraded_queries = [row for row in query_rows if row["degraded"]]
    other_queries = [row for row in query_rows if not row["degraded"]]
    severe_qids = {
        str(row["qid"])
        for row in degraded_queries
        if float(row["delta_after_minus_before"]) <= -0.1
    }
    pair_gap_counts = Counter(int(row["qrel_gap"]) for row in degraded_pairs)
    qrel_pair_counts = Counter(str(row["qrel_pair"]) for row in degraded_pairs)

    pair_type_features: dict[str, dict[str, object]] = {}
    for pair_type in ("PN", "graded-positive"):
        selected = [row for row in degraded_pairs if row["pair_type"] == pair_type]
        pair_type_features[pair_type] = {
            "count": len(selected),
            "share": len(selected) / len(degraded_pairs) if degraded_pairs else 0.0,
            "queries": len({str(row["qid"]) for row in selected}),
            "top10_boundary_count": sum(bool(row["touches_top10_boundary"]) for row in selected),
            "teacher_margin_z": quantiles([float(row["teacher_margin_z"]) for row in selected]),
            "after_wrong_margin_z": quantiles([float(row["after_wrong_margin_z"]) for row in selected]),
        }
    severe_pairs = [row for row in degraded_pairs if str(row["qid"]) in severe_qids]
    severe_pn_count = sum(row["pair_type"] == "PN" for row in severe_pairs)
    severe_pn_boundary = sum(
        row["pair_type"] == "PN" and bool(row["touches_top10_boundary"])
        for row in severe_pairs
    )
    severe_graded_boundary = sum(
        row["pair_type"] == "graded-positive" and bool(row["touches_top10_boundary"])
        for row in severe_pairs
    )
    degraded_nn_rows = [row for row in nn_rows if str(row["qid"]) in degraded_qids]
    degraded_nn_total = sum(int(row["explicit_nn_pairs"]) for row in degraded_queries)
    degraded_nn_boundary = sum(
        bool(row["touches_top10_boundary"]) for row in degraded_nn_rows
    )

    def query_group_features(rows: list[dict[str, object]]) -> dict[str, dict[str, float | None]]:
        return {
            key: quantiles([float(row[key]) for row in rows])
            for key in (
                "query_words",
                "judged_candidates",
                "teacher_correct_after_wrong_rate",
                "new_regression_rate",
                "top10_boundary_regression_pairs",
            )
        }

    summary = {
        "before_name": args.before_name,
        "after_name": args.after_name,
        "queries": len(query_rows),
        "degraded_queries": len(degraded_queries),
        "improved_queries": sum(float(row["delta_after_minus_before"]) > 1e-12 for row in query_rows),
        "tied_queries": sum(abs(float(row["delta_after_minus_before"])) <= 1e-12 for row in query_rows),
        "mean_before_ndcg@10": statistics.fmean(float(row["before_ndcg@10"]) for row in query_rows),
        "mean_after_ndcg@10": statistics.fmean(float(row["after_ndcg@10"]) for row in query_rows),
        "new_regression_pairs": len(pair_rows),
        "new_regression_pairs_in_degraded_queries": len(degraded_pairs),
        "explicit_nn_diagnostic_in_degraded_queries": {
            "total_pairs": degraded_nn_total,
            "teacher_student_disagreements": len(degraded_nn_rows),
            "disagreement_rate": len(degraded_nn_rows) / degraded_nn_total if degraded_nn_total else 0.0,
            "top10_boundary_disagreements": degraded_nn_boundary,
            "teacher_abs_margin_z": quantiles([float(row["teacher_abs_margin_z"]) for row in degraded_nn_rows]),
            "student_opposite_margin_z": quantiles([float(row["student_opposite_margin_z"]) for row in degraded_nn_rows]),
        },
        "severe_regressions": {
            "threshold": -0.1,
            "queries": len(severe_qids),
            "pairs": len(severe_pairs),
            "pn_pairs": severe_pn_count,
            "pn_share": severe_pn_count / len(severe_pairs) if severe_pairs else 0.0,
            "pn_top10_boundary_pairs": severe_pn_boundary,
            "graded_positive_top10_boundary_pairs": severe_graded_boundary,
        },
        "degraded_query_features": query_group_features(degraded_queries),
        "other_query_features": query_group_features(other_queries),
        "degraded_pair_features": {
            "teacher_margin_z": quantiles([float(row["teacher_margin_z"]) for row in degraded_pairs]),
            "after_wrong_margin_z": quantiles([float(row["after_wrong_margin_z"]) for row in degraded_pairs]),
            "before_margin_z": quantiles([float(row["before_margin_z"]) for row in degraded_pairs]),
            "qrel_gap_counts": dict(sorted(pair_gap_counts.items())),
            "qrel_pair_counts": dict(sorted(qrel_pair_counts.items())),
            "top10_boundary_count": sum(bool(row["touches_top10_boundary"]) for row in degraded_pairs),
            "pair_type_features": pair_type_features,
        },
        "other_pair_features": {
            "teacher_margin_z": quantiles([float(row["teacher_margin_z"]) for row in stable_pairs]),
            "after_wrong_margin_z": quantiles([float(row["after_wrong_margin_z"]) for row in stable_pairs]),
        },
    }
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    ranked_queries = sorted(degraded_queries, key=lambda row: float(row["delta_after_minus_before"]))
    feature = summary["degraded_pair_features"]
    pair_type_features = feature["pair_type_features"]
    boundary_count = int(feature["top10_boundary_count"])
    degraded_query_features = summary["degraded_query_features"]
    other_query_features = summary["other_query_features"]
    md = [
        "# Teacher 排对、student 翻错的退步分析",
        "",
        "## 定义",
        "",
        f"- 改动前 student：`{args.before_name}`。",
        f"- 改动后 student：`{args.after_name}`。",
        "- query 退步：改动后逐 query nDCG@10 低于改动前。",
        "- 新增退步 pair：两个文档有不同 qrel，改动前 student 排对，改动后 student 排错，且 teacher 排对。",
        "- margin 均在每条 query 的 Top-100 内做 z-score 后计算；只使用 qrels 中明确标注的文档。",
        "- 等级组合包含明确标注的 qrel=0；未标注文档不参加 pair 统计。",
        "- 本报告将 qrel>0 对 qrel=0 记为 PN，将两个不同等级的 qrel>0 文档记为正例内部 pair；qrel=0 对 qrel=0 没有真值先后，不能据此判断 NN 对错。",
        "",
        "## 汇总",
        "",
        f"- 平均 nDCG@10：{summary['mean_before_ndcg@10']:.4f} → {summary['mean_after_ndcg@10']:.4f}（{summary['mean_after_ndcg@10']-summary['mean_before_ndcg@10']:+.4f}）。",
        f"- {summary['queries']} 条 query 中，退步 {summary['degraded_queries']} 条、提升 {summary['improved_queries']} 条、不变 {summary['tied_queries']} 条。",
        f"- 共 {summary['new_regression_pairs']} 个 teacher 排对而 student 新翻错的 pair，其中 {len(degraded_pairs)} 个位于 nDCG@10 退步 query。",
        f"- 退步 query 中有 {boundary_count}/{len(degraded_pairs) if degraded_pairs else 0} 个新增退步 pair 触及 Top-10 边界。",
        "",
        "## Query 级特征对照",
        "",
        "| 特征 | 退步 query 均值 / 中位数 | 其余 query 均值 / 中位数 |",
        "|---|---:|---:|",
    ]
    for label, key, percent in (
        ("query 词数", "query_words", False),
        ("Top-100 内已标注文档数", "judged_candidates", False),
        ("teacher 对而改动后 student 错的 pair 比例", "teacher_correct_after_wrong_rate", True),
        ("新增退步 pair 比例", "new_regression_rate", True),
        ("触及 Top-10 边界的新增退步 pair 数", "top10_boundary_regression_pairs", False),
    ):
        left, right = degraded_query_features[key], other_query_features[key]
        scale, suffix = (100.0, "%") if percent else (1.0, "")
        md.append(
            f"| {label} | {fmt(left['mean'] * scale)}{suffix} / {fmt(left['median'] * scale)}{suffix} | "
            f"{fmt(right['mean'] * scale)}{suffix} / {fmt(right['median'] * scale)}{suffix} |"
        )
    md.extend([
        "",
        "## 退步 query 内新增退步 pair 的量化特征",
        "",
        "| 特征 | 均值 | 中位数 | Q1 | Q3 |",
        "|---|---:|---:|---:|---:|",
    ])
    for label, key in (("teacher 正确 margin（z）", "teacher_margin_z"), ("改动后错误 margin（z）", "after_wrong_margin_z"), ("改动前正确 margin（z）", "before_margin_z")):
        values = feature[key]
        md.append(f"| {label} | {fmt(values['mean'])} | {fmt(values['median'])} | {fmt(values['q25'])} | {fmt(values['q75'])} |")
    md.extend([
        "",
        "qrel gap 分布：" + "，".join(f"gap={key}: {value}" for key, value in pair_gap_counts.items()),
        "",
        "qrel 等级组合：" + "，".join(f"{key}: {value}" for key, value in sorted(qrel_pair_counts.items())),
        "",
        "### PN 与正例内部 pair",
        "",
        "| 类型 | 数量（占比） | 涉及 query | Top-10 边界数量（占该类） | teacher margin 中位数 | student 错误 margin 中位数 |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for pair_type in ("PN", "graded-positive"):
        values = pair_type_features[pair_type]
        count = int(values["count"])
        boundary = int(values["top10_boundary_count"])
        md.append(
            f"| {pair_type} | {count}（{float(values['share']):.1%}） | {values['queries']} | "
            f"{boundary}（{boundary / count if count else 0.0:.1%}） | "
            f"{fmt(values['teacher_margin_z']['median'])} | {fmt(values['after_wrong_margin_z']['median'])} |"
        )
    severe = summary["severe_regressions"]
    nn_feature = summary["explicit_nn_diagnostic_in_degraded_queries"]
    md.extend([
        "",
        f"若只看 ΔnDCG@10≤-0.10 的 {severe['queries']} 条严重退步 query，"
        f"PN 占 {severe['pn_pairs']}/{severe['pairs']}（{severe['pn_share']:.1%}）；"
        f"Top-10 边界错误中 PN 为 {severe['pn_top10_boundary_pairs']} 个，"
        f"正例内部为 {severe['graded_positive_top10_boundary_pairs']} 个。",
        "",
        "### 显式 NN（qrel=0 对 qrel=0）诊断",
        "",
        f"退步 query 中共有 {nn_feature['total_pairs']} 个显式 NN pair；teacher 与 student 对其中 "
        f"{nn_feature['teacher_student_disagreements']} 个排序方向不同（{nn_feature['disagreement_rate']:.1%}），"
        f"{nn_feature['top10_boundary_disagreements']} 个分歧触及 Top-10 边界。"
        "由于两侧 qrel 都是0，这些只能视为排序分歧，不能判定teacher正确/student错误，也不会直接改变nDCG。",
        "",
        "## 退步最大的 query",
        "",
        "| qid | query | ΔnDCG@10 | PN | 正例内部 | PN 占比 | Top-10 边界 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ])
    for row in ranked_queries[:15]:
        query_text = str(row["query"]).replace("|", "\\|")
        md.append(
            f"| {row['qid']} | {query_text} | {float(row['delta_after_minus_before']):+.4f} | "
            f"{row['pn_regression_pairs']} | {row['graded_positive_regression_pairs']} | "
            f"{float(row['pn_regression_share']):.1%} | {row['top10_boundary_regression_pairs']} |"
        )
    md.extend(["", "完整明细见 `queries.csv`、`pairs.csv` 与 `nn_disagreements.csv`。", ""])
    (args.output_dir / "report.md").write_text("\n".join(md), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
