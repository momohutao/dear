#!/usr/bin/env python3
"""Streaming, CPU-only selection of 1 positive + 7 existing negatives.

Rule labels are mining hypotheses, NOT relevance judgments or teacher scores.
See HARD_NEGATIVE_MINING.md for baseline and classifier definitions.
"""
import argparse
from collections import Counter
import gzip
import hashlib
import itertools
import json
import os
from pathlib import Path
import random
import re
import tempfile
import time


STOP = set("a an the of to is are was were be been being do does did what where when who why how in on at for from with by and or as it its this that these those can could should would may might about definition define meaning type types cause causes use used uses s".split())
DOMAINS = {
    "medical": set("medical medicare medicaid patient patients hospital clinical doctor physician lung lungs breathing respiratory intubation ventilator disease treatment surgery diagnosis symptom symptoms".split()),
    "building": set("hvac architecture architectural building buildings duct ducts airflow indoor thermostat insulation attic heating".split()),
    "computing": set("software hardware computer computers programming python javascript database databases compiler server servers linux windows cpu".split()),
    "biology": set("biological biology organism organisms animal animals species cell cells genetic genetics tissue tissues bacteria".split()),
    "finance": set("financial finance bank banking mortgage stock stocks shares investor investment interest loan loans credit".split()),
    "cooking": set("cooking cook recipe recipes food foods meat beef steak eggs chicken oven grill bake baking kitchen".split()),
    "law": set("legal law laws court courts statute legislation attorney lawsuit judicial constitutional".split()),
}
COMPATIBLE_DOMAINS = {frozenset(("medical", "biology"))}
OPPOSITES = [("left", "right"), ("upper", "lower"), ("before", "after"),
             ("adult", "child"), ("male", "female"), ("acute", "chronic"),
             ("indoor", "outdoor"), ("positive", "negative"), ("increase", "decrease")]
QUALIFIERS = set(itertools.chain.from_iterable(OPPOSITES)) | set(
    "medicare medicaid pregnant pregnancy pediatric infant newborn elderly federal state local online offline free paid electric electrical manual automatic vegan vegetarian gluten lactose".split())
ALIASES = {
    "child": {"child", "children", "pediatric", "paediatric"},
    "adult": {"adult", "adults"}, "female": {"female", "women", "woman"},
    "male": {"male", "men", "man"}, "pregnant": {"pregnant", "pregnancy"},
    "pregnancy": {"pregnant", "pregnancy"},
}
INTENT_RULES = [
    ("list", r"\b(?:what|which)\s+(?:types?|kinds?|foods?|items?)\b|\blist of\b|\bexamples? of\b"),
    ("comparison", r"\b(?:difference|differences|compare|comparison|versus|vs)\b"),
    ("cause", r"\b(?:cause|causes|caused|why|reason|reasons)\b"),
    ("procedure", r"\bhow (?:to|do|does|can|should)\b|\bsteps to\b"),
    ("duration", r"\bhow long\b|\bduration\b"),
    ("quantity", r"\bhow (?:much|many)\b|\bcost\b|\bprice\b"),
    ("location", r"\bwhere\b|\blocation\b"),
    ("definition", r"\b(?:definition|define|meaning|mean)\b|^what (?:is|are)\b"),
]
INTENT_RULES = [(name, re.compile(pattern)) for name, pattern in INTENT_RULES]
DOC_MARKERS = {
    "definition": re.compile(r"\b(?:defined as|definition|means|refers to|is a|is an|are a)\b"),
    "cause": re.compile(r"\b(?:caus\w*|because|due to|result from|results from|risk factors?)\b"),
    "procedure": re.compile(r"\b(?:step\s*\d|first|next|finally|instructions|how to)\b"),
    "list": re.compile(r"\b(?:include|includes|including|such as|examples|types of|kinds of)\b|(?:[^,]+,){2}|\b1[.)]\s"),
    "comparison": re.compile(r"\b(?:whereas|unlike|versus|compared|difference|both|than)\b"),
    "duration": re.compile(r"\b\d+(?:\.\d+)?\s*(?:seconds?|minutes?|hours?|days?|weeks?|months?|years?)\b"),
    "quantity": re.compile(r"\b\d+(?:\.\d+)?\b|[$£€]"),
    "location": re.compile(r"\b(?:located|situated|address|north|south|east|west|born in|filmed in)\b"),
}
LABELS = ("domain_confusion", "intent_mismatch", "keyword_constraint_missing", "other")


def tokens(text):
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def passage_text(doc):
    return ((doc.get("title") or "") + " " + doc["text"]).strip().lower()


def fingerprint(doc):
    # Ignore title differences so copies of the same passage cannot be positive/negative.
    return " ".join(doc["text"].lower().split())


def seed_for(seed, qid, query):
    return int.from_bytes(hashlib.sha256(f"{seed}\0{qid}\0{query}".encode()).digest()[:8], "big")


def make_context(query, positives):
    qt = tokens(query)
    content = qt - STOP
    pt = set().union(*(tokens(passage_text(p)) for p in positives))
    # Require positive evidence for a qualifier, reducing query-only false alarms.
    constraints = sorted(t for t in qt if (t in QUALIFIERS or t.isdigit())
                         and ALIASES.get(t, {t}) & pt)
    counts = {d: len(qt & terms) for d, terms in DOMAINS.items()}
    if not max(counts.values()):
        counts = {d: len(pt & terms) for d, terms in DOMAINS.items()}
        threshold = 2
    else:
        threshold = 1
    maximum = max(counts.values())
    winners = [d for d, n in counts.items() if n == maximum and n >= threshold]
    domain = winners[0] if len(winners) == 1 else None
    intent = next((name for name, pat in INTENT_RULES if pat.search(query.lower())), None)
    if intent and not any(DOC_MARKERS[intent].search(passage_text(p)) for p in positives):
        intent = None
    return content, constraints, domain, intent


def classify(doc, context):
    content, constraints, expected_domain, intent = context
    text = passage_text(doc)
    dt = tokens(text)
    overlap = len(content & dt)
    coverage = overlap / max(1, len(content))
    labels, reasons = [], []
    # A hard negative must share topic words. Long documents do not get an
    # automatic score bonus from repeating those words.
    topical = overlap >= min(2, len(content)) and coverage >= 0.35 and bool(content)
    if topical:
        missing = [t for t in constraints if not (ALIASES.get(t, {t}) & dt)]
        if missing:
            labels.append("keyword_constraint_missing")
            reasons.append("missing_positive_supported_constraints=" + ",".join(missing))
        if expected_domain:
            scores = {d: len(dt & terms) for d, terms in DOMAINS.items()}
            other = max((d for d in scores if d != expected_domain), key=scores.get)
            if (frozenset((other, expected_domain)) not in COMPATIBLE_DOMAINS
                    and scores[other] >= 2 and scores[other] >= scores[expected_domain] + 2):
                labels.append("domain_confusion")
                reasons.append(f"domain={other};expected={expected_domain}")
        if intent and not DOC_MARKERS[intent].search(text):
            alternatives = [name for name, pat in DOC_MARKERS.items()
                            if name != intent and pat.search(text)]
            if alternatives:
                labels.append("intent_mismatch")
                reasons.append(f"expected_intent={intent};observed=" + ",".join(alternatives))
    primary = next((label for label in LABELS if label in labels), "other")
    return {"primary": primary, "labels": labels, "score": round(coverage + 0.1 * len(labels), 6),
            "coverage": round(coverage, 6), "reasons": reasons}


def validate_doc(doc):
    if not isinstance(doc, dict) or "docid" not in doc or not isinstance(doc.get("text"), str):
        raise ValueError("passages require docid and string text")
    if not str(doc["docid"]).strip() or not doc["text"].strip():
        raise ValueError("empty docid or text")
    if doc.get("title") is not None and not isinstance(doc["title"], str):
        raise ValueError("title must be string or null")


def mine_row(row, seed=42, hard_slots=4, short_pool="repeat"):
    if not isinstance(row.get("query"), str) or not row["query"].strip() or "query_id" not in row:
        raise ValueError("query_id and nonempty query required")
    positives = row.get("positive_passages", [])
    negatives = row.get("negative_passages", [])
    if not positives:
        raise ValueError("no positive passages")
    for doc in itertools.chain(positives, negatives):
        validate_doc(doc)
    posids = {str(p["docid"]) for p in positives}
    postexts = {fingerprint(p) for p in positives}
    ids, texts, pool = set(), set(), []
    excluded = Counter()
    for doc in negatives:
        did, txt = str(doc["docid"]), fingerprint(doc)
        if did in posids or txt in postexts:
            excluded["known_positive_or_positive_copy"] += 1
        elif did in ids or txt in texts:
            excluded["duplicate_negative"] += 1
        else:
            pool.append(doc)
            ids.add(did)
            texts.add(txt)
    if not pool or (len(pool) < 7 and short_pool == "error"):
        raise ValueError(f"only {len(pool)} unique eligible negatives; cannot emit 7 without duplicates")
    rng = random.Random(seed_for(seed, str(row["query_id"]), row["query"]))
    positive = positives[rng.randrange(len(positives))]
    baseline = (rng.sample(range(len(pool)), 7) if len(pool) >= 7
                else rng.choices(range(len(pool)), k=7))
    context = make_context(row["query"], [positive])
    tags = [classify(doc, context) for doc in pool]
    queues = {label: sorted((i for i, t in enumerate(tags) if label in t["labels"]),
                             key=lambda i: (-tags[i]["score"], str(pool[i]["docid"])))
              for label in LABELS[:-1]}
    selected = []
    # Round robin encourages diversity instead of filling all slots with one type.
    for round_id in range(hard_slots):
        for label in LABELS[:-1]:
            candidate = next((i for i in queues[label] if i not in selected), None)
            if candidate is not None and len(selected) < hard_slots:
                selected.append(candidate)
    mined_count = len(selected)
    # Preserve baseline negatives where possible, keeping replacement count meaningful.
    selected.extend(i for i in baseline if i not in selected)
    selected = selected[:7]
    if len(pool) < 7:
        # Same small-pool policy as the trainer; do not amplify a rule label by
        # deliberately repeating the hardest candidate. Leave the baseline intact.
        selected = list(baseline)
        mined_count = 0
    assert len(selected) == 7
    rng.shuffle(selected)
    output = dict(row)
    output["positive_passages"] = [positive]
    output["negative_passages"] = [pool[i] for i in selected]
    return output, pool, tags, baseline, selected, mined_count, excluded


def distribution(counts):
    total = sum(counts.values())
    return {label: {"count": counts[label], "ratio": counts[label] / total if total else 0.0}
            for label in LABELS}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("train_intent_hard.jsonl.gz"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hard-slots", type=int, choices=range(8), default=4)
    parser.add_argument("--limit", type=int, default=0, help="0 means all rows")
    parser.add_argument("--audit-rows", type=int, default=100, help="number of detailed example rows")
    parser.add_argument("--invalid", choices=["error", "skip"], default="error")
    parser.add_argument("--short-pool", choices=["repeat", "error"], default="repeat",
                        help="repeat existing negatives when fewer than 7, matching trainer")
    args = parser.parse_args()
    if args.limit < 0 or args.audit_rows < 0:
        parser.error("limit and audit-rows must be nonnegative")
    output = args.output.resolve()
    if output == args.input.resolve():
        parser.error("input and output must differ")
    stats_path = Path(str(output) + ".stats.json")
    audit_path = Path(str(output) + ".audit.jsonl")
    for path in (output, stats_path, audit_path):
        if path.exists():
            parser.error(f"refusing to overwrite {path}")
    output.parent.mkdir(parents=True, exist_ok=True)
    counts = {name: Counter() for name in ("pool", "before", "after")}
    multi = {name: Counter() for name in counts}
    totals = Counter()
    excluded = Counter()
    start = time.monotonic()
    # Temp files are on the destination filesystem. No partial final dataset is
    # published if a malformed row or a write failure aborts the run.
    with tempfile.TemporaryDirectory(prefix=".intent-mining-", dir=output.parent) as tmp:
        tmpout, tmpaudit, tmpstats = (Path(tmp) / name for name in ("data.gz", "audit.jsonl", "stats.json"))
        opener = gzip.open if args.input.suffix == ".gz" else open
        with opener(args.input, "rt", encoding="utf-8") as source, open(tmpout, "wb") as raw, \
                gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0, compresslevel=3) as dest, \
                open(tmpaudit, "w", encoding="utf-8") as audit:
            stream = itertools.islice(source, args.limit) if args.limit else source
            for number, line in enumerate(stream, 1):
                totals["input_rows"] += 1
                try:
                    row = json.loads(line)
                    result, pool, tags, before, after, mined, dropped = mine_row(row, args.seed, args.hard_slots, args.short_pool)
                except (ValueError, TypeError, KeyError, AttributeError) as exc:
                    if args.invalid == "error":
                        raise ValueError(f"input line {number}: {exc}") from exc
                    totals["skipped_rows"] += 1
                    audit.write(json.dumps({"line": number, "error": str(exc)}, ensure_ascii=False) + "\n")
                    continue
                totals["output_rows"] += 1
                totals["mined_slots"] += mined
                totals["rows_with_hard_candidates"] += int(any(t["labels"] for t in tags))
                totals["short_pool_rows"] += int(len(pool) < 7)
                totals["repeated_output_slots"] += len(after) - len(set(after))
                totals["replaced_negative_slots"] += len(set(after) - set(before))
                totals["rows_changed"] += int(set(after) != set(before))
                totals["original_negative_slots"] += len(row["negative_passages"])
                excluded.update(dropped)
                for name, indices in (("pool", range(len(pool))), ("before", before), ("after", after)):
                    counts[name].update(tags[i]["primary"] for i in indices)
                    multi[name].update(label for i in indices for label in tags[i]["labels"])
                dest.write((json.dumps(result, ensure_ascii=False, separators=(",", ":")) + "\n").encode())
                if totals["output_rows"] <= args.audit_rows:
                    audit.write(json.dumps({"qid": row["query_id"], "query": row["query"],
                        "positive_docid": result["positive_passages"][0]["docid"],
                        "before_docids": [pool[i]["docid"] for i in before],
                        "after_docids": [pool[i]["docid"] for i in after],
                        "candidates": [{"docid": p["docid"], **t} for p, t in zip(pool, tags)]},
                        ensure_ascii=False) + "\n")
                if number % 10000 == 0:
                    print(f"rows={number} written={totals['output_rows']} elapsed={time.monotonic()-start:.1f}s", flush=True)
        stats = {"algorithm": "lexical_rules_v1", "input": str(args.input.resolve()),
                 "output": str(output), "seed": args.seed, "hard_slots": args.hard_slots,
                 "short_pool": args.short_pool,
                 "limit": args.limit, "totals": dict(totals), "excluded": dict(excluded),
                 "baseline": "per-query seeded uniform 7 from eligible original pool; not an actual trainer batch",
                 "primary_priority": list(LABELS),
                 "primary_distributions": {name: distribution(c) for name, c in counts.items()},
                 "multi_label_counts": {name: dict(c) for name, c in multi.items()},
                 "elapsed_seconds": round(time.monotonic() - start, 2),
                 "limitations": ["Rule labels are hypotheses, not teacher-verified relevance.",
                                 "Only existing per-query negative candidates are searched.",
                                 "No DL19 qrels or query IDs are used for mining."]}
        tmpstats.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(tmpaudit, audit_path)
        os.replace(tmpstats, stats_path)
        os.replace(tmpout, output)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
