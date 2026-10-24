#!/usr/bin/env python3
"""Convert MVP validation JSONL into DeAR inference input and TREC qrels."""

import argparse
import json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--qrels", required=True)
    args = parser.parse_args()

    rows = 0
    candidates = 0
    with open(args.input, encoding="utf-8") as source, \
            open(args.output, "w", encoding="utf-8") as output, \
            open(args.qrels, "w", encoding="utf-8") as qrels:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            qid = str(record["qid"])
            query = record["q_text"]
            labels = record["qrels"]
            for item in record["bm25_results"]:
                docid = str(item["pid"])
                output.write(json.dumps({
                    "query_id": qid,
                    "docid": docid,
                    "query": query,
                    "title": "",
                    "text": item["text"],
                }, ensure_ascii=False) + "\n")
                qrels.write(f"{qid} 0 {docid} {labels.get(item['pid'], labels.get(docid, 0))}\n")
                candidates += 1
            rows += 1
    print(f"queries={rows} candidates={candidates}")


if __name__ == "__main__":
    main()
