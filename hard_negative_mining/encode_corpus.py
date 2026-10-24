#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from dense_encoder import DenseEncoder


def main() -> None:
    parser = argparse.ArgumentParser(description="Encode one deterministic shard of MS MARCO passages.")
    parser.add_argument("--collection", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="float16")
    parser.add_argument("--pooling", choices=("mean", "cls"), default="mean")
    parser.add_argument("--passage-prefix", default="passage: ")
    parser.add_argument("--query-prefix", default="query: ")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0, help="Per-shard smoke-test limit; 0 means all")
    args = parser.parse_args()
    if not 0 <= args.shard_id < args.num_shards:
        parser.error("shard-id must be in [0, num-shards)")

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    stem = f"part{args.shard_id:05d}"
    embedding_path = output / f"embeddings.{stem}.f16"
    docid_path = output / f"docids.{stem}.i64"
    metadata_path = output / f"{stem}.meta.json"
    for path in (embedding_path, docid_path, metadata_path):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}")
    embedding_tmp = embedding_path.with_name(embedding_path.name + f".{os.getpid()}.tmp")
    docid_tmp = docid_path.with_name(docid_path.name + f".{os.getpid()}.tmp")

    encoder = DenseEncoder(args.model, args.device, args.dtype, args.pooling, args.max_length)
    texts, docids, count = [], [], 0
    with open(args.collection, "rt", encoding="utf-8") as reader, open(embedding_tmp, "wb") as eout, open(docid_tmp, "wb") as iout:
        for line_number, line in enumerate(reader):
            if line_number % args.num_shards != args.shard_id:
                continue
            if args.limit and count >= args.limit:
                break
            docid, text = line.rstrip("\n").split("\t", 1)
            try:
                docids.append(int(docid))
            except ValueError as exc:
                raise ValueError(f"FAISS ids must be integers, got {docid!r}") from exc
            texts.append(args.passage_prefix + text)
            if len(texts) == args.batch_size:
                encoder.encode(texts).numpy().astype("<f2").tofile(eout)
                np.asarray(docids, dtype="<i8").tofile(iout)
                count += len(texts)
                texts.clear(); docids.clear()
                if count % (args.batch_size * 100) == 0:
                    print(f"shard={args.shard_id} encoded={count}", flush=True)
        if texts:
            encoder.encode(texts).numpy().astype("<f2").tofile(eout)
            np.asarray(docids, dtype="<i8").tofile(iout)
            count += len(texts)

    os.replace(embedding_tmp, embedding_path)
    os.replace(docid_tmp, docid_path)
    metadata = {
        "shard_id": args.shard_id,
        "num_shards": args.num_shards,
        "count": count,
        "dimension": encoder.dimension,
        "model": str(Path(args.model).resolve()) if Path(args.model).exists() else args.model,
        "pooling": args.pooling,
        "max_length": args.max_length,
        "passage_prefix": args.passage_prefix,
        "query_prefix": args.query_prefix,
        "dtype": "float16",
        "embedding_file": embedding_path.name,
        "docid_file": docid_path.name,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"done: shard={args.shard_id} count={count}")


if __name__ == "__main__":
    main()

