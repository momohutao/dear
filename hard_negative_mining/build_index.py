#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import faiss
import numpy as np


def load_parts(directory: Path):
    metadata = [json.loads(path.read_text()) for path in sorted(directory.glob("part*.meta.json"))]
    if not metadata:
        raise FileNotFoundError(f"No part*.meta.json in {directory}")
    keys = ("dimension", "model", "pooling", "query_prefix", "passage_prefix", "max_length")
    for item in metadata[1:]:
        if any(item[key] != metadata[0][key] for key in keys):
            raise ValueError("Embedding shard configurations do not match")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a cosine/IP FAISS index from float16 shards.")
    parser.add_argument("--embedding-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--index-type", choices=("ivfpq", "flat"), default="ivfpq")
    parser.add_argument("--nlist", type=int, default=16384)
    parser.add_argument("--nprobe", type=int, default=32)
    parser.add_argument("--pq-m", type=int, default=96)
    parser.add_argument("--pq-bits", type=int, default=8)
    parser.add_argument("--train-size", type=int, default=1000000)
    parser.add_argument("--add-batch-size", type=int, default=100000)
    parser.add_argument("--threads", type=int, default=32)
    args = parser.parse_args()
    source = Path(args.embedding_dir)
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    parts = load_parts(source)
    dimension = int(parts[0]["dimension"])
    total = sum(int(item["count"]) for item in parts)
    faiss.omp_set_num_threads(args.threads)

    if args.index_type == "flat":
        index = faiss.IndexIDMap2(faiss.IndexFlatIP(dimension))
        actual_nlist = None
    else:
        if dimension % args.pq_m:
            raise ValueError(f"dimension {dimension} must be divisible by pq-m {args.pq_m}")
        actual_nlist = min(args.nlist, max(1, total // 50))
        quantizer = faiss.IndexFlatIP(dimension)
        index = faiss.IndexIVFPQ(
            quantizer, dimension, actual_nlist, args.pq_m, args.pq_bits, faiss.METRIC_INNER_PRODUCT
        )
        rng = np.random.default_rng(42)
        samples = []
        target = min(args.train_size, total)
        for item in parts:
            count = int(item["count"])
            take = max(1, round(target * count / total))
            take = min(take, count)
            embeddings = np.memmap(source / item["embedding_file"], dtype="<f2", mode="r", shape=(count, dimension))
            indices = rng.choice(count, size=take, replace=False)
            samples.append(np.asarray(embeddings[indices], dtype="float32"))
        training = np.concatenate(samples, axis=0)[:target]
        print(f"training IVF-PQ with {len(training)} vectors; nlist={actual_nlist}", flush=True)
        index.train(training)

    added = 0
    for item in parts:
        count = int(item["count"])
        embeddings = np.memmap(source / item["embedding_file"], dtype="<f2", mode="r", shape=(count, dimension))
        ids = np.memmap(source / item["docid_file"], dtype="<i8", mode="r", shape=(count,))
        for start in range(0, count, args.add_batch_size):
            end = min(start + args.add_batch_size, count)
            index.add_with_ids(np.asarray(embeddings[start:end], dtype="float32"), np.asarray(ids[start:end]))
            added += end - start
            print(f"indexed={added}/{total}", flush=True)
    if hasattr(index, "nprobe"):
        index.nprobe = args.nprobe
    temporary = output.with_name(output.name + f".{os.getpid()}.tmp")
    faiss.write_index(index, str(temporary))
    os.replace(temporary, output)
    index_metadata = dict(parts[0])
    index_metadata.update({
        "total": total,
        "index_type": args.index_type,
        "nlist": actual_nlist,
        "nprobe": args.nprobe,
        "index_file": output.name,
    })
    output.with_suffix(output.suffix + ".meta.json").write_text(
        json.dumps(index_metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"done: {total} vectors -> {output}")


if __name__ == "__main__":
    main()

