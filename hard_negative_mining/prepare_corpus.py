#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sqlite3
from pathlib import Path

from io_utils import text_hash


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a read-only-friendly MS MARCO SQLite doc store.")
    parser.add_argument("--collection", required=True, help="MS MARCO collection.tsv")
    parser.add_argument("--output", required=True, help="Output corpus.sqlite")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--commit-every", type=int, default=50000)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    output = Path(args.output)
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists: {output}; pass --overwrite to replace it")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + f".{os.getpid()}.tmp")
    if temporary.exists():
        temporary.unlink()
    connection = sqlite3.connect(temporary)
    try:
        connection.executescript(
            "PRAGMA journal_mode=OFF; PRAGMA synchronous=OFF; PRAGMA temp_store=MEMORY;"
            "CREATE TABLE docs(docid TEXT PRIMARY KEY, text TEXT NOT NULL, text_hash TEXT NOT NULL);"
        )
        pending = []
        count = 0
        with open(args.collection, "rt", encoding="utf-8") as reader:
            for line_number, line in enumerate(reader, 1):
                if args.limit and count >= args.limit:
                    break
                try:
                    docid, text = line.rstrip("\n").split("\t", 1)
                except ValueError as exc:
                    raise ValueError(f"Malformed collection row {line_number}") from exc
                pending.append((docid, text, text_hash(text)))
                count += 1
                if len(pending) >= args.commit_every:
                    connection.executemany("INSERT INTO docs VALUES(?,?,?)", pending)
                    connection.commit()
                    pending.clear()
                    print(f"stored={count}", flush=True)
        if pending:
            connection.executemany("INSERT INTO docs VALUES(?,?,?)", pending)
            connection.commit()
        connection.close()
        os.replace(temporary, output)
        print(f"done: {count} documents -> {output}")
    finally:
        try:
            connection.close()
        except Exception:
            pass
        if temporary.exists():
            temporary.unlink()


if __name__ == "__main__":
    main()

