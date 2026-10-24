from __future__ import annotations

import contextlib
import gzip
import hashlib
import json
import os
import sqlite3
from pathlib import Path
from typing import Iterable, Iterator, Sequence


def open_text(path: str | Path, mode: str):
    path = str(path)
    kwargs = {"encoding": "utf-8"} if "b" not in mode else {}
    return gzip.open(path, mode, **kwargs) if path.endswith(".gz") else open(path, mode, **kwargs)


def iter_jsonl(path: str | Path) -> Iterator[dict]:
    with open_text(path, "rt") as reader:
        for line_number, line in enumerate(reader, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc


@contextlib.contextmanager
def atomic_jsonl_writer(path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = ".tmp.gz" if str(path).endswith(".gz") else ".tmp"
    temporary = path.with_name(path.name + f".{os.getpid()}{suffix}")
    try:
        with open_text(temporary, "wt") as writer:
            yield writer
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_jsonl_row(writer, row: dict) -> None:
    writer.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def normalized_text(text: str) -> str:
    return " ".join(str(text).lower().split())


def text_hash(text: str) -> str:
    return hashlib.sha1(normalized_text(text).encode("utf-8")).hexdigest()


def passage_id(passage: dict) -> str:
    value = passage.get("docid", passage.get("id", ""))
    return str(value)


def passage_text(passage: dict) -> str:
    title = str(passage.get("title", "")).strip()
    body = str(passage.get("text", "")).strip()
    return (title + " " + body).strip()


def batched(items: Iterable, size: int):
    batch = []
    for item in items:
        batch.append(item)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


class CorpusStore:
    def __init__(self, path: str | Path):
        self.connection = sqlite3.connect(f"file:{Path(path).resolve()}?mode=ro", uri=True)

    def close(self) -> None:
        self.connection.close()

    def fetch_many(self, docids: Sequence[str]) -> dict[str, dict]:
        result: dict[str, dict] = {}
        unique = list(dict.fromkeys(str(x) for x in docids))
        for start in range(0, len(unique), 900):
            chunk = unique[start : start + 900]
            placeholders = ",".join("?" for _ in chunk)
            query = f"SELECT docid, text, text_hash FROM docs WHERE docid IN ({placeholders})"
            for docid, text, digest in self.connection.execute(query, chunk):
                result[str(docid)] = {
                    "docid": str(docid),
                    "title": "",
                    "text": text,
                    "text_hash": digest,
                }
        return result

