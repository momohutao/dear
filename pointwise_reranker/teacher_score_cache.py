"""QID/docid-addressed RankLLaMA score cache for distribution TTA."""

from __future__ import annotations

import json
import os
from typing import Dict, Iterable, Mapping, Optional, Tuple


CacheKey = Tuple[str, str, str, int]


def teacher_identifier(path: str) -> str:
    """A stable local checkpoint identifier; never rely on candidate position."""
    return os.path.abspath(path)


def load_teacher_score_cache(
    path: Optional[str], teacher_id: str, candidate_k: int
) -> Dict[CacheKey, float]:
    if not path or not os.path.isfile(path):
        return {}
    scores: Dict[CacheKey, float] = {}
    with open(path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                # Pre-hard-gate cache records did not carry candidate_k.
                # They are valid legacy records, but cannot safely satisfy a
                # candidate-K-addressed lookup, so ignore and recompute them.
                if not isinstance(record, dict) or "candidate_k" not in record:
                    continue
                key = (
                    str(record["qid"]),
                    str(record["docid"]),
                    str(record["teacher_id"]),
                    int(record["candidate_k"]),
                )
                score = float(record["score"])
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                raise ValueError(f"Invalid teacher cache record at {path}:{line_number}") from error
            if key[2] == teacher_id and key[3] == int(candidate_k):
                scores[key] = score
    return scores


def append_teacher_score_cache(
    path: Optional[str], teacher_id: str, candidate_k: int, records: Iterable[Mapping[str, object]]
) -> int:
    if not path:
        return 0
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    count = 0
    with open(path, "a", encoding="utf-8") as handle:
        for record in records:
            payload = {
                "qid": str(record["qid"]),
                "docid": str(record["docid"]),
                "teacher_id": teacher_id,
                "candidate_k": int(candidate_k),
                "score": float(record["score"]),
            }
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count
