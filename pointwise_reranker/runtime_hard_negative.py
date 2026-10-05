"""Read completed mining sidecars; gate BEFORE Dataset changes a random group."""
import hashlib
import json
import math
import logging
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load_candidates(directory, train_paths, q_max_len, p_max_len):
    directory = Path(directory)
    complete = json.loads((directory / 'COMPLETE.json').read_text())
    if not complete.get('complete') or not complete.get('full_dataset'):
        raise ValueError('Runtime injection requires a completed FULL mining run')
    if sha256(directory / 'manifest.json') != complete['manifest_sha256']:
        raise ValueError('mining manifest digest mismatch')
    manifest = json.loads((directory / 'manifest.json').read_text())
    if manifest['format'] != 'student_top1_teacher_margin_v1':
        raise ValueError('Unsupported mining format')
    settings = manifest['settings']
    if settings['q_max_len'] != q_max_len or settings['p_max_len'] != p_max_len:
        raise ValueError('Mining/training tokenization lengths differ')
    if len(train_paths or []) != 1 or sha256(train_paths[0]) != manifest['input_sha256']:
        raise ValueError('Mining sidecar belongs to a different training input')
    path = directory / 'candidates.jsonl'
    if sha256(path) != complete['candidates_sha256']:
        raise ValueError('candidate digest mismatch')
    candidates = {}
    source_count = total_pairs = 0
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            qid = row['query_id']
            source_count += 1
            if qid in candidates:
                raise ValueError(f'duplicate candidate qid={qid}')
            threshold = row['teacher_margin_threshold']
            if not math.isfinite(threshold) or threshold < 0.5:
                raise ValueError('Teacher margin threshold must be at least 0.5')
            positives = []
            for pair in row['accepted_positives']:
                pos, neg = pair['teacher_positive_score'], row['teacher_negative_score']
                if not all(math.isfinite(v) for v in (pos, neg, pair['teacher_margin'])):
                    raise ValueError('nonfinite teacher evidence')
                if pos - neg <= threshold or not math.isclose(pos - neg, pair['teacher_margin'], abs_tol=1e-8):
                    raise ValueError('candidate failed teacher evidence validation')
                total_pairs += 1
                positives.append(pair['positive_docid'])
            if not row['accepted_positives']:
                raise ValueError('candidate has no approved positive')
            candidates[qid] = (row['source_row_sha256'], row['hard_negative_index'], row['hard_docid'], positives)
    if source_count != complete['counts']['queries_with_candidate']:
        raise ValueError('candidate count mismatch')
    logging.getLogger(__name__).info(
        'HARD_CANDIDATES source=teacher_approved_top1 teacher_pairs=%d eligible_queries=%d',
        total_pairs, source_count)
    return candidates


def candidate_indices(example, candidates):
    approved = [-1] * len(example['positive_passages'])
    record = candidates.get(str(example['query_id']))
    if record is None:
        return approved
    expected_hash, index, did, positive_docids = record
    # Dataset.map supplies a LazyRow Mapping, which JSON cannot serialize
    # directly. Materialize all fields while preserving the mining row hash.
    digest = hashlib.sha256(json.dumps(dict(example), sort_keys=True, ensure_ascii=False,
                                     separators=(',', ':')).encode()).hexdigest()
    if digest != expected_hash or str(example['negative_passages'][index]['docid']) != did:
        raise ValueError('candidate qid/docid/text/order mismatch')
    for i, pos in enumerate(example['positive_passages']):
        if str(pos['docid']) in positive_docids:
            approved[i] = index
    return approved


def inject_one(neg_indices, candidate_index, probability, rng):
    """Preserve baseline unless an approved top1 passes the probability draw.

    If top1 is already in the random group, leave the group untouched: it is an
    ordinary random occurrence, not an injected document. At most one changed
    slot, including short pools sampled with replacement.
    """
    if not 0.0 <= probability <= 1.0:
        raise ValueError('hard_negative_probability must be in [0,1]')
    indices = list(neg_indices)
    if probability == 0 or candidate_index < 0 or not indices:
        return indices, -1
    if rng.random() >= probability or candidate_index in indices:
        return indices, -1
    position = rng.randrange(len(indices))
    indices[position] = candidate_index
    return indices, position


def injection_decision(neg_indices, candidate_index, probability, rng):
    """Exclusive reasons: missing -> probability -> present -> replaced.

    Always consume two independent-stream draws, even for rejected samples.
    This keeps probability and slot draws aligned across probability experiments.
    """
    if not 0 <= probability <= 1:
        raise ValueError('hard_negative_probability must be in [0,1]')
    u, slot_u = rng.random(), rng.random()
    indices = list(neg_indices)
    if candidate_index < 0:
        return indices, -1, 'missing_candidate'
    if u >= probability:
        return indices, -1, 'probability_missed'
    if candidate_index in indices:
        return indices, -1, 'already_present'
    if not indices:
        raise ValueError('empty negative group')
    slot = int(slot_u * len(indices))
    indices[slot] = candidate_index
    return indices, slot, 'replaced'
