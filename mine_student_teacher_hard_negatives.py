#!/usr/bin/env python3
"""Frozen student top1 + fresh teacher filtering. Never rewrites training data.

Output is a sidecar keyed by query and selected positive, suitable for runtime
injection while retaining all original positives and negative passages.
"""
import argparse
from collections import Counter
import gzip
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import shutil
import time

from mine_intent_hard_negatives import classify, fingerprint, make_context


def digest_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def json_write(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def row_hash(row):
    # Full row identity includes the passage order and exact text, not only qid.
    return hashlib.sha256(json.dumps(row, sort_keys=True, ensure_ascii=False,
                                    separators=(',', ':')).encode()).hexdigest()


def choose_top1(row, negative_scores):
    negatives = row['negative_passages']
    if len(negative_scores) != len(negatives):
        raise ValueError('negative scores must align with every original negative')
    if not negatives or not all(math.isfinite(v) for v in negative_scores):
        raise ValueError('empty pool or nonfinite student score')
    return min(range(len(negatives)), key=lambda i: (-negative_scores[i], str(negatives[i]['docid']), i))


def make_record(row, student_pos, student_neg, teacher_pos, teacher_neg, margin):
    index = choose_top1(row, student_neg)
    hard = row['negative_passages'][index]
    positives = row['positive_passages']
    if not (len(student_pos) == len(teacher_pos) == len(positives)):
        raise ValueError('positive scores must align with every positive')
    if not all(math.isfinite(v) for v in [*student_pos, *teacher_pos, teacher_neg]):
        raise ValueError('nonfinite score')
    conflict = (str(hard['docid']) in {str(p['docid']) for p in positives}
                or fingerprint(hard) in {fingerprint(p) for p in positives})
    accepted, rejected = [], []
    for p, sp, tp in zip(positives, student_pos, teacher_pos):
        gap = tp - teacher_neg
        decision = {
            'positive_docid': str(p['docid']), 'student_positive_score': sp,
            'teacher_positive_score': tp, 'teacher_margin': gap,
            'student_negative_above_positive': student_neg[index] > sp,
            'rule_tags': classify(hard, make_context(row['query'], [p])),
        }
        if not conflict and gap > margin:
            accepted.append(decision)
        else:
            decision['reason'] = 'known_positive_or_copy' if conflict else 'teacher_margin_not_strictly_above_threshold'
            rejected.append(decision)
    return {
        'query_id': str(row['query_id']), 'query': row['query'],
        'source_row_sha256': row_hash(row),
        'hard_docid': str(hard['docid']), 'hard_negative_index': index,
        'hard_passage': hard,
        'student_negative_score': student_neg[index],
        'teacher_negative_score': teacher_neg, 'teacher_margin_threshold': margin,
        'negative_count': len(student_neg),
        'student_negative_docids': [str(p['docid']) for p in row['negative_passages']],
        'student_negative_scores': student_neg,
        'accepted_positives': accepted, 'rejected_positives': rejected,
    }


def encode_pairs(tokenizer, query, passages, q_max_len, p_max_len):
    # Exact training preprocessor / create_one_example layout, including BOS
    # before query and document, and no tokenizer-added special tokens.
    q = tokenizer.encode('query: ' + query, add_special_tokens=False,
                         max_length=q_max_len, truncation=True)
    texts = ['document: ' + (p.get('title') or '') + ' ' + p['text'] for p in passages]
    encoded = tokenizer(texts, add_special_tokens=False, max_length=p_max_len - 3,
                        truncation=True)['input_ids']
    return [[tokenizer.bos_token_id] + q + [tokenizer.bos_token_id] + d for d in encoded]


class Scorer:
    def __init__(self, model, tokenizer, batch_size, device):
        import torch
        self.torch, self.model, self.tokenizer = torch, model, tokenizer
        self.batch_size, self.device = batch_size, device
        self.model.eval()
        self.model.config.use_cache = False
        self.model.requires_grad_(False)

    def __call__(self, sequences):
        if not sequences:
            return []
        # Length buckets avoid wasting attention/MLP work on padding; scatter
        # scores back to original qid/docid order before selecting the top1.
        order = sorted(range(len(sequences)), key=lambda i: len(sequences[i]))
        result = [None] * len(sequences)
        start = 0
        while start < len(order):
            indices = order[start:start + self.batch_size]
            batch = self.tokenizer.pad({'input_ids': [sequences[i] for i in indices]},
                                       padding=True, return_tensors='pt')
            batch = {k: v.to(self.device) for k, v in batch.items()}
            try:
                with self.torch.inference_mode():
                    values = self.model(**batch, return_dict=True).logits[:, 0].float().cpu().tolist()
            except self.torch.cuda.OutOfMemoryError:
                del batch
                self.torch.cuda.empty_cache()
                if self.batch_size <= 1:
                    raise
                self.batch_size //= 2
                print(f'OOM: reducing scoring batch size to {self.batch_size}', flush=True)
                continue
            for index, value in zip(indices, values):
                if not math.isfinite(value):
                    raise ValueError('nonfinite model score; refusing candidate output')
                result[index] = value
            start += len(indices)
        return result


def load_models(args, snapshot):
    import torch
    from peft import PeftConfig, PeftModel
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    dtype = torch.float16 if args.device.startswith('cuda') else torch.float32
    torch.manual_seed(42)
    student_tok = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    teacher_tok = AutoTokenizer.from_pretrained(args.teacher, local_files_only=True)
    for tok in (student_tok, teacher_tok):
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        tok.padding_side = 'right'
    config = PeftConfig.from_pretrained(snapshot)
    base = AutoModelForSequenceClassification.from_pretrained(
        config.base_model_name_or_path, num_labels=1, torch_dtype=dtype,
        attn_implementation='sdpa', local_files_only=True)
    base.config.pad_token_id = student_tok.pad_token_id
    model = PeftModel.from_pretrained(base, snapshot, is_trainable=False)
    # This is also the repository DL19 inference loading path.
    student = model.merge_and_unload().to(args.device)
    student = Scorer(student, student_tok, args.student_batch_size, args.device)
    teacher = AutoModelForSequenceClassification.from_pretrained(
        args.teacher, num_labels=1, torch_dtype=dtype,
        attn_implementation='sdpa', local_files_only=True).to(args.device)
    teacher.config.pad_token_id = teacher_tok.pad_token_id
    teacher = Scorer(teacher, teacher_tok, args.teacher_batch_size, args.device)
    return student, teacher


def prepare_manifest(args, output):
    # Freeze adapter/tokenizer files into this run. Never change source weights.
    snapshot = output / 'mining_student'
    manifest_path = output / 'manifest.json'
    settings = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
                if k not in ('resume', 'student_batch_size', 'teacher_batch_size', 'chunk_queries')}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest['settings'] != settings:
            raise ValueError('resume settings differ from existing mining run')
        for name, expected in manifest['snapshot_sha256'].items():
            if digest_file(snapshot / name) != expected:
                raise ValueError('frozen mining student was modified')
        if digest_file(args.input) != manifest['input_sha256']:
            raise ValueError('training input changed since mining started')
        if digest_file(__file__) != manifest['implementation_sha256']:
            raise ValueError('mining implementation changed; start a separate run')
        for path, expected in manifest['model_files_sha256'].items():
            if digest_file(path) != expected:
                raise ValueError(f'frozen base/teacher/tokenizer changed: {path}')
        return snapshot, manifest
    snapshot.mkdir(exist_ok=True)
    for path in args.student.iterdir():
        if path.is_file() and path.suffix in ('.json', '.bin', '.safetensors', '.model'):
            shutil.copy2(path, snapshot / path.name)
    hashes = {p.name: digest_file(p) for p in snapshot.iterdir() if p.is_file()}
    base = Path(json.loads((snapshot / 'adapter_config.json').read_text())['base_model_name_or_path'])
    # Content hashes bind frozen base, teacher and tokenizer to recorded scores.
    sources = [p for directory in (base, args.teacher) for p in directory.iterdir()
               if p.is_file() and p.suffix in ('.json', '.safetensors', '.bin', '.model')]
    print('Hashing input, frozen base and teacher for provenance...', flush=True)
    source_hashes = {str(p): digest_file(p) for p in sources}
    import torch
    import transformers
    import peft
    manifest = {'format': 'student_top1_teacher_margin_v1', 'settings': settings,
                'versions': {'torch': torch.__version__, 'transformers': transformers.__version__, 'peft': peft.__version__},
                'input_sha256': digest_file(args.input), 'snapshot_sha256': hashes,
                'model_files_sha256': source_hashes,
                'implementation_sha256': digest_file(__file__),
                'acceptance': 'top1 over ALL original negatives; no top2 fallback; teacher(pos)-teacher(top1)>threshold per positive',
                'encoding': 'training format: BOS query: q + BOS document: title space text; no added specials',
                'teacher_source': 'fresh inference for all positives and the selected top1; historical scores are not used'}
    json_write(manifest_path, manifest)
    return snapshot, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=Path('/hy-tmp/DeAR/data/msmarco-passage/train.jsonl.gz'))
    parser.add_argument('--student', type=Path, default=Path('/hy-tmp/DeAR/outputs/dear-kk-gap-lambda-5e-5'))
    parser.add_argument('--teacher', type=Path, default=Path('/hy-tmp/DeAR/models/llama2-13b-rankllama-teacher'))
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--teacher-margin', type=float, default=0.5)
    parser.add_argument('--q-max-len', type=int, default=32)
    parser.add_argument('--p-max-len', type=int, default=196)
    parser.add_argument('--student-batch-size', type=int, default=128)
    parser.add_argument('--teacher-batch-size', type=int, default=32)
    parser.add_argument('--chunk-queries', type=int, default=128)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--limit', type=int, default=0, help='0: complete input; otherwise smoke run')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if not math.isfinite(args.teacher_margin) or args.teacher_margin < 0:
        parser.error('teacher-margin must be finite and nonnegative')
    if min(args.chunk_queries, args.student_batch_size, args.teacher_batch_size) < 1 or args.limit < 0:
        parser.error('invalid chunk/batch/limit')
    out = args.output_dir.resolve()
    if out.exists() and any(out.iterdir()) and not args.resume:
        parser.error('output exists; use --resume for an interrupted run')
    out.mkdir(parents=True, exist_ok=True)
    # An advisory lock prevents two resumed processes writing the same run.
    import fcntl
    lock = (out / 'writer.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (out / 'COMPLETE.json').exists():
        print('Mining already complete:', out, flush=True)
        return
    snapshot, manifest = prepare_manifest(args, out)
    progress_file = out / 'progress.json'
    progress = json.loads(progress_file.read_text()) if progress_file.exists() else {}
    counts = Counter(progress.get('counts', {}))
    start_row = counts['queries_scored']
    start_time = time.monotonic()
    offset_files = {'audit': out / 'audit.partial.jsonl', 'candidates': out / 'candidates.partial.jsonl'}
    # Discard only uncommitted tails after a crash, retaining all checkpointed
    # chunks and restoring the exact original row offset.
    handles = {}
    for name, path in offset_files.items():
        if path.exists():
            handle = path.open('r+b')
            committed = progress.get('offsets', {}).get(name, 0)
            if path.stat().st_size < committed:
                raise ValueError('partial output shorter than committed checkpoint')
            handle.truncate(committed)
            handle.seek(committed)
        else:
            handle = path.open('w+b')
        handles[name] = handle
    student, teacher = load_models(args, snapshot)
    print('Both models frozen; starting inference, no optimizer or training.', flush=True)
    with gzip.open(args.input, 'rt', encoding='utf-8') as source:
        stream = itertools.islice(source, start_row, args.limit or None)
        while True:
            rows = [json.loads(line) for line in itertools.islice(stream, args.chunk_queries)]
            if not rows:
                break
            student_seqs, lengths = [], []
            for row in rows:
                positives, negatives = row['positive_passages'], row['negative_passages']
                if not positives or not negatives:
                    raise ValueError(f'empty passage pool qid={row["query_id"]}')
                seqs = encode_pairs(student.tokenizer, row['query'], positives + negatives,
                                    args.q_max_len, args.p_max_len)
                student_seqs.extend(seqs)
                lengths.append((len(positives), len(negatives)))
            student_values = student(student_seqs)
            teacher_seqs, segments, offset = [], [], 0
            for row, (npos, nneg) in zip(rows, lengths):
                sp = student_values[offset:offset+npos]
                sn = student_values[offset+npos:offset+npos+nneg]
                offset += npos + nneg
                index = choose_top1(row, sn)
                teacher_seqs.extend(encode_pairs(teacher.tokenizer, row['query'],
                    row['positive_passages'] + [row['negative_passages'][index]],
                    args.q_max_len, args.p_max_len))
                segments.append((sp, sn))
            teacher_values = teacher(teacher_seqs)
            offset = 0
            for row, (sp, sn), (npos, nneg) in zip(rows, segments, lengths):
                tp = teacher_values[offset:offset+npos]
                tn = teacher_values[offset+npos]
                offset += npos + 1
                record = make_record(row, sp, sn, tp, tn, args.teacher_margin)
                record['source_line'] = counts['queries_scored'] + 1
                counts['queries_scored'] += 1
                counts['negative_passages_scored'] += nneg
                counts['positive_passages_scored'] += npos
                counts['accepted_positive_pairs'] += len(record['accepted_positives'])
                counts['rejected_positive_pairs'] += len(record['rejected_positives'])
                counts['accepted_student_error_pairs'] += sum(p['student_negative_above_positive'] for p in record['accepted_positives'])
                payload = (json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n').encode()
                handles['audit'].write(payload)
                if record['accepted_positives']:
                    counts['queries_with_candidate'] += 1
                    # Keep candidate artifact compact but preserve per-positive
                    # evidence and original row hash for later runtime joins.
                    candidate = {k: v for k, v in record.items() if k not in ('student_negative_docids','student_negative_scores','rejected_positives')}
                    handles['candidates'].write((json.dumps(candidate, ensure_ascii=False, allow_nan=False)+'\n').encode())
                else:
                    counts['queries_without_candidate'] += 1
            for handle in handles.values():
                handle.flush()
                os.fsync(handle.fileno())
            elapsed = time.monotonic() - start_time
            progress = {'counts': dict(counts), 'offsets': {k: v.tell() for k,v in handles.items()},
                        'session_elapsed_seconds': elapsed,
                        'queries_per_second': (counts['queries_scored']-start_row)/max(elapsed,1e-9),
                        'student_batch_size': student.batch_size, 'teacher_batch_size': teacher.batch_size}
            json_write(progress_file, progress)
            print(json.dumps(progress, ensure_ascii=False), flush=True)
    for handle in handles.values():
        handle.close()
    for name, path in offset_files.items():
        shutil.copyfile(path, out / (name + '.jsonl'))
    # Consumers must require this marker and verify the candidate digest. Partial
    # artifacts are deliberately named differently and are never training input.
    complete = {**progress, 'complete': True, 'full_dataset': args.limit == 0,
                'candidates_sha256': digest_file(out / 'candidates.jsonl'),
                'audit_sha256': digest_file(out / 'audit.jsonl'),
                'manifest_sha256': digest_file(out / 'manifest.json')}
    json_write(out / 'COMPLETE.json', complete)
    print('COMPLETE', json.dumps(complete, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
