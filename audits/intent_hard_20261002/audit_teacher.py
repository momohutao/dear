"""Audit existing score artifacts only; never run a teacher model."""
from collections import Counter
import gzip
import hashlib
import itertools
import json
import math
from pathlib import Path
import time

OUT=Path(__file__).resolve().parent
BASE=Path('/hy-tmp/DeAR')
counts=Counter()
errors=[]
start=time.monotonic()
def fail(key,line):
    counts[key]+=1
    if len(errors)<50: errors.append({'check':key,'line':line})
def cached_rows():
    for p in sorted((BASE/'data').glob('train.teacher.part*.jsonl.gz')):
        n=0
        with gzip.open(p,'rt') as f:
            for line in f:
                n+=1
                yield json.loads(line)
        counts[p.name+'_rows']=n
def core(p):
    return {k:v for k,v in p.items() if k not in ('teacher_score','teacher_margin')}
with gzip.open(BASE/'data/msmarco-passage/train.jsonl.gz','rt') as source:
    for line,(raw,cache) in enumerate(itertools.zip_longest(source,cached_rows()),1):
        if raw is None or cache is None:
            fail('unpaired_row',line);continue
        row=json.loads(raw)
        counts['paired_rows']+=1
        if row['query_id']!=cache['query_id'] or row['query']!=cache['query']: fail('query_order_or_text_mismatch',line)
        for field in ('positive_passages','negative_passages'):
            original,cached=row[field],cache[field]
            if len(original)!=len(cached): fail('passage_count_mismatch',line)
            for index,(a,b) in enumerate(itertools.zip_longest(original,cached)):
                if a is None or b is None: continue
                counts['checked_passages']+=1
                if str(a['docid'])!=str(b['docid']): fail('docid_order_mismatch',line)
                if a!=core(b): fail('passage_content_mismatch',line)
                score=b.get('teacher_score')
                if not isinstance(score,(int,float)) or not math.isfinite(score):
                    fail('invalid_teacher_score',line)
                    counts[field+'_invalid_score']+=1
                    if 'teacher_score' not in b:
                        counts[field+'_missing_score']+=1
                        counts['missing_score_positive_index_gt_zero']+=field=='positive_passages' and index>0
                    else: counts['present_but_invalid_score']+=1
                    if len(errors)<50:
                        errors[-1].update({'docid':str(b['docid']),'field':field,'index':index,'teacher_score':score})
                if field=='negative_passages':
                    counts['checked_margins']+=1
                    # This cache's margins are referenced to its first positive.
                    expected=cache['positive_passages'][0]['teacher_score']-score
                    if not math.isclose(expected,b.get('teacher_margin',math.inf),abs_tol=1e-8): fail('teacher_margin_arithmetic_mismatch',line)
        if line%20000==0: print(f'rows={line} elapsed={time.monotonic()-start:.1f}s',flush=True)
inference=[]
for path in sorted((BASE/'cache/teacher_scores').glob('*.jsonl')):
    c=Counter(); values={}; teachers=set(); ks=set(); qids=set()
    with path.open() as f:
        for ln,text in enumerate(f,1):
            r=json.loads(text);c['records']+=1
            key=(str(r['qid']),str(r['docid']),str(r['teacher_id']),r.get('candidate_k'))
            if key in values:
                c['duplicate_keys']+=1
                if values[key]!=r['score']:c['conflicting_scores']+=1
            values[key]=r['score'];teachers.add(key[2]);ks.add(key[3]);qids.add(key[0])
            if 'candidate_k' not in r:c['legacy_without_candidate_k']+=1
            if not math.isfinite(float(r['score'])):c['nonfinite_scores']+=1
    inference.append({'path':str(path),'counts':dict(c),'unique_keys':len(values),'unique_qids':len(qids),'teacher_ids':sorted(teachers),'candidate_k':sorted(ks,key=lambda k:(k is not None,k or 0))})
report={'counts':dict(counts),'errors':errors,'inference_caches':inference,'limitations':['Training entry uses neither the scored shards nor inference caches.','This checks score-to-docid attachment and margin arithmetic; it cannot prove score numerical correctness without rerunning the teacher.','Cached margins refer to cached first positive; recompute if choosing a different positive.','Inference keys omit query/doc text, tokenizer and truncation settings; changing these at the same key can yield stale scores.'],'elapsed_seconds':round(time.monotonic()-start,2)}
(OUT/'teacher_cache_summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(report,ensure_ascii=False,indent=2))
