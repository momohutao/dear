"""Read-only full-file audit; writes only local audit artifacts. CPU/stdlib only."""
import ast
from collections import Counter
import gzip
import hashlib
import itertools
import json
from pathlib import Path
import random
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from mine_intent_hard_negatives import fingerprint, seed_for, make_context, classify, mine_row, LABELS

OUT = Path(__file__).resolve().parent
DATA = Path('/hy-tmp/DeAR/data/msmarco-passage')
stats = json.loads((DATA / 'train_intent_hard.jsonl.gz.stats.json').read_text())
counts = Counter()
hists = {k: Counter() for k in ('old_positive_pool', 'old_negative_pool', 'eligible_negative_pool', 'new_positive_pool', 'new_negative_pool', 'replacement_slots')}
primary = Counter()
multi = Counter()
excluded = Counter()
seen_old, seen_new = set(), set()
digests = [hashlib.sha256(), hashlib.sha256()]
samples = {label: [] for label in LABELS[:-1]}
sample_seen = Counter()
sample_rng = {label: random.Random(20261002 + i) for i, label in enumerate(LABELS[:-1])}
errors = []
start = time.monotonic()

def fail(name, line):
    counts[name] += 1
    if len(errors) < 100:
        errors.append({'check': name, 'line': line})

with gzip.open(stats['input'], 'rt') as source, gzip.open(stats['output'], 'rt') as target:
    for line, (a, b) in enumerate(itertools.zip_longest(source, target), 1):
        if a is None or b is None:
            fail('unpaired_rows', line)
            counts['old_rows'] += a is not None
            counts['new_rows'] += b is not None
            continue
        old, new = json.loads(a), json.loads(b)
        counts['old_rows'] += 1
        counts['new_rows'] += 1
        for j, row in enumerate((old, new)):
            digests[j].update((json.dumps([row['query_id'], row['query']], ensure_ascii=False) + '\n').encode())
        oq, nq = str(old['query_id']), str(new['query_id'])
        if oq in seen_old: counts['old_duplicate_qid_rows'] += 1
        if nq in seen_new: counts['new_duplicate_qid_rows'] += 1
        seen_old.add(oq); seen_new.add(nq)
        if old['query_id'] != new['query_id']: fail('qid_order_mismatch', line)
        if old['query'] != new['query']: fail('query_text_mismatch', line)
        if {k:v for k,v in old.items() if k not in ('positive_passages','negative_passages')} != {k:v for k,v in new.items() if k not in ('positive_passages','negative_passages')}:
            fail('other_fields_changed', line)
        op, on = old['positive_passages'], old['negative_passages']
        np, nn = new['positive_passages'], new['negative_passages']
        for key, value in (('old_positive_pool',len(op)),('old_negative_pool',len(on)),('new_positive_pool',len(np)),('new_negative_pool',len(nn))): hists[key][value] += 1
        counts['old_positive_slots'] += len(op)
        counts['old_negative_slots'] += len(on)
        counts['new_negative_slots'] += len(nn)
        counts['multi_positive_rows'] += len(op) > 1
        counts['positive_pool_reduced_rows'] += op != np
        if len(np)!=1 or len(nn)!=7: fail('new_group_size_mismatch',line)
        rng = random.Random(seed_for(stats['seed'], oq, old['query']))
        expected_pos = op[rng.randrange(len(op))]
        if np != [expected_pos]: fail('positive_seed_replay_mismatch',line)
        counts['positive_differs_from_original_first'] += np[0] != op[0]
        ids, texts, pool = set(), set(), []
        posids = {str(p['docid']) for p in op}
        postexts = {fingerprint(p) for p in op}
        for doc in on:
            did, txt = str(doc['docid']), fingerprint(doc)
            if did in posids or txt in postexts: excluded['known_positive_or_positive_copy'] += 1
            elif did in ids or txt in texts: excluded['duplicate_negative'] += 1
            else:
                pool.append(doc); ids.add(did); texts.add(txt)
        counts['eligible_negative_slots'] += len(pool)
        hists['eligible_negative_pool'][len(pool)] += 1
        counts['short_pool_rows'] += len(pool) < 7
        baseline = rng.sample(range(len(pool)),7) if len(pool)>=7 else rng.choices(range(len(pool)), k=7)
        before = {str(pool[i]['docid']) for i in baseline}
        after = {str(p['docid']) for p in nn}
        replacements = len(after-before)
        counts['replaced_negative_slots'] += replacements
        counts['rows_changed'] += before != after
        hists['replacement_slots'][replacements] += 1
        counts['repeat_docid_slots'] += len(nn)-len(after)
        counts['repeat_text_slots'] += len(nn)-len({fingerprint(p) for p in nn})
        counts['repeat_docid_rows'] += len(nn)>len(after)
        counts['repeat_text_rows'] += len(nn)>len({fingerprint(p) for p in nn})
        counts['new_unique_negative_slots'] += len(after)
        pool_map = {str(p['docid']):p for p in pool}
        for p in nn:
            if p != pool_map.get(str(p['docid'])): fail('negative_not_original_eligible_document',line)
            if str(p['docid']) in posids or fingerprint(p) in postexts: fail('negative_is_known_positive',line)
            if p.get('title') is None: counts['null_or_missing_negative_title'] += 1
        if np[0].get('title') is None: counts['null_or_missing_positive_title'] += 1
        context = make_context(new['query'],np)
        tags = [classify(p,context) for p in nn]
        primary.update(t['primary'] for t in tags)
        multi.update(label for t in tags for label in t['labels'])
        counts['hard_labeled_slots'] += sum(bool(t['labels']) for t in tags)
        counts['hard_labeled_rows'] += any(t['labels'] for t in tags)
        counts['replacement_slots_with_rule_label'] += sum(bool(t['labels']) for p,t in zip(nn,tags) if str(p['docid']) not in before)
        for label in samples:
            candidates = [(p,t) for p,t in zip(nn,tags) if label in t['labels']]
            if not candidates: continue
            sample_seen[label] += 1
            srng = sample_rng[label]
            pick = len(samples[label]) if len(samples[label])<30 else srng.randrange(sample_seen[label])
            if pick >= 30: continue
            p,t = srng.choice(candidates)
            record = {'rule':label,'line':line,'query_id':nq,'query':new['query'],'positive':np[0],'negative':p,'hit':t,'context':{'content_tokens':sorted(context[0]),'constraints':context[1],'expected_domain':context[2],'expected_intent':context[3]},'baseline_docids':sorted(before),'negative_was_replacement':str(p['docid']) not in before,'all_selected_negatives':nn,'original_row':old,'new_row':new}
            if pick == len(samples[label]): samples[label].append(record)
            else: samples[label][pick] = record
        if line<=100:
            if mine_row(old, stats['seed'], stats['hard_slots'],stats['short_pool'])[0] != new: fail('first_100_full_mining_replay_mismatch',line)
        if line%10000==0: print(f'rows={line} elapsed={time.monotonic()-start:.1f}s',flush=True)

# Execute the current loader's actual __getitem__ body with identity document
# encodings. This isolates selection/alignment without model/tokenizer loading.
tree = ast.parse((ROOT/'pointwise_reranker/data.py').read_text())
cls = next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='RerankerTrainDataset')
method = next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='__getitem__')
method.returns = None
for arg in method.args.args: arg.annotation = None
ns = {'random':random}
exec(compile(ast.fix_missing_locations(ast.Module(body=[method],type_ignores=[])), str(ROOT/'pointwise_reranker/data.py'), 'exec'),ns)
loader_records=[]
for label, records in samples.items():
    for record in records:
        old,new = record.pop('original_row'),record.pop('new_row')
        replay = mine_row(old,stats['seed'],stats['hard_slots'],stats['short_pool'])
        if replay[0]!=new: fail('random_sample_full_mining_replay_mismatch',record['line'])
        for name,row in (('old',old),('new',new)):
            group={'query_student':[], 'query_teacher':[], 'repro_qid':str(row['query_id'])}
            for kind,field in (('positive','positive_passages'),('negative','negative_passages')):
                group[f'repro_{kind}_docids']=[str(p['docid']) for p in row[field]]
                for who in ('student','teacher'):
                    group[f'{kind}s_{who}']=[str(p['docid']) for p in row[field]]
            dummy=SimpleNamespace(train_data=[group],data_args=SimpleNamespace(train_n_passages=8,positive_passage_no_shuffle=False,negative_passage_no_shuffle=False),tokenizer_student='s',tokenizer_teacher='t',create_one_example=lambda q,p,t:p)
            draws=[]
            random.seed(42)
            for _ in range(20):
                s,t,metadata=ns['__getitem__'](dummy,0)
                if s!=t or s!=[metadata['positive_docid']]+metadata['negative_docids']: fail('loader_student_teacher_docid_mismatch',record['line'])
                draws.append(metadata)
            loader_records.append({'rule':label,'line':record['line'],'dataset':name,'distinct_positive_docids':len({x['positive_docid'] for x in draws}),'distinct_negative_docids':len({d for x in draws for d in x['negative_docids']}),'distinct_negative_sets':len({tuple(sorted(x['negative_docids'])) for x in draws}),'distinct_negative_orders':len({tuple(x['negative_docids']) for x in draws}),'draws':draws})

comparisons={k:{'observed':counts[k],'stored':stats['totals'][k],'equal':counts[k]==stats['totals'][k]} for k in ('short_pool_rows','replaced_negative_slots','rows_changed')}
comparisons['repeated_output_slots']={'observed':counts['repeat_docid_slots'],'stored':stats['totals']['repeated_output_slots'],'equal':counts['repeat_docid_slots']==stats['totals']['repeated_output_slots']}
report={'input':stats['input'],'output':stats['output'],'seed':stats['seed'],'counts':dict(counts),'unique_qids':{'old':len(seen_old),'new':len(seen_new)},'ordered_query_sha256':{'old':digests[0].hexdigest(),'new':digests[1].hexdigest()},'histograms':{k:dict(sorted(v.items())) for k,v in hists.items()},'excluded':dict(excluded),'after_primary_counts':dict(primary),'after_multi_label_counts':dict(multi),'stored_stats_checks':comparisons,'stored_primary_equal':dict(primary)=={k:v['count'] for k,v in stats['primary_distributions']['after'].items()},'stored_multi_equal':dict(multi)==stats['multi_label_counts']['after'],'stored_exclusions_equal':dict(excluded)==stats['excluded'],'samples':{'seed':20261002,'method':'uniform reservoir of matching query rows per rule; random matching selected negative within row; labels may overlap','matching_rows':dict(sample_seen),'counts':{k:len(v) for k,v in samples.items()}},'mining_full_replay_rows':100+sum(map(len,samples.values())),'loader_probe':'Actual __getitem__ AST, identity doc encodings; 20 repeated visits per row. No HF loading, real tokenizer, model forward, or historical epoch replay.','errors':errors,'elapsed_seconds':round(time.monotonic()-start,2)}
den=counts['new_negative_slots']
report['ratios']={k:counts[k]/den for k in ('replaced_negative_slots','repeat_docid_slots','repeat_text_slots','hard_labeled_slots','replacement_slots_with_rule_label')}
report['ratios']['changed_rows']=counts['rows_changed']/counts['new_rows']
report['ratios']['repeat_docid_rows']=counts['repeat_docid_rows']/counts['new_rows']
(OUT/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
(OUT/'loader_probes.json').write_text(json.dumps(loader_records,ensure_ascii=False,indent=2)+'\n')
with (OUT/'manual_samples.jsonl').open('w') as f:
    for label,records in samples.items():
        for record in sorted(records,key=lambda r:r['line']): f.write(json.dumps(record,ensure_ascii=False)+'\n')
md=['# 三类规则人工复核样本','', '每类随机抽取 30 个命中 query，再随机取一篇命中负例。种子 20261002；抽样范围为新训练文件实际保留的负例。规则标签仅为假设，人工判断栏留空。','']
for label,records in samples.items():
    md.extend([f'## {label}', ''])
    for i,r in enumerate(sorted(records,key=lambda r:r['line']),1):
        md.extend([f'### {i}. QID {r["query_id"]}（第 {r["line"]} 行）','',f'Query: {r["query"]}','',f'正例 docid={r["positive"]["docid"]}: {r["positive"].get("title") or ""} {r["positive"]["text"]}','',f'负例 docid={r["negative"]["docid"]}: {r["negative"].get("title") or ""} {r["negative"]["text"]}','',f'命中原因: {" | ".join(r["hit"]["reasons"])}；覆盖率={r["hit"]["coverage"]}；标签={",".join(r["hit"]["labels"])}；是否替换新增={r["negative_was_replacement"]}','',f'规则上下文: {json.dumps(r["context"],ensure_ascii=False)}','', '人工判断（真负例 / 假负例 / 不确定）: ______', ''])
(OUT/'manual_samples.md').write_text('\n'.join(md))
print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)
