"""CPU sample verification through real preprocessing, dataset and collator."""
from collections import Counter
import json
import os
from pathlib import Path
import random
import sys
from types import SimpleNamespace

os.environ['HF_HUB_OFFLINE']='1'
OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
sys.path.insert(0,str(ROOT/'pointwise_reranker'))
from transformers import AutoTokenizer
from datasets import Dataset
from data import RerankerTrainPreProcessor, RerankerTrainDataset, RerankerTrainCollator

cache=json.loads((OUT/'token_cache_summary.json').read_text())['cache_path']
ds=Dataset.from_file(cache)
student=AutoTokenizer.from_pretrained('/hy-tmp/DeAR/models/Llama-3.2-3B',local_files_only=True)
teacher=AutoTokenizer.from_pretrained('/hy-tmp/DeAR/models/llama2-13b-rankllama-teacher',local_files_only=True)
for tok in (student,teacher):
    if tok.pad_token is None:tok.pad_token=tok.eos_token
    tok.padding_side='right'
processor=RerankerTrainPreProcessor(student,teacher,32,196)
args=SimpleNamespace(q_max_len=32,p_max_len=196,train_n_passages=8,positive_passage_no_shuffle=False,negative_passage_no_shuffle=False)
collator=RerankerTrainCollator(student,teacher,32,196)
c=Counter();errors=[];probes=[];unique_lines=set()
for text in (OUT/'manual_samples.jsonl').open():
    r=json.loads(text)
    if r['line'] in unique_lines:continue
    unique_lines.add(r['line'])
    raw={'query_id':r['query_id'],'query':r['query'],'positive_passages':[r['positive']],'negative_passages':r['all_selected_negatives']}
    expected=processor(raw);actual=ds[r['line']-1]
    c['checked_rows']+=1
    for field,value in expected.items():
        if actual[field]!=value:
            c['token_cache_field_mismatch']+=1;errors.append({'line':r['line'],'field':field})
    train=RerankerTrainDataset(args,[actual],student,teacher)
    random.seed(42)
    visits=[train[0] for _ in range(3)]
    c['real_loader_visits']+=3
    batch=collator(visits[:2])
    for who,tok in (('student',student),('teacher',teacher)):
        for j,visit in enumerate(visits[:2]):
            features=visit[0 if who=='student' else 1]
            for k,(model_input,regions) in enumerate(features):
                c['checked_collated_sequences']+=1
                if batch[who]['input_ids'][j*8+k,:len(model_input['input_ids'])].tolist()!=model_input['input_ids']:
                    c['collated_token_order_mismatch']+=1
    for s,t,metadata in visits:
        ids=[metadata['positive_docid']]+metadata['negative_docids']
        passage_by_id={str(p['docid']):p for p in [r['positive']]+r['all_selected_negatives']}
        for who,features in (('student',s),('teacher',t)):
            for k,did in enumerate(ids):
                key=('positives_' if k==0 else 'negatives_')+who
                docids=actual['repro_positive_docids' if k==0 else 'repro_negative_docids']
                idx=docids.index(did)
                encoding=actual[key][idx]
                pair=train.create_one_example(actual['query_'+who],encoding,student if who=='student' else teacher)
                if features[k]!=pair:c['loader_docid_token_alignment_mismatch']+=1
        if metadata['qid']!=r['query_id']:c['loader_qid_mismatch']+=1
    probes.append({'line':r['line'],'qid':r['query_id'],'three_visits':[v[2] for v in visits]})
report={'counts':dict(c),'errors':errors,'cache_path':cache,'q_max_len':32,'p_max_len':196,'student_tokenizer':student.name_or_path,'teacher_tokenizer':teacher.name_or_path,'scope':'Real local tokenizers, current preprocessor/dataset/collator on reservoir samples, CPU only; no model/teacher forward or training.','probes':probes}
(OUT/'real_token_probe.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'counts':dict(c),'errors':errors},ensure_ascii=False,indent=2))
