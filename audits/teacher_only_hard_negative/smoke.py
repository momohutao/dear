"""CPU-only smoke: verified full sidecar, real tokenizer, Dataset and Collator."""
import gzip,json,logging,random,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2] / 'pointwise_reranker'))
from runtime_hard_negative import load_candidates
from arguments import DataArguments
from data import RerankerTrainPreProcessor,RerankerTrainDataset,RerankerTrainCollator
from transformers import AutoTokenizer
from datasets import Dataset
logging.basicConfig(level=logging.INFO)
source='/hy-tmp/DeAR/data/msmarco-passage/train.jsonl.gz'
candidates=load_candidates('/hy-tmp/DeAR/data/student_top1_teacher05_7371',[source],32,196)
tokenizer=AutoTokenizer.from_pretrained('/hy-tmp/DeAR/models/Llama-3.2-3B',local_files_only=True)
if tokenizer.pad_token_id is None: tokenizer.pad_token=tokenizer.eos_token
rows=[]
with gzip.open(source,'rt') as f:
 for line in f:
  rows.append(json.loads(line))
  if len(rows)==128: break
pre=RerankerTrainPreProcessor(tokenizer,tokenizer,32,196,hard_candidates=candidates)
tokenized=Dataset.from_list(rows).map(pre,remove_columns=list(rows[0]),load_from_cache_file=False)
args=DataArguments(dataset_name='json',hard_negative_candidates='/hy-tmp/DeAR/data/student_top1_teacher05_7371',hard_negative_probability=.1,train_n_passages=8,q_max_len=32,p_max_len=196)
ds=RerankerTrainDataset(args,tokenized,tokenizer,tokenizer)
random.seed(42)
injected=0
examples=[]
for i in range(1280):
 item=ds[i%128]; meta=item[2]
 injected+=meta['hard_injected']
 if meta['hard_injected']:
  record=candidates[str(meta['qid'])]
  assert str(meta['positive_docid']) in record[3]
  assert record[2] in [str(did) for did in meta['negative_docids']]
 assert len(meta['negative_docids'])==7
 assert len(set(meta['negative_docids']))==7
 if len(examples)<2: examples.append(item)
batch=RerankerTrainCollator(tokenizer,tokenizer)(examples)
assert len(batch['_hard_injected'])==2
assert batch['student']['input_ids'].shape[0]==16
result={'sample_rows':128,'sample_draws':1280,'actual_replacements':injected,'actual_replacement_rate':injected/1280,'full_sidecar_eligible_queries':sum(bool(r[3]) for r in candidates.values()),'full_sidecar_eligible_pairs':sum(len(r[3]) for r in candidates.values()),'student_batch_shape':list(batch['student']['input_ids'].shape),'note':'CPU teacher-only data-pipeline smoke; no model forward/backward or training launched.'}
Path(__file__).with_name('smoke_results.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
