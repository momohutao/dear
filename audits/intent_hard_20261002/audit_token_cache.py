"""Check existing HF cache metadata against the actual new JSONL, read-only."""
from collections import Counter
import gzip
import itertools
import json
from pathlib import Path
import time
import pyarrow as pa

OUT=Path(__file__).resolve().parent
inventory=json.loads((OUT/'hf_cache_inventory.json').read_text())
entry=next(e for e in inventory if any('train_intent_hard' in p for p in e['download_sources']))
directory=Path(entry['info']).parent
path=directory/entry['tokenization_caches'][0]['file']
def metadata_rows():
    with pa.memory_map(str(path),'r') as f:
        for batch in pa.ipc.open_stream(f):
            yield from batch.select(['repro_qid','repro_positive_docids','repro_negative_docids']).to_pylist()
c=Counter();errors=[];start=time.monotonic()
with gzip.open(next(iter(entry['download_sources'])),'rt') as f:
    for line,(raw,cache) in enumerate(itertools.zip_longest(f,metadata_rows()),1):
        if raw is None or cache is None:
            c['unpaired_rows']+=1
            continue
        r=json.loads(raw);c['checked_rows']+=1
        for k,value in [('repro_qid',str(r['query_id'])),('repro_positive_docids',[str(p['docid']) for p in r['positive_passages']]),('repro_negative_docids',[str(p['docid']) for p in r['negative_passages']])]:
            if cache[k]!=value:
                c[k+'_mismatch']+=1
                if len(errors)<30:errors.append({'line':line,'field':k})
        if line%50000==0:print(f'rows={line} elapsed={time.monotonic()-start:.1f}s',flush=True)
report={'cache_path':str(path),'counts':dict(c),'errors':errors,'elapsed_seconds':round(time.monotonic()-start,2),'scope':'Full cache docid metadata order; actual token content checked separately on random samples.'}
(OUT/'token_cache_summary.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
