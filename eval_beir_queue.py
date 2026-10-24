import concurrent.futures
import json
import os
from pathlib import Path
import queue
import re
import subprocess
import time

BASE = Path('/root/dear_mse/results/qk-mvp-all500-beir8')
BEST = BASE.parent / 'qk-mvp-best-beir8'
DATASETS = 'covid nfc touche dbpedia scifact signal news robust04'.split()
TASKS = queue.Queue()

def score(folder, dataset):
    path = folder / dataset / 'evaluation.log'
    if not path.exists():
        return None
    matches = re.findall(r'^ndcg_cut_10\s+all\s+([0-9.]+)\s*$', path.read_text(), re.M)
    return float(matches[-1]) if matches else None

def summarize(folder):
    scores = [score(folder, d) for d in DATASETS]
    if any(s is None for s in scores):
        return
    (folder / 'beir8_summary.tsv').write_text('dataset\tndcg@10\tndcg@10_x100\n' + ''.join(f'{d}\t{s:.4f}\t{s*100:.2f}\n' for d, s in zip(DATASETS, scores)))
    mean = sum(scores)/8
    (folder / 'beir8_mean.log').write_text(f'BEIR-8 mean nDCG@10: {mean:.4f}\nBEIR-8 mean nDCG@10 x100: {mean*100:.2f}\n')

def worker(gpu):
    failures = []
    while True:
        try:
            ckpt, folder, dataset = TASKS.get_nowait()
        except queue.Empty:
            return failures
        dest = folder / dataset
        dest.mkdir(parents=True, exist_ok=True)
        # An evaluation score is the completion evidence. Quarantine unverified raw output.
        raw = dest / f'{dataset}.txt'
        if raw.exists():
            raw.rename(dest / f'{dataset}.txt.unverified-{time.time_ns()}')
        env = dict(os.environ, CKPT_DIR=str(ckpt), RESULT_ROOT=str(folder), BEIR_DATASET=dataset, BEIR_GPU=str(gpu))
        print(f'START GPU={gpu} {folder.name}/{dataset}', flush=True)
        with (dest / 'queue.log').open('a') as log:
            result = subprocess.run(['bash', '/root/dear-sll-vv/full_beir.sh'], env=env, stdout=log, stderr=subprocess.STDOUT)
        ok = result.returncode == 0 and score(folder, dataset) is not None
        print(f'{"DONE" if ok else "FAILED"} GPU={gpu} {folder.name}/{dataset}', flush=True)
        if not ok:
            failures.append(f'{folder.name}/{dataset}')

if __name__ == '__main__':
    # Let the existing checkpoint finish before allocating its GPUs.
    pid = os.environ.get('WAIT_PID')
    while pid:
        try:
            state = Path(f'/proc/{pid}/stat').read_text().split()[2]
        except FileNotFoundError:
            break
        if state == 'Z':
            break
        time.sleep(5)
    folders = []
    for seed in (42, 43, 44):
        root = Path(f'/hy-tmp/DeAR/outputs/dear-qk-mvp-5pass-seed{seed}-earlystop-new')
        for ckpt in sorted(root.glob('adapter-step-*'), key=lambda p: int(p.name.split('-')[-1])):
            if int(ckpt.name.split('-')[-1]) % 500:
                continue
            folder = BASE / f'seed{seed}-{ckpt.name}'
            previous = BEST / folder.name
            if not folder.exists() and all(score(previous, d) is not None for d in DATASETS):
                folder.symlink_to(previous, target_is_directory=True)
            folder.mkdir(parents=True, exist_ok=True)
            folders.append(folder)
            for dataset in DATASETS:
                if score(folder, dataset) is None:
                    TASKS.put((ckpt, folder, dataset))
            summarize(folder)
    print(f'Pending dataset tasks: {TASKS.qsize()}', flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(worker, gpu) for gpu in range(4)]
        while not all(f.done() for f in futures):
            for folder in folders:
                summarize(folder)
            time.sleep(10)
        failures = [item for f in futures for item in f.result()]
    for folder in folders:
        summarize(folder)
    print('FINISHED failures=' + json.dumps(failures), flush=True)
    raise SystemExit(bool(failures))
