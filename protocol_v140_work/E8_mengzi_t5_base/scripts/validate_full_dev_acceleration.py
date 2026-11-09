"""Paired complete-Dev diagnostic, not a formal Test or resumed training run."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT.parent/'shared')]
import torch
from ed2 import engine
from ed2.data import Codec,benchmark_public,benchmark_gold
from ed2.io import read,sha,digest,write,write_rows,source_fingerprint
from ed2.runtime import require_runtime,evaluation_environment


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists() or any((ROOT/'outputs').glob('ed2a_*')):
        raise RuntimeError('New diagnostic output and empty formal outputs required')
    occupied=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],capture_output=True,text=True,timeout=15,check=True).stdout.strip()
    if occupied: raise RuntimeError('GPU occupied: '+occupied)
    args.output.mkdir(parents=True)
    cfg=read(ROOT/'configs/ed2a.json'); data=ROOT.parent/'shared/data/v1.5.0'
    # Explicitly inspect an archived, immutable diagnostic checkpoint. Do not
    # call train/resume or weaken the formal checkpoint identity guards.
    print('LOADING_FIXED_DIAGNOSTIC_CHECKPOINT',str(args.checkpoint),flush=True)
    ckpt=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    require_runtime(ckpt['identity'])
    model=engine.model_from_checkpoint(ckpt,torch.device('cuda:0')); model.eval()
    checkpoint_info={'kind':ckpt['kind'],'epoch':ckpt['epoch'],'original_identity':ckpt['identity']}
    del ckpt
    codec=Codec(data,cfg); public=benchmark_public(data,'dev_full')
    assert len(public)==15024
    encoded=[codec.encode(q) for q in public]
    gold=lambda:benchmark_gold(data,'dev_full',public)
    spec=importlib.util.spec_from_file_location('ed2._frozen_reference_decoder',args.reference)
    reference=importlib.util.module_from_spec(spec); spec.loader.exec_module(reference)
    candidate=engine.predict_encoded_batch
    report={'kind':'PAIRED_FULL_DEV_DIAGNOSTIC_NOT_FORMAL_TEST','code_sha256':source_fingerprint(ROOT),
            'config_sha256':sha(ROOT/'configs/ed2a.json'),'data_sha256':sha(data/'DATA_SHA256SUMS.json'),
            'checkpoint_sha256':sha(args.checkpoint),'checkpoint':checkpoint_info,
            'reference_decoder_sha256':sha(args.reference),'public_sha256':sha(data/'benchmark/dev_full_public.jsonl'),
            'evaluation_runtime':evaluation_environment('cuda:0',cfg),'test_answers_used':False,
            'training_resumed':False,'cases':[]}
    saved=None
    for name,fn,backend in [('reference',reference.predict_encoded_batch,'eager'),
                             ('accelerated',candidate,cfg['evaluation_backend'])]:
        raw_hashes=[]
        def audited(*a,**kw):
            predictions=fn(*a,**kw)
            raw_hashes.extend(digest(p) for p in predictions)
            return predictions
        engine.predict_encoded_batch=audited
        config=dict(cfg,evaluation_backend=backend)
        stats={}; torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        print('FULL_DEV_START',name,flush=True)
        torch.cuda.synchronize(); started=time.perf_counter()
        try:
            metrics,values=engine.evaluate(model,codec,public,gold,config,torch.device('cuda:0'),encoded,
                                           execution_report=stats)
            torch.cuda.synchronize(); seconds=time.perf_counter()-started
        finally:
            engine.predict_encoded_batch=candidate
        entry={'name':name,'seconds_including_cold_capture':seconds,'questions':len(values),
               'raw_predictions_digest':digest(raw_hashes),'scored_predictions_digest':digest(values),
               'metrics':metrics,'execution':stats,'peak_mib':torch.cuda.max_memory_allocated()/2**20}
        write_rows(args.output/(name+'_dev_predictions.jsonl'),values)
        if saved is None:
            saved=(metrics,values,raw_hashes)
        else:
            entry.update(all_metrics_identical=metrics==saved[0],
                         all_scored_predictions_identical=values==saved[1],
                         all_raw_predictions_identical=raw_hashes==saved[2],
                         speedup=report['cases'][0]['seconds_including_cold_capture']/seconds)
        report['cases'].append(entry)
        print('FULL_DEV_RESULT',json.dumps({k:v for k,v in entry.items() if k!='metrics'}),flush=True)
        write(args.output/'report.json',report)
    last=report['cases'][-1]
    assert last['all_metrics_identical'] and last['all_scored_predictions_identical'] and last['all_raw_predictions_identical']
    assert last['execution']['captures']>0 and last['execution']['replays']>0
    report['status']='PASS'
    write(args.output/'report.json',report)
    print('FULL_DEV_EQUIVALENCE_PASS',flush=True)


if __name__=='__main__':main()
