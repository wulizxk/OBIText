"""Read-only model/data diagnostics; writes only a new verification directory."""
import argparse
from contextlib import contextmanager
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT.parent/'shared')]
import torch
from ed2 import decoding
from ed2.data import Codec, SpanSampler, pad_inputs
from ed2.engine import seed_all, model_from_checkpoint
from ed2.io import read, write, sha
from ed2.model import autocast, decoder_hidden, project_hidden
from ed2.runtime import require_runtime


class GraphProbe:
    def __init__(self, cfg):
        self.cfg = cfg
        self.cache, self.seen = {}, {}
        self.stats = {'captures': 0, 'replays': 0, 'fallback': 0, 'capture_seconds': 0.0}

    def call(self, kind, fn, args):
        key = (kind, tuple((tuple(x.shape), x.dtype, tuple(x.stride())) for x in args))
        self.seen[key] = self.seen.get(key, 0) + 1
        if key not in self.cache:
            if self.seen[key] < 3 or len(self.cache) >= 24:
                return fn(*args)
            started = time.perf_counter()
            static = [x.detach().clone() for x in args]
            dtype = torch.bfloat16 if self.cfg['precision'] == 'bf16' else torch.float32
            def invoke():
                with torch.no_grad(), torch.autocast('cuda', dtype=dtype,
                                                     enabled=dtype != torch.float32, cache_enabled=False):
                    return fn(*static)
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(2):
                    warm = invoke()
            torch.cuda.current_stream().wait_stream(stream)
            del warm
            reference = fn(*args)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                output = invoke()
            graph.replay()
            if not torch.equal(reference, output):
                raise AssertionError('Graph output not bitwise equal: ' + repr(key))
            self.cache[key] = graph, static, output
            self.stats['captures'] += 1
            self.stats['capture_seconds'] += time.perf_counter() - started
        graph, static, output = self.cache[key]
        for target, source in zip(static, args):
            target.copy_(source)
        graph.replay()
        self.stats['replays'] += 1
        return output.clone()


@contextmanager
def graph_hooks(model, cfg):
    graph = GraphProbe(cfg)
    original_encoder = model.encoder.forward
    original_decoder = decoding.decoder_hidden
    def encode(input_ids, attention_mask, **kwargs):
        result = graph.call('encoder', lambda a, b: original_encoder(
            input_ids=a, attention_mask=b, use_cache=False, return_dict=True).last_hidden_state,
            (input_ids, attention_mask))
        return SimpleNamespace(last_hidden_state=result)
    def decode(model_arg, ids, mask, states, encoder_mask):
        return graph.call('decoder', lambda a, b, c, d: original_decoder(model_arg,a,b,c,d),
                          (ids,mask,states,encoder_mask))
    model.encoder.forward = encode
    decoding.decoder_hidden = decode
    try:
        yield graph
    finally:
        model.encoder.forward = original_encoder
        decoding.decoder_hidden = original_decoder
        torch.cuda.synchronize()
        graph.cache.clear()


def questions_from_train(sampler, per_length=128):
    public = []
    for length in range(1,6):
        for item in sampler.pools[length][:per_length]:
            row=sampler.rows[item['source_id']]; start=item['starts'][0]
            public.append({'source_id':item['source_id'],'gap_length':length,
                           'prefix_units':row['model_tokens'][:start],
                           'suffix_units':row['model_tokens'][start+length:]})
    return public


def predict(model, codec, public, cfg):
    results=[]
    for length in range(1,6):
        group=sorted([q for q in public if q['gap_length']==length],
                     key=lambda q:len(codec.encode(q)['input_ids']))
        for start in range(0,len(group),cfg['eval_question_batch_size']):
            results.extend(decoding.predict_batch(model,codec,group[start:start+cfg['eval_question_batch_size']],cfg,torch.device('cuda:0')))
    return results


def compare(a,b):
    assert len(a)==len(b)
    return {'questions':len(a),'same_top1':sum(x['paths'][0]==y['paths'][0] for x,y in zip(a,b)),
            'same_topk':sum(x['paths']==y['paths'] for x,y in zip(a,b)),
            'identical_complete_predictions':a==b}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists() or any((ROOT/'outputs').glob('ed2a_*')):
        raise RuntimeError('Use a new output; no formal E8 run may exist')
    occupied=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],capture_output=True,text=True,timeout=15,check=True).stdout.strip()
    if occupied: raise RuntimeError('GPU occupied: '+occupied)
    args.output.mkdir(parents=True)
    cfg=read(ROOT/'configs/ed2a.json'); data=ROOT.parent/'shared/data/v1.5.0'
    checkpoint=ROOT/'verification/gpu_smoke_v150/smoke.pt'
    print('LOAD_DIAGNOSTIC_SMOKE_CHECKPOINT',flush=True)
    ckpt=torch.load(checkpoint,map_location='cpu',weights_only=True)
    require_runtime(ckpt['identity']); seed_all(42)
    model=model_from_checkpoint(ckpt,torch.device('cuda:0')); model.eval()
    codec=Codec(data,cfg); sampler=SpanSampler(data,42)
    public=questions_from_train(sampler)
    report={'kind':'TRAIN_ONLY_VALIDATION_DIAGNOSTIC','checkpoint_sha256':sha(checkpoint),
            'checkpoint_kind':ckpt['kind'],'test_answers_used':False,'formal_training_started':False,'cases':[]}
    del ckpt
    reference={}
    for precision,qbatch,fbatch in [('bf16',32,64),('bf16',128,256),('fp32',32,64),('fp32',128,256)]:
        config=dict(cfg,precision=precision,eval_question_batch_size=qbatch,eval_forward_batch_size=fbatch)
        torch.cuda.synchronize(); started=time.perf_counter()
        predictions=predict(model,codec,public,config)
        torch.cuda.synchronize(); elapsed=time.perf_counter()-started
        if precision not in reference: reference[precision]=predictions
        entry={'precision':precision,'question_batch':qbatch,'forward_batch':fbatch,'seconds':elapsed,
               **compare(reference[precision],predictions)}
        report['cases'].append(entry); print('BATCH_CASE',json.dumps(entry),flush=True); write(args.output/'report.json',report)
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    with graph_hooks(model,cfg) as backend:
        for repeat in range(2):
            torch.cuda.synchronize(); started=time.perf_counter()
            predictions=predict(model,codec,public,cfg)
            torch.cuda.synchronize(); elapsed=time.perf_counter()-started
            entry={'backend':'cuda_graph_prototype','repeat':repeat,'seconds':elapsed,
                   'peak_mib':torch.cuda.max_memory_allocated()/2**20,
                   **compare(reference['bf16'],predictions),'stats':dict(backend.stats)}
            report['cases'].append(entry); print('GRAPH_CASE',json.dumps(entry),flush=True); write(args.output/'report.json',report)
    report['status']='COMPLETE'; write(args.output/'report.json',report)
    print('DIAGNOSTIC_COMPLETE',flush=True)


if __name__=='__main__': main()
