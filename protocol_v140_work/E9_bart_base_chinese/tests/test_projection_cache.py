"""Exact CPU cached/uncached comparisons; no formal weights, GPU or Test runs."""
import copy
from types import SimpleNamespace

import pytest
import torch

from ed3 import engine, diagnostics
from ed3.model import project_hidden, scaler_for, optimizer_for
from ed3.projection_cache import EvaluationProjectionCache
from test_training_loop import MiniCodec
from test_e9_review_fixes import launch_extensions


def toy_codec():
    return SimpleNamespace(target_ids=list(range(5,17)), candidate_ids=list(range(5,18)),
                           units=[f'c{i}' for i in range(12)], end_gap_id=17, decoder_start_id=102,
                           tokenizer=SimpleNamespace(pad_token_id=0))


def test_cached_projection_exact_fp32_rows_bias_and_hits(tiny):
    tiny.eval()
    with torch.no_grad():
        tiny.final_logits_bias.copy_(torch.linspace(-.4,.4,128).unsqueeze(0))
        hidden=torch.randn(32,16)
        ids=torch.tensor([7,2,90,19])
        expected=project_hidden(tiny,hidden,ids)
        with EvaluationProjectionCache(tiny) as cache:
            for _ in range(5):
                report={}
                actual=project_hidden(tiny,hidden,ids,report,cache)
                assert torch.equal(actual,expected)
                assert report=={'linear':'torch.float32','bias':'torch.float32','result':'torch.float32'}
            assert cache.misses==1 and cache.hits==4
            assert all(not value.requires_grad for row in cache.entries.values() for value in row[2:])
        assert cache.closed and not cache.entries and cache.model is None


@pytest.mark.parametrize('change',['weights','bias','indices','replace_weight','wrong_model','training','grad_enabled','closed'])
def test_projection_cache_rejects_stale_or_training_use(tiny,change):
    tiny.eval();ids=torch.tensor([5,6,7]);hidden=torch.randn(2,16)
    with EvaluationProjectionCache(tiny) as cache,torch.no_grad():
        project_hidden(tiny,hidden,ids,projection_cache=cache)
        model=tiny
        if change=='weights':tiny.lm_head.weight.add_(.01)
        if change=='bias':tiny.final_logits_bias.add_(.01)
        if change=='indices':ids[0]=8
        if change=='replace_weight':tiny.lm_head.weight=torch.nn.Parameter(tiny.lm_head.weight.clone())
        if change=='wrong_model':model=copy.deepcopy(tiny)
        if change=='training':tiny.train()
        if change=='closed':cache.__exit__(None,None,None)
        with torch.set_grad_enabled(change=='grad_enabled'),pytest.raises(ValueError):
            project_hidden(model,hidden,ids,projection_cache=cache)


def test_cache_is_cleared_on_exception(tiny):
    tiny.eval()
    cache=EvaluationProjectionCache(tiny)
    with pytest.raises(RuntimeError),cache,torch.no_grad():
        project_hidden(tiny,torch.ones(1,16),torch.tensor([5,6]),projection_cache=cache)
        raise RuntimeError('diagnostic failure')
    assert cache.closed and not cache.entries and cache.weight is None and cache.bias is None
    with pytest.raises(ValueError):cache.__enter__()


def test_full_generation_cached_vs_uncached_candidates_scores_ranks_and_metrics(tiny,cfg):
    codec=toy_codec()
    config=dict(cfg,eval_question_batch_size=3,eval_forward_batch_size=6)
    public=[];encoded=[];answers={}
    for length in range(1,6):
        for i in range(7):
            q={'example_id':f'{length}-{i}','source_id':f'TRAIN-{i}','group_id':'TRAIN',
               'gap_length':length,'prefix_units':['c0'] if i%2 else [],
               'suffix_units':[] if i%2 else ['c1']}
            public.append(q);answers[q['example_id']]=['c0']*length
            encoded.append({'input_ids':[2]*(128 if i==6 else i+2),'gap_length':length})
    reference=diagnostics.generate_public(tiny,codec,public,encoded,config,'cpu')
    raw=[];execution={}
    metrics,_=engine.evaluate(tiny,codec,public,lambda:answers,config,torch.device('cpu'),encoded,
                              execution_report=execution,raw_output=raw)
    comparison=diagnostics.compare_predictions(reference,raw,public,answers,codec.units)
    diagnostics.require_identical(comparison)
    assert comparison['candidate_metrics']==metrics
    report=execution['output_projection_cache']
    assert report['misses']==2 and report['hits']>2 and report['closed']
    assert report['peak_bytes']==(12+13)*(16+1)*4


def test_new_evaluation_rebuilds_rows_after_model_update(tiny):
    tiny.eval();ids=torch.tensor([5,6]);hidden=torch.ones(1,16)
    with torch.no_grad():
        with EvaluationProjectionCache(tiny) as first:
            before=project_hidden(tiny,hidden,ids,projection_cache=first)
        tiny.lm_head.weight.add_(1)
        with EvaluationProjectionCache(tiny) as second:
            after=project_hidden(tiny,hidden,ids,projection_cache=second)
            assert torch.equal(after,project_hidden(tiny,hidden,ids))
        assert not torch.equal(before,after)


def test_unscaled_update_does_not_snapshot_rng_each_step(tiny,cfg,monkeypatch):
    optimizer=optimizer_for(tiny,cfg);scaler=scaler_for('cpu',cfg)
    def unwanted(*args,**kwargs):pytest.fail('Unscaled update cannot retry, so no per-step RNG snapshot is needed')
    monkeypatch.setattr(torch,'get_rng_state',unwanted)
    monkeypatch.setattr(torch.cuda,'get_rng_state',unwanted)
    loss,norm=engine.update(tiny,MiniCodec(None,cfg),[{'gold':5},{'gold':6}],cfg,
                            torch.device('cpu'),optimizer,scaler)
    assert loss>0 and norm>0


def test_scaled_capable_update_preserves_rng_snapshot(tiny,cfg,monkeypatch):
    optimizer=optimizer_for(tiny,cfg);scaler=scaler_for('cpu',cfg)
    # Exercise the CPU control-flow branch only; this is not CUDA FP16 arithmetic.
    monkeypatch.setattr(scaler,'is_enabled',lambda:True)
    real=torch.get_rng_state;calls=[]
    monkeypatch.setattr(torch,'get_rng_state',lambda:(calls.append(True) or real()))
    engine.update(tiny,MiniCodec(None,cfg),[{'gold':5}],cfg,torch.device('cpu'),optimizer,scaler)
    assert calls==[True]


@pytest.mark.parametrize('change',['missing_reference','wrong_reference','not_closed','no_hits','too_many_misses','wrong_bytes'])
def test_launch_requires_real_projection_cache_comparison(cfg,change):
    report=launch_extensions(cfg)
    cache=report['evaluation_execution_before_reload']['output_projection_cache']
    if change=='missing_reference':report.pop('projection_cache_reference')
    if change=='wrong_reference':report['projection_cache_reference']='cached_vs_cached_only'
    if change=='not_closed':cache['closed']=False
    if change=='no_hits':cache['hits']=0
    if change=='too_many_misses':cache['misses']=100
    if change=='wrong_bytes':cache['peak_bytes']=1
    with pytest.raises(ValueError):diagnostics.validate_launch_diagnostics(report,cfg)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='Real CUDA BF16 cached/uncached projection requires an idle authorized GPU')
def test_cuda_bf16_projection_cache_exact(tiny,cfg):
    from ed3.model import autocast
    from ed3.runtime import configure_execution,require_cuda_precision
    configure_execution(42);require_cuda_precision('cuda:0',cfg)
    tiny.to('cuda:0').eval();ids=torch.tensor(list(range(5,17)),device='cuda:0')
    hidden=torch.randn(64,16,device='cuda:0')
    with torch.no_grad(),autocast('cuda:0',cfg),EvaluationProjectionCache(tiny) as cache:
        assert torch.equal(project_hidden(tiny,hidden,ids),project_hidden(tiny,hidden,ids,projection_cache=cache))
