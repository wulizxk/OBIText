"""Regressions from the independent E9 review; no real weights or formal jobs."""
import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from transformers import BartForConditionalGeneration

from ed3 import diagnostics, engine, formal_controller
from ed3.model import (FORMAL_MODEL_CONTRACT, validate_formal_model_config,
                       validate_formal_model, validate_checkpoint_tensors, resize_embeddings)
from ed3.io import read, rows, write, write_rows, sha
from ed3.decoding import predict_encoded_batch
from test_bart import native_config


def formal_config():
    config = native_config()
    config.vocab_size = 54295
    config.use_cache = False
    return config


def test_formal_model_contract_positive_without_allocating_weights():
    config = formal_config()
    validate_formal_model_config(config)
    with torch.device('meta'):
        model = BartForConditionalGeneration(config)
    validate_formal_model(model)


@pytest.mark.parametrize('field', sorted(FORMAL_MODEL_CONTRACT))
@pytest.mark.parametrize('change', ['value', 'missing'])
def test_checkpoint_rejects_semantic_drift_before_construction(field, change, monkeypatch):
    config = formal_config().to_dict()
    old = config[field]
    if change == 'missing':
        config.pop(field)
    else:
        config[field] = (not old if type(old) is bool else old+1 if type(old) in (int, float) else old+'_WRONG')
    import transformers
    monkeypatch.setattr(transformers, 'BartForConditionalGeneration', lambda *_: pytest.fail('Constructed wrong model'))
    with pytest.raises(ValueError):
        engine.model_from_checkpoint({'model_config': config, 'model': {}}, 'cpu')


def test_tiny_checkpoint_cannot_enter_formal_loader(tiny):
    with pytest.raises(ValueError, match='model_config'):
        engine.model_from_checkpoint({'model_config': tiny.config.to_dict(), 'model': tiny.state_dict()}, 'cpu')


@pytest.mark.parametrize('field,value', [('d_model', 768.), ('use_cache', 0), ('dropout', True),
                                       ('_attn_implementation', 'sdpa')])
def test_contract_rejects_type_coercion_and_backend(field, value):
    with pytest.raises(ValueError):
        validate_formal_model_config(dict(formal_config().to_dict(), **{field: value}))


def expanded_checkpoint():
    shared = torch.zeros(1).expand(54295, 768)
    return {'model.shared.weight': shared, 'model.encoder.embed_tokens.weight': shared,
            'model.decoder.embed_tokens.weight': shared, 'lm_head.weight': shared,
            'final_logits_bias': torch.zeros(1).expand(1, 54295)}


def test_checkpoint_aliases_and_dtype_are_not_silently_overwritten():
    state = expanded_checkpoint()
    validate_checkpoint_tensors(state)
    for bad in [dict(state, **{'lm_head.weight': torch.ones(1).expand(54295,768)}),
                dict(state, **{'final_logits_bias': torch.zeros(1,54295,dtype=torch.float16)}),
                dict(state, **{'model.shared.weight': torch.zeros(1)})]:
        with pytest.raises(ValueError):
            validate_checkpoint_tensors(bad)


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.float64,
                                  torch.int64, torch.bool, torch.complex64, torch.complex128])
def test_checkpoint_rejects_every_non_fp32_state_dtype(dtype):
    state = expanded_checkpoint()
    state['final_logits_bias'] = torch.zeros(1, dtype=dtype).expand(1, 54295)
    with pytest.raises(ValueError, match='FP32'):
        validate_checkpoint_tensors(state)


def comparison_fixture():
    units = [f'c{i}' for i in range(12)]
    q = {'example_id': 'Q', 'source_id': 'TRAIN', 'group_id': 'GROUP', 'gap_length': 1}
    prediction = {'paths': [[i] for i in range(10)], 'scores': [-float(i+1) for i in range(10)],
                  'single_ranking': list(range(12)), 'end_gap_id':17,
                  'score_kind':'seq2seq_direct_character_log_probability'}
    return [q], {'Q':['c0']}, units, [prediction]


@pytest.mark.parametrize('change', ['score_only', 'tail_rank_only', 'candidate_order', 'shorter'])
def test_performance_comparison_detects_changes_even_with_same_metrics(change):
    q, answers, units, original = comparison_fixture()
    altered = copy.deepcopy(original)
    if change == 'score_only':altered[0]['scores'][-1] += 1e-8
    if change == 'tail_rank_only':altered[0]['single_ranking'][-2:] = [11,10]
    if change == 'candidate_order':altered[0]['paths'][0:2] = [[1],[0]]
    if change == 'shorter':altered=[]
    report = diagnostics.compare_predictions(original, altered, q, answers, units)
    assert report['status'] in ('CHANGED','FAILED')
    with pytest.raises(ValueError):diagnostics.require_identical(report)


def test_performance_comparison_accepts_exact_outputs_and_all_ranks():
    q, answers, units, predictions = comparison_fixture()
    report = diagnostics.compare_predictions(predictions, copy.deepcopy(predictions), q, answers, units)
    diagnostics.require_identical(report)
    assert report['max_candidate_score_abs_error']==0


def launch_extensions(cfg):
    native = {'status':'PASS','oracle':'BartForConditionalGeneration.forward','device':'cuda:0',
              'questions':5,'gap_lengths':[1,2,3,4,5],'finite':True,'test_answers_used':False,
              'atol':.05,'rtol':.01,'max_abs_error':0.,'max_error_to_allowed_ratio':0.,
              'native_linear_dtype':'torch.bfloat16','native_logits_dtype':'torch.float32',
              'restricted_dtypes':{'linear':'torch.bfloat16','bias':'torch.float32','result':'torch.float32'},
              'parameter_dtype':'torch.float32'}
    shape = {'max_encoder_shape':[32,128],'max_decoder_shape':[64,8],'gap_lengths':[1,2,3,4,5],
             'boundary_sides':['initial','final'],'test_answers_used':False,'maximum_gap_length':5,'question_count':194}
    equality = {'status':'IDENTICAL','questions':194,'different_question_counts_by_field':{},
                'max_candidate_score_abs_error':0.,'score_atol':0.,'score_rtol':0.}
    return {'diagnostic_evidence_schema':2,'native_forward_reference':native,'generation_shape_evidence':shape,
            'projection_cache_reference':'uncached_public_reference_vs_formal_cached_evaluation',
            **{key: {'output_projection_cache': {'enabled': True, 'closed': True, 'misses': 2,
                                                'hits': 10, 'peak_bytes': (4116+4117)*(768+1)*4}}
               for key in ('evaluation_execution_before_reload', 'evaluation_execution_after_reload')},
            'full_prediction_eager_comparison':copy.deepcopy(equality),
            'full_prediction_checkpoint_comparison':copy.deepcopy(equality)}


@pytest.mark.parametrize('change', ['missing_native','cpu','fp32_linear','native_nan','native_negative','native_ratio',
                                   'native_tolerance','native_wrong_lengths','native_wrong_questions',
                                   'q3','f10','input64','missing_scores','changed','missing_boundary','missing_shapes'])
def test_launch_gate_rejects_incomplete_new_evidence(cfg, change):
    r=launch_extensions(cfg)
    if change=='missing_native':r.pop('native_forward_reference')
    if change=='cpu':r['native_forward_reference']['device']='cpu'
    if change=='fp32_linear':r['native_forward_reference']['restricted_dtypes']['linear']='torch.float32'
    if change=='native_nan':r['native_forward_reference']['max_abs_error']=float('nan')
    if change=='native_negative':r['native_forward_reference']['max_abs_error']=-1
    if change=='native_ratio':r['native_forward_reference']['max_error_to_allowed_ratio']=1.1
    if change=='native_tolerance':r['native_forward_reference']['atol']=100
    if change=='native_wrong_lengths':r['native_forward_reference']['gap_lengths']=[1]
    if change=='native_wrong_questions':r['native_forward_reference']['questions']=1
    if change=='q3':r['generation_shape_evidence']['max_encoder_shape'][0]=3
    if change=='f10':r['generation_shape_evidence']['max_decoder_shape'][0]=10
    if change=='input64':r['generation_shape_evidence']['max_encoder_shape'][1]=64
    if change=='missing_shapes':r.pop('generation_shape_evidence')
    if change=='missing_boundary':r['generation_shape_evidence']['boundary_sides']=[]
    if change=='missing_scores':r['full_prediction_eager_comparison'].pop('score_atol')
    if change=='changed':r['full_prediction_checkpoint_comparison']['status']='CHANGED'
    with pytest.raises(ValueError):diagnostics.validate_launch_diagnostics(r,cfg)


def test_launch_diagnostics_positive_schema(cfg):
    diagnostics.validate_launch_diagnostics(launch_extensions(cfg),cfg)


def test_actual_fixture_covers_real_batch_stress_and_both_boundaries(data_root, codec, cfg):
    from ed3.data import SpanSampler
    public, answers = diagnostics.train_fixtures(SpanSampler(data_root,42),codec,cfg)
    assert len(public)==len(answers)==194
    assert all('gold_units' not in q for q in public)
    assert {q.get('boundary_side') for q in public} >= {'initial','final'}
    stress=[q for q in public if q['source_id']=='TRAIN_DERIVED_SYNTHETIC']
    assert len(stress)==32 and all(len(codec.encode(q)['input_ids'])==128 and q['gap_length']==5 for q in stress)


def test_generation_profile_observes_real_q32_f64_shapes_on_tiny_cpu(tiny,cfg):
    codec=SimpleNamespace(target_ids=list(range(5,17)),candidate_ids=list(range(5,18)),end_gap_id=17,
                          decoder_start_id=102,tokenizer=SimpleNamespace(pad_token_id=0))
    encoded=[{'input_ids':[2]*128,'gap_length':5} for _ in range(32)]
    profile=diagnostics.GenerationProfile('cpu')
    with profile.measure('total'):
        result=predict_encoded_batch(tiny,codec,encoded,cfg,torch.device('cpu'),diagnostics=profile)
    assert len(result)==32
    assert profile.report()['max_encoder_shape']==[32,128]
    assert profile.report()['max_decoder_shape']==[64,8]
    assert profile.report()['calls']['encoder']==1


def native_questions(codec):
    return [{'gap_length':L,'prefix_units':['王'],'suffix_units':['吉'],'gold_units':['王']*L} for L in range(1,6)]


def test_independent_native_forward_reference_cpu(tiny,codec,cfg):
    resize_embeddings(tiny,codec.vocab_size)
    report=diagnostics.native_forward_reference(tiny,codec,native_questions(codec),cfg,'cpu')
    assert report['finite'] and report['native_linear_dtype']=='torch.float32'
    assert report['max_error_to_allowed_ratio']<=1


def test_native_reference_detects_bias_error(tiny,codec,cfg,monkeypatch):
    resize_embeddings(tiny,codec.vocab_size)
    real=diagnostics.restricted_logits
    monkeypatch.setattr(diagnostics,'restricted_logits',lambda *a,**kw: real(*a,**kw)+10)
    with pytest.raises(AssertionError):
        diagnostics.native_forward_reference(tiny,codec,native_questions(codec),cfg,'cpu')


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA native BART BF16 reference and batch shape regression')
def test_native_forward_reference_cuda(tiny,codec,cfg):
    resize_embeddings(tiny,codec.vocab_size)
    tiny.to('cuda:0')
    report=diagnostics.native_forward_reference(tiny,codec,native_questions(codec),cfg,'cuda:0')
    assert report['restricted_dtypes']['linear']=='torch.bfloat16'


def test_gpu_environment_query_is_short_lived_subprocess(cfg,monkeypatch,tmp_path):
    seen=[]
    def run(cmd,**kw):
        seen.append((cmd,kw));return SimpleNamespace(stdout='{"hardware":{"type":"cuda"}}')
    monkeypatch.setattr(formal_controller.subprocess,'run',run)
    monkeypatch.setattr(formal_controller,'ROOT',tmp_path)
    assert formal_controller.current_gpu_environment(cfg)=={'hardware':{'type':'cuda'}}
    assert seen[0][0][1:3]==['-B','-c'] and seen[0][1]['timeout']==45
    assert seen[0][1]['env']['CUDA_VISIBLE_DEVICES']=='0'


def load_probe():
    path=Path(__file__).resolve().parents[1]/'scripts/performance_probe.py'
    spec=importlib.util.spec_from_file_location('e9_probe',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def test_performance_probe_requires_explicit_gpu_and_bounded_arguments(tmp_path,monkeypatch):
    probe=load_probe()
    monkeypatch.setattr(probe.subprocess,'Popen',lambda *_:pytest.fail('No job may be launched'))
    for args in [[],['--allow-gpu','--max-seconds','3600'],['--allow-gpu','--repeats','1']]:
        with pytest.raises(ValueError):probe.main(['--output',str(tmp_path/'new'),*args])


def test_scheduler_boundary_contract_is_explicit(cfg):
    optimizer=torch.optim.SGD([torch.nn.Parameter(torch.tensor(0.))],lr=cfg['learning_rate'])
    f=engine.scheduler_for(optimizer,cfg).lr_lambdas[0]
    steps=(cfg['samples_per_epoch']+cfg['effective_batch_size']-1)//cfg['effective_batch_size']
    assert f(0)==1/steps and f(steps-1)==f(steps)==1 and f(steps+1)<1 and f(steps*cfg['max_epochs'])==0


def test_end_column_optimization_matches_frozen_reference_and_reduces_transfer(tiny,cfg,monkeypatch):
    from legacy_decode_reference import old_predict_encoded_batch
    codec=SimpleNamespace(target_ids=list(range(5,17)),candidate_ids=list(range(5,18)),end_gap_id=17,
                          decoder_start_id=102,tokenizer=SimpleNamespace(pad_token_id=0))
    encoded=[{'input_ids':[2,3,4,5][:L] or [2],'gap_length':L} for L in range(1,6)]
    seen=[]
    cpu=torch.Tensor.cpu
    def trace(self,*a,**kw):
        seen.append(list(self.shape));return cpu(self,*a,**kw)
    monkeypatch.setattr(torch.Tensor,'cpu',trace)
    old=old_predict_encoded_batch(tiny,codec,encoded,cfg,torch.device('cpu'))
    old_shapes=list(seen);seen.clear()
    new=predict_encoded_batch(tiny,codec,encoded,cfg,torch.device('cpu'))
    assert old==new  # Includes all candidates, order, exact scores and every rank.
    assert any(len(s)==2 and s[-1]==13 for s in old_shapes)
    assert not any(len(s)==2 and s[-1]==13 for s in seen)
    assert any(len(s)==1 for s in seen)


def smoke_artifact_fixture(tmp_path,cfg,monkeypatch):
    folder=tmp_path/'gpu_smoke';folder.mkdir()
    project=tmp_path/'project';project.mkdir()
    monkeypatch.setattr(formal_controller,'ROOT',project)
    units=[f'c{i}' for i in range(4116)]
    write(tmp_path/'shared/data/v1.5.0/target_vocab.json',{'tokens':units})
    public=[];answers={};predictions=[]
    for i in range(194):
        L=i%5+1
        q={'example_id':f'Q{i}','source_id':f'TRAIN{i}','group_id':'G','gap_length':L,
           'prefix_units':['c0'],'suffix_units':['c1']}
        public.append(q);answers[q['example_id']]=['c0']*L
        n=10 if L==1 else 5
        predictions.append({'paths':[[j]*L for j in range(n)],'scores':[-float(j+1) for j in range(n)],
            'single_ranking':list(range(4116)) if L==1 else None,'end_gap_id':54287,
            'score_kind':'seq2seq_direct_character_log_probability' if L==1 else 'seq2seq_autoregressive_character_plus_end_log_probability'})
    report=launch_extensions(cfg)
    comparison=diagnostics.compare_predictions(predictions,predictions,public,answers,units)
    report['full_prediction_eager_comparison']=comparison
    report['full_prediction_checkpoint_comparison']=comparison
    write_rows(folder/'train_public.jsonl',public);write(folder/'train_gold.json',answers)
    for name in ['generation_before.jsonl','generation_eager.jsonl','generation_after.jsonl']:
        write_rows(folder/name,predictions)
    write(folder/'native_forward_reference.json',report['native_forward_reference'])
    report['diagnostic_artifacts']={p.name:sha(p) for p in folder.iterdir()}
    return folder,report


def test_smoke_artifact_replay_accepts_complete_predictions(tmp_path,cfg,monkeypatch):
    folder,report=smoke_artifact_fixture(tmp_path,cfg,monkeypatch)
    formal_controller.require_smoke_artifacts(folder,report)


@pytest.mark.parametrize('change',['missing','hash','score_rehashed','rank_rehashed','native_rehashed'])
def test_smoke_artifact_replay_rejects_corruption(tmp_path,cfg,monkeypatch,change):
    folder,report=smoke_artifact_fixture(tmp_path,cfg,monkeypatch)
    path=folder/'generation_after.jsonl'
    if change=='missing':path.rename(folder/'moved_for_test.jsonl')
    elif change=='hash':report['diagnostic_artifacts'][path.name]='0'*64
    elif change=='native_rehashed':
        path=folder/'native_forward_reference.json'
        obj=read(path);obj['max_abs_error']=1;write(path,obj)
        report['diagnostic_artifacts'][path.name]=sha(path)
    else:
        obj=rows(path)
        if change=='score_rehashed':obj[0]['scores'][-1]+=1e-8
        else:obj[0]['single_ranking'][-2:]=[3510,3509]
        write_rows(path,obj);report['diagnostic_artifacts'][path.name]=sha(path)
    with pytest.raises(ValueError):formal_controller.require_smoke_artifacts(folder,report)
