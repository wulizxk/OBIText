"""Regression evidence for E8 review F1--F4. All outputs are synthetic fixtures.

No real pretrained weights, formal runs, frozen Test model predictions or GPUs.
"""
import copy
from unittest.mock import patch
import benchmark_protocol_v140 as protocol
from contextlib import ExitStack
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from ed2 import cli, engine, evaluation_artifacts as artifacts, formal_controller as controller, run_state
from ed2.io import digest, read, rows, sha, source_fingerprint, write, write_rows
from ed2.metrics import aggregate, score_prediction
from ed2.model import resize_embeddings
from test_audit_fixes import evaluation_report, make_completed_fixture, formal_environment


def build_artifact_fixture(root, cfg):
    """Small synthetic examples for each manuscript suite and gap length."""
    data=root/'data';outputs=root/'outputs'
    targets=[f'SYNTHETIC_UNIT_{i:04}' for i in range(4116)]
    write(data/'target_vocab.json',{'tokens':targets,'size':4116})
    predictions={}
    suite_specs = copy.deepcopy(protocol.EVALUATION_SUITES)
    for suite, spec in suite_specs.items():
        spec['lengths'] = {key: 2 for key in spec['lengths']}
        lengths = [int(key) for key, count in spec['lengths'].items() for _ in range(count)]
        public=[];gold=[];scored=[]
        for i,length in enumerate(lengths):
            q={'example_id':f'SYNTHETIC_{suite}_{i:04}','source_id':f'SYNTHETIC_SOURCE_{suite}_{i:04}',
               'group_id':f'SYNTHETIC_GROUP_{suite}_{i:04}','gap_length':length,
               'prefix_units':[targets[10]],'suffix_units':[targets[11]],'start_unit':1}
            if suite in ('test_prefix', 'test_suffix'):
                q['boundary_side']='prefix' if suite=='test_prefix' else 'suffix'
                q['prefix_units' if suite=='test_prefix' else 'suffix_units']=[]
            public.append(q)
            answer=[targets[0]]*length
            gold.append({'example_id':q['example_id'],'gold_units':answer})
            count=10 if length==1 else 5
            pred={'paths':[[j]*length for j in range(count)],'scores':[-3.-.1*j for j in range(count)],
                  'single_ranking':list(range(4116)) if length==1 else None,
                  'score_kind':'seq2seq_direct_character_log_probability' if length==1 else
                               'seq2seq_autoregressive_character_plus_end_log_probability'}
            scored.append(score_prediction(q,pred,answer,targets))
        public_path = data / protocol.benchmark_relative_path(suite, 'public')
        gold_path = data / protocol.benchmark_relative_path(suite, 'gold')
        write_rows(public_path, public)
        write_rows(gold_path, gold)
        spec['hashes'] = {'public.jsonl': sha(public_path), 'gold.jsonl': sha(gold_path)}
        predictions[suite]=scored
    files={str(p.relative_to(data)):sha(p) for p in data.rglob('*') if p.is_file()}
    write(data/'DATA_SHA256SUMS.json',{'version':'v1.5.0','status':'SYNTHETIC_FIXTURE_ONLY','files':files})
    reports={}
    for seed in cfg['seeds']:
        report=evaluation_report(cfg,seed)
        report['identity'].update(data_sha256=sha(data/'DATA_SHA256SUMS.json'),
                                  code_sha256=source_fingerprint(artifacts.ROOT),
                                  weights_sha256=digest('SYNTHETIC_WEIGHTS_NEVER_LOADED'))
        report['training_completion']['identity_sha256']=digest(report['identity'])
        report['completion_sha256']=digest(report['training_completion'])
        report['suites']={s:aggregate(predictions[s]) for s in artifacts.SUITES}
        folder=outputs/f'ed2a_v150_s{seed}_test'
        for suite in artifacts.SUITES: write_rows(folder/f'{suite}_predictions.jsonl',predictions[suite])
        report['test_artifact_schema_version']=artifacts.ARTIFACT_SCHEMA
        with patch.object(protocol, 'EVALUATION_SUITES', suite_specs), \
             patch.object(protocol, 'DATA_MANIFEST_SHA256', sha(data/'DATA_SHA256SUMS.json')):
            report['prediction_artifacts']=artifacts.prediction_manifest(folder,data)
        reports[seed]=report
    batch={'schema_version':1,'seeds':[42,3407,2026],
           'identity_without_seed_sha256':digest({k:v for k,v in reports[42]['identity'].items() if k!='seed'}),
           'completions':{str(s):reports[s]['training_completion'] for s in cfg['seeds']}}
    for seed,report in reports.items():
        report['training_batch_completion']=batch;report['training_batch_sha256']=digest(batch)
        write(outputs/f'ed2a_v150_s{seed}_test/metrics.json',report)
    return {'data':data,'outputs':outputs,'manifest_sha':sha(data/'DATA_SHA256SUMS.json'), 'suite_specs': suite_specs}


@pytest.mark.parametrize('factor',[.25,1.,2.])
def test_added_input_and_untied_output_use_explicit_native_t5_normal(tiny,factor):
    tiny.config.initializer_factor=factor
    before={k:v.clone() for k,v in tiny.state_dict().items()}
    result=resize_embeddings(tiny,4136)
    for name in ['shared.weight','encoder.embed_tokens.weight','decoder.embed_tokens.weight','lm_head.weight']:
        torch.testing.assert_close(tiny.state_dict()[name][:40],before[name],rtol=0,atol=0)
        new=tiny.state_dict()[name][40:]
        assert abs(new.mean().item())<.025*factor
        assert .97*factor<new.std().item()<1.03*factor
    for name,value in before.items():
        if name not in ['shared.weight','encoder.embed_tokens.weight','decoder.embed_tokens.weight','lm_head.weight']:
            torch.testing.assert_close(tiny.state_dict()[name],value,rtol=0,atol=0)
    assert tiny.shared.weight is tiny.encoder.embed_tokens.weight is tiny.decoder.embed_tokens.weight
    assert tiny.shared.weight is not tiny.lm_head.weight
    assert result['initialization']=='explicit_native_T5_normal_new_rows_only'
    assert result['new_row_normal_std']==factor


def test_resize_same_seed_reproduces_new_tables_and_same_size_is_noop(tiny):
    other=copy.deepcopy(tiny)
    torch.manual_seed(3407);resize_embeddings(tiny,200)
    torch.manual_seed(3407);resize_embeddings(other,200)
    assert torch.equal(tiny.shared.weight,other.shared.weight) and torch.equal(tiny.lm_head.weight,other.lm_head.weight)
    before=copy.deepcopy(tiny.state_dict());rng=torch.get_rng_state()
    assert resize_embeddings(tiny,200)['new_rows_initialized']==0
    assert torch.equal(rng,torch.get_rng_state())
    assert all(torch.equal(v,tiny.state_dict()[k]) for k,v in before.items())


@pytest.mark.parametrize('factor',[0.,-1.,float('nan'),float('inf')])
def test_bad_initializer_rejected_without_resizing(tiny,factor):
    tiny.config.initializer_factor=factor
    with pytest.raises(ValueError,match='initializer_factor'): resize_embeddings(tiny,200)
    assert tiny.shared.weight.shape[0]==40 and tiny.lm_head.weight.shape[0]==40


def refresh_manifest(folder,data,report):
    report['prediction_artifacts']=artifacts.prediction_manifest(folder,data)
    write(folder/'metrics.json',report)


@pytest.mark.parametrize('case',['missing_predictions','truncated','duplicate_id','wrong_id','wrong_gold',
    'wrong_candidate_length','protected_candidate','duplicate_candidate','wrong_scores','wrong_rank','row_metric',
    'metric_missing','metric_out_of_range','metric_nan','metric_boolean','boundary_count','public_sha',
    'prediction_sha','missing_batch','mixed_batch','wrong_code','missing_suite','missing_row_field',
    'mass_over_one','schema_boolean'])
def test_strict_artifact_gate_rejects_corruption(artifact_bundle,case):
    folder=artifact_bundle['directories'][0];data=artifact_bundle['data'];report=read(folder/'metrics.json')
    prediction_path=folder/'test_random_predictions.jsonl'
    records=rows(prediction_path)
    if case=='missing_predictions': prediction_path.rename(folder/'missing.audit_backup')
    elif case=='truncated': records.pop()
    elif case=='duplicate_id': records[1]=copy.deepcopy(records[0])
    elif case=='wrong_id': records[0]['example_id']='SYNTHETIC_NOT_A_FROZEN_ID'
    elif case=='wrong_gold': records[0]['gold_units']=['SYNTHETIC_UNIT_0001']
    elif case=='wrong_candidate_length': records[0]['candidates'][0].append('SYNTHETIC_UNIT_0000')
    elif case=='protected_candidate': records[0]['candidates'][0]=['[LACUNA_1]']
    elif case=='duplicate_candidate': records[0]['candidates'][1]=records[0]['candidates'][0]
    elif case=='wrong_scores': records[0]['candidate_scores'][0]=1.
    elif case=='mass_over_one': records[0]['candidate_scores']=[-.01*(j+1) for j in range(10)]
    elif case=='schema_boolean': report['test_artifact_schema_version']=True
    elif case=='wrong_rank': records[0]['gold_rank_full_vocabulary']=12
    elif case=='row_metric': records[0]['em1']=0.
    elif case=='metric_missing': report['suites']['test_random'].pop('macro5_em1')
    elif case=='metric_out_of_range': report['suites']['test_random']['macro5_em1']=2.
    elif case=='metric_nan': report['suites']['test_random']['macro5_em1']=float('nan')
    elif case=='metric_boolean': report['suites']['test_random']['macro5_em1']=True
    elif case=='boundary_count': report['suites']['test_prefix']['by_length']['1']['questions']=1
    elif case=='public_sha': report['prediction_artifacts']['test_random']['public_sha256']='a'*64
    elif case=='prediction_sha': report['prediction_artifacts']['test_random']['sha256']='b'*64
    elif case=='missing_batch': report.pop('training_batch_completion')
    elif case=='mixed_batch': report['training_batch_completion']['completions']['3407']['identity_sha256']='c'*64
    elif case=='wrong_code': report['identity']['code_sha256']='wrong'
    elif case=='missing_suite': report['suites'].pop('test_expert')
    elif case=='missing_row_field': records[0].pop('boundary_side')
    row_cases={'truncated','duplicate_id','wrong_id','wrong_gold','wrong_candidate_length','protected_candidate',
               'duplicate_candidate','wrong_scores','wrong_rank','row_metric','missing_row_field','mass_over_one'}
    if case in row_cases:
        write_rows(prediction_path,records)
        refresh_manifest(folder,data,report)
    if case=='metric_nan':
        # Deliberately malformed external JSON; production writer disallows NaN.
        import json
        (folder/'metrics.json').write_text(json.dumps(report,allow_nan=True))
    else: write(folder/'metrics.json',report)
    with pytest.raises((ValueError,FileNotFoundError)):
        artifacts.require_test_artifacts(folder,data=data)


def test_summary_and_acceptance_positive_and_corruption(artifact_bundle,tmp_path):
    paths=artifact_bundle['directories'];data=artifact_bundle['data'];summary=tmp_path/'summary.json'
    cli.summarize(paths,summary,data=data)
    assert cli.require_summary_artifacts(summary,paths,data=data)['status']=='accepted'
    result=read(summary)
    assert result['formal_three_seed_complete'] and len(result['artifact_acceptances'])==3
    result['metrics']['/test_random/macro5_em1']['mean']=2.
    write(summary,result)
    with pytest.raises(ValueError): cli.require_summary_artifacts(summary,paths,data=data)


def test_summary_rejects_missing_predictions_in_one_seed(artifact_bundle,tmp_path):
    paths=artifact_bundle['directories'];(paths[1]/'test_expert_predictions.jsonl').rename(paths[1]/'missing.audit_backup')
    with pytest.raises(ValueError): cli.summarize(paths,tmp_path/'no_summary.json',data=artifact_bundle['data'])
    assert not (tmp_path/'no_summary.json').exists()


class FakeProcess:
    pid=999999999
    def __init__(self,*a,**kw): pass
    def wait(self,*a,**kw): return 0
    def poll(self): return 0


@pytest.mark.parametrize('stage',['train','test','summarize'])
def test_exit_zero_missing_artifacts_is_not_succeeded(monkeypatch,tmp_path,stage):
    control=tmp_path/'controller';control.mkdir()
    monkeypatch.setattr(controller,'CONTROL',control);monkeypatch.setattr(controller,'OUTPUTS',tmp_path/'absent')
    monkeypatch.setattr(controller,'SUMMARY',tmp_path/'absent_summary.json')
    monkeypatch.setattr(controller,'require_training_gpus_free',lambda:None)
    if stage=='test': monkeypatch.setattr(controller,'require_completed_run',lambda *_:({},{}))
    monkeypatch.setattr(controller.subprocess,'Popen',FakeProcess)
    state={'jobs':{}}
    assert controller.phase(stage,state) is False
    job=next(iter(state['jobs'][stage].values()))
    assert job['status']=='artifact_validation_failed' and job['exit_code']==job['process_exit_code']==0
    assert job['artifact_validation_status']=='failed' and job['validation_error']
    assert len(state['jobs'][stage])==1  # no subsequent seed after artifact failure


def test_controller_only_commits_success_after_acceptance(monkeypatch,tmp_path):
    monkeypatch.setattr(controller,'CONTROL',tmp_path)
    monkeypatch.setattr(controller,'require_training_gpus_free',lambda:None)
    monkeypatch.setattr(controller.subprocess,'Popen',FakeProcess)
    state={'jobs':{}};calls=[]
    def accept(stage,seed):
        assert state['jobs'][stage][str(seed)]['status']=='validating'
        assert state['jobs'][stage][str(seed)]['artifact_validation_status']=='pending'
        calls.append(seed);return {'status':'accepted','seed':seed}
    monkeypatch.setattr(controller,'accept_stage_artifacts',accept)
    assert controller.phase('train',state) is True
    assert calls==[42,3407,2026]
    assert all(j['status']=='succeeded' and j['artifact_validation_status']=='accepted' for j in state['jobs']['train'].values())


def make_batch(tmp_path,cfg,tiny,identity_overrides=None):
    runs=[]
    for seed in cfg['seeds']:
        out,_=make_completed_fixture(tmp_path,cfg,tiny,seed=seed,name=f'ed2a_v150_s{seed}',identity_overrides=identity_overrides)
        runs.append(out)
    return runs


def test_direct_test_requires_other_two_seeds_before_dispatch(tmp_path,cfg,tiny,monkeypatch):
    out,_=make_completed_fixture(tmp_path,cfg,tiny)
    monkeypatch.setattr(engine,'require_formal_cuda',lambda d,*_:torch.device(d))
    monkeypatch.setattr(engine,'_evaluate_checkpoint',lambda *a,**kw:pytest.fail('No Test dispatch'))
    with pytest.raises(ValueError,match='All three seeds'):
        engine.evaluate_checkpoint(tmp_path,tmp_path,out/'best.pt',tmp_path/'forbidden',torch.device('cpu'),
                                   cfg['test_suites'],confirm_final=True)
    assert not (tmp_path/'forbidden').exists()


@pytest.mark.parametrize('case',['missing','duplicate_path','wrong_seed','mixed_identity','active_writer'])
def test_batch_gate_rejects_invalid_or_active_sibling(tmp_path,cfg,tiny,case):
    runs=make_batch(tmp_path,cfg,tiny)
    if case=='missing': (runs[1]/'TRAINING_COMPLETE.json').rename(runs[1]/'marker.audit_backup')
    if case=='duplicate_path': runs[2]=runs[1]
    if case in ('wrong_seed','mixed_identity'):
        # Recreate a separately valid completion with a duplicate seed / different identity.
        alternate,_=make_completed_fixture(tmp_path,cfg,tiny,seed=42 if case=='wrong_seed' else 2026,
                    name='alternate',identity_overrides={'weights_sha256':'different'} if case=='mixed_identity' else None)
        runs[2]=alternate
    with ExitStack() as stack:
        if case=='active_writer':
            import fcntl
            handle=stack.enter_context((runs[1]/'.run.lock').open('a'))
            fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with pytest.raises((ValueError,BlockingIOError)):
            with run_state.completed_training_batch(runs[0]/'best.pt',runs): pytest.fail('Batch should be rejected')


def test_complete_batch_accepts_default_and_explicit_paths(tmp_path,cfg,tiny):
    runs=make_batch(tmp_path,cfg,tiny)
    for seed,root in zip(cfg['seeds'],runs):
        with run_state.completed_training_batch(root/'best.pt') as (point,marker,batch):
            assert point['identity']['seed']==seed
            assert batch['completions'][str(seed)]==marker and set(batch['completions'])=={'42','3407','2026'}
    with run_state.completed_training_batch(runs[0]/'best.pt',list(reversed(runs))) as (_,_,batch):
        assert batch['seeds']==cfg['seeds']


def test_test_writer_and_acceptance_integrate_with_three_completions(artifact_bundle,tmp_path,cfg,tiny,monkeypatch):
    data=artifact_bundle['data'];template=read(artifact_bundle['directories'][0]/'metrics.json')
    overrides={k:v for k,v in template['identity'].items() if k not in ('cfg','seed')}
    runs=make_batch(tmp_path,cfg,tiny,identity_overrides=overrides)
    contract=artifacts.load_test_contract(data)
    monkeypatch.setattr(engine,'require_formal_cuda',lambda d,*_:torch.device(d))
    monkeypatch.setattr(engine,'evaluation_environment',lambda *_:formal_environment(cfg))
    monkeypatch.setattr(engine,'inspect_checkpoint',lambda project,data,path,device,checkpoint:
                        (checkpoint,cfg,None,None))
    monkeypatch.setattr(engine,'benchmark_public',lambda _,s:contract['suites'][s]['public'])
    monkeypatch.setattr(engine,'benchmark_gold',lambda _,s,p:contract['suites'][s]['gold'])
    def evaluate(model,codec,public,answers,*_):
        suite=next(s for s in artifacts.SUITES if public[0]['example_id']==contract['suites'][s]['public'][0]['example_id'])
        assert callable(answers)
        return template['suites'][suite],rows(artifact_bundle['directories'][0]/f'{suite}_predictions.jsonl')
    monkeypatch.setattr(engine,'evaluate',evaluate)
    result=tmp_path/'synthetic_test_writer'
    engine.evaluate_checkpoint(artifacts.ROOT,data,runs[0]/'best.pt',result,torch.device('cpu'),cfg['test_suites'],True)
    accepted=artifacts.require_test_artifacts(result,data=data)
    assert accepted['status']=='accepted' and accepted['seed']==42
