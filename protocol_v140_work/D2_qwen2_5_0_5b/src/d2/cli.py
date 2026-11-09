import argparse
import json
import re
import statistics
from pathlib import Path
from benchmark_protocol_v140 import validate_formal_suite_metrics

from .io import read, verify_data, write, digest


def summarize(paths, output):
    from .runtime import require_evaluation_report
    from .protocol import validate_formal_config
    from .run_state import SCHEMA_VERSION, steps_for
    if not paths:
        raise ValueError('Provide at least one evaluation report')
    reports = [read(Path(p) / 'metrics.json') for p in paths]
    environments = [require_evaluation_report(r) for r in reports]
    required_seeds = {42, 3407, 2026}
    if len(reports) != 3 or {r['seed'] for r in reports} != required_seeds:
        raise ValueError('Formal summary requires exactly seeds 42, 3407 and 2026')
    completion_digests = []
    for report, environment in zip(reports, environments):
        cfg = validate_formal_config(report['identity']['cfg'])
        if environment.get('hardware', {}).get('type') != 'cuda' or environment.get('effective_precision') != 'fp16':
            raise ValueError('Formal summary requires CUDA FP16 evaluation reports')
        completion = report.get('training_completion')
        completion_sha = report.get('completion_sha256')
        if not isinstance(completion, dict) or digest(completion) != completion_sha:
            raise ValueError('Formal summary requires the verified training-completion record and digest')
        if (completion.get('schema_version') != SCHEMA_VERSION or completion.get('status') != 'complete'
                or completion.get('identity_sha256') != digest(report['identity'])):
            raise ValueError('Training completion is not bound to this evaluation identity')
        best = completion.get('best', {})
        artifacts = completion.get('artifacts', {})
        epochs = completion.get('epochs_ran')
        stopping = completion.get('stopping', {})
        required_artifacts = {'last.pt', 'best.pt', 'history.json', 'best_metrics.json',
                              'run.json', 'dev_best_predictions.jsonl'}
        valid_hashes = (set(artifacts) == required_artifacts
                        and all(isinstance(value, str) and re.fullmatch(r'[0-9a-f]{64}', value)
                                for value in artifacts.values()))
        if (completion.get('max_epochs') != cfg['max_epochs'] or type(epochs) is not int
                or not 1 <= epochs <= cfg['max_epochs']
                or completion.get('steps') != steps_for(cfg, epochs)
                or completion.get('test_evaluated') is not False or not valid_hashes):
            raise ValueError('Training completion structure or artifact manifest is invalid')
        without_improvement = epochs - best.get('epoch', -1)
        if (stopping.get('patience') != cfg['early_stopping_patience']
                or stopping.get('epochs_without_improvement') != without_improvement
                or stopping.get('reason') not in {'early_stopping', 'max_epochs'}
                or (stopping.get('reason') == 'early_stopping'
                    and (epochs >= cfg['max_epochs'] or without_improvement != cfg['early_stopping_patience']))
                or (stopping.get('reason') == 'max_epochs' and epochs != cfg['max_epochs'])):
            raise ValueError('Training completion stopping record is invalid')
        if (best.get('epoch') != report.get('epoch')
                or best.get('checkpoint_sha256') != report.get('checkpoint_sha256')
                or artifacts.get('best.pt') != report.get('checkpoint_sha256')
                or artifacts.get('dev_best_predictions.jsonl') != best.get('predictions_sha256')):
            raise ValueError('Training completion is not bound to the evaluated best checkpoint')
        completion_digests.append(completion_sha)
        validate_formal_suite_metrics(report.get('suites', {}))

    if any(env != environments[0] for env in environments):
        raise ValueError('Cannot aggregate different actual evaluation environments/precision/hardware')
    identities = [{k: v for k, v in r['identity'].items() if k != 'seed'} for r in reports]
    if any(i != identities[0] for i in identities):
        raise ValueError('Cannot aggregate different configurations/data/code/runtime')
    if len({r['seed'] for r in reports}) != len(reports):
        raise ValueError('Duplicate seeds in summary')
    if len(set(completion_digests)) != len(reports):
        raise ValueError('Distinct seeds must carry distinct bound training completions')

    def flatten(obj, prefix=''):
        result = {}
        for k, v in obj.items():
            key = prefix + '/' + k
            if isinstance(v, dict):
                result.update(flatten(v, key))
            elif isinstance(v, (float, int)):
                result[key] = v
        return result
    flat = [flatten(r['suites']) for r in reports]
    if any(set(r) != set(flat[0]) for r in flat):
        raise ValueError('Different evaluated suites/metrics')
    summary = {'seeds': [r['seed'] for r in reports], 'n': len(reports),
               'formal_three_seed_complete': True,
               'metrics': {k: {'mean': statistics.mean(r[k] for r in flat),
                               'sample_std': statistics.stdev(r[k] for r in flat) if len(flat) > 1 else None}
                           for k in flat[0]}, 'identity_without_seed': identities[0],
               'evaluation_runtime': environments[0]}
    if Path(output).exists():
        raise FileExistsError('Summary output exists')
    write(output, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main(project):
    parser = argparse.ArgumentParser(description='D2-A: Qwen2.5-0.5B FIM transcription restoration; no DAPT')
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument('--config', type=Path, default=project / 'configs/d2a.json')
    common.add_argument('--data', type=Path, default=project.parent / 'shared/data/v1.5.0')
    common.add_argument('--weights', type=Path, default=project.parent / 'shared/pretrained/D2')
    common.add_argument('--device', default='cuda:0')
    subs = parser.add_subparsers(dest='command', required=True)
    for name in ('verify', 'fetch-model'):
        subs.add_parser(name, parents=[common])
    training = subs.add_parser('train', parents=[common])
    training.add_argument('--seed', type=int, required=True, choices=[42, 3407, 2026])
    training.add_argument('--output', type=Path, required=True)
    training.add_argument('--resume', action='store_true')
    for name in ('smoke',):
        subs.add_parser(name, parents=[common]).add_argument('--output', type=Path, required=True)
    for name in ('validate', 'test'):
        p = subs.add_parser(name, parents=[common])
        p.add_argument('--checkpoint', type=Path, required=True)
        p.add_argument('--output', type=Path, required=True)
        if name == 'test':
            p.add_argument('--confirm-final', action='store_true', help='Model/hyperparameters are finalized before reading Test results')
    p = subs.add_parser('predict', parents=[common])
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--text', required=True, help='Normalized transcription with one contiguous [MASK] span')
    p = subs.add_parser('summarize')
    p.add_argument('--evaluations', nargs='+', required=True)
    p.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.command == 'summarize':
        return summarize(args.evaluations, args.output)
    cfg = read(args.config)
    if args.command == 'verify':
        from .data import Codec
        from .engine import validate_config
        validate_config(cfg)
        checksum = verify_data(args.data)
        codec = Codec(args.data, cfg)
        print(json.dumps({'status': 'PASS', 'data_manifest_sha256': checksum,
                          'input_vocab_size': codec.vocab_size, 'prediction_targets': len(codec.units),
                          'decoder_control_target': '<END_GAP>', 'architecture': cfg['architecture'],
                          'dev_questions': 15024, 'dapt': False}, indent=2))
        return
    if args.command == 'fetch-model':
        from .download import fetch
        return fetch(args.weights, cfg)
    import torch
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; set --device cpu only for diagnostics')
    from .engine import train, evaluate_checkpoint, inspect_checkpoint
    if args.command == 'train':
        return train(project, args.data, args.weights, args.output, cfg, args.seed, device, args.resume)
    if args.command == 'smoke':
        from .smoke import smoke
        return smoke(project, args.data, args.weights, args.output, cfg, device)
    if args.command in ('test', 'validate'):
        if args.command == 'test':
            if not args.confirm_final:
                parser.error('Test requires --confirm-final; do not use Test for tuning or checkpoint selection')
        suites = cfg['test_suites'] if args.command == 'test' else ['dev_full']
        return evaluate_checkpoint(project, args.data, args.checkpoint, args.output, device, suites,
                                   confirm_final=args.command == 'test' and args.confirm_final)
    if args.command == 'predict':
        from .decoding import predict_batch
        _, saved_cfg, codec, model = inspect_checkpoint(project, args.data, args.checkpoint, device)
        q = codec.parse_text(args.text)
        prediction = predict_batch(model, codec, [q], saved_cfg, device)[0]
        results = [{'rank': i + 1, 'tokens': [codec.units[j] for j in path],
                    'text': ''.join(codec.units[j] for j in path), 'pseudo_log_probability': score}
                   for i, (path, score) in enumerate(zip(prediction['paths'], prediction['scores']))]
        print(json.dumps({'input': args.text, 'gap_length': q['gap_length'], 'predictions': results}, ensure_ascii=False, indent=2))
