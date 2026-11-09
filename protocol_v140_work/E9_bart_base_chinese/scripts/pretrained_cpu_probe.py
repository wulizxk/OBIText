"""Bounded real-weight CPU loading/numerical probe; no GPU, backward or Test."""
import argparse
import json
import os
from pathlib import Path
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT.parent / 'shared')]


def main():
    import torch
    from ed3.data import Codec, SpanSampler
    from ed3.io import read, write, source_fingerprint, sha
    from ed3.model import load_pretrained, restricted_logits
    from ed3.runtime import configure_execution, runtime_versions
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Choose a new CPU evidence path')
    torch.set_num_threads(1)
    configure_execution(42)
    cfg = read(ROOT / 'configs/ed3a.json')
    data = ROOT.parent / 'shared/data/v1.5.0'
    codec, sampler = Codec(data, cfg), SpanSampler(data, 42)
    model, info = load_pretrained(ROOT / 'weights/ED3', codec, cfg)
    model.eval()
    questions = []
    for length in (1, 2, 5):
        item = sampler.pools[length][0]
        row, start = sampler.rows[item['source_id']], item['starts'][0]
        questions.append({'gap_length': length, 'prefix_units': row['model_tokens'][:start],
                          'suffix_units': row['model_tokens'][start+length:],
                          'gold_units': row['model_tokens'][start:start+length]})
    batch = codec.collate(questions, 'cpu')
    candidates = torch.tensor(codec.candidate_ids)
    with torch.no_grad():
        native = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'],
                       decoder_input_ids=batch['decoder_input_ids'],
                       decoder_attention_mask=batch['decoder_attention_mask'], use_cache=False).logits
        expected = native[batch['owners'], batch['positions']][:, candidates]
        actual = restricted_logits(model, batch['input_ids'], batch['attention_mask'],
            batch['owners'], batch['positions'], candidates,
            batch['decoder_input_ids'], batch['decoder_attention_mask'])
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)
        loss = torch.nn.functional.cross_entropy(actual, batch['targets'])
        if not torch.isfinite(loss):
            raise FloatingPointError('Non-finite real-weight CPU diagnostic loss')
    report = {'scope': 'REAL_WEIGHT_CPU_LOAD_AND_TRAIN_FIXTURE_FORWARD_ONLY', 'status': 'PASS',
              'code_sha256': source_fingerprint(ROOT), 'probe_sha256': sha(Path(__file__)),
              'runtime': runtime_versions(), 'pretrained_loading': info,
              'parameter_count_after_resize': sum(p.numel() for p in model.parameters()),
              'native_vocab': 51271, 'resized_vocab': codec.vocab_size,
              'input_output_tied': model.get_input_embeddings().weight is model.lm_head.weight,
              'train_fixture_questions': 3, 'gap_lengths': [1, 2, 5],
              'restricted_native_logits_max_abs_error': float((actual-expected).abs().max()),
              'diagnostic_token_mean_loss_not_training_epoch_loss': float(loss),
              'device': 'cpu', 'dtype': 'float32', 'gpu_used': False,
              'backward_executed': False, 'optimizer_updates': 0,
              'test_gold_used': False, 'formal_training_started': False,
              'does_not_verify': ['GPU BF16', 'maximum batch GPU memory', 'GPU throughput']}
    write(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
