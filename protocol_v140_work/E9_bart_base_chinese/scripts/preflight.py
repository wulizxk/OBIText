"""CPU-only build preflight; no GPU access, model loading or formal dispatch."""
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

from ed3.io import read, sha, source_fingerprint, verify_data, write
from ed3.protocol import FORMAL_CONFIG_SHA256, validate_formal_config
from ed3.data import Codec


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, help='Optional new evidence path; never overwrite')
    args = parser.parse_args(argv)
    if args.output and args.output.exists():
        raise FileExistsError('Choose a new preflight evidence path')
    cfg = validate_formal_config(read(ROOT / 'configs/ed3a.json'))
    if sha(ROOT / 'configs/ed3a.json') != FORMAL_CONFIG_SHA256:
        raise ValueError('E9 configuration fingerprint mismatch')
    frozen = read(ROOT / 'verification/source_freeze_v150.json')
    current = source_fingerprint(ROOT)
    if frozen['code_sha256'] != current:
        raise ValueError('E9 source changed after the recorded build')
    data = ROOT.parent / 'shared/data/v1.5.0'
    data_sha = verify_data(data)
    codec = Codec(data, cfg)
    # Public-only structural checks; no Test gold parsing or model execution.
    from ed3.data import benchmark_public
    counts = {suite: len(benchmark_public(data, suite))
              for suite in ['dev_full']}
    if counts != {'dev_full': 15024}:
        raise ValueError('Public benchmark counts differ from the frozen protocol')
    weights = ROOT / 'weights/ED3'
    weights_status = 'NOT_DOWNLOADED'
    if (weights / 'provenance.json').is_file():
        from ed3.model import verify_weights
        verify_weights(weights, cfg)
        weights_status = 'SHA_VERIFIED_NOT_LOADED'
    report = {'scope': 'CPU code/data preflight only', 'status': 'PASS',
              'code_sha256': current, 'config_sha256': FORMAL_CONFIG_SHA256,
              'data_manifest_sha256': data_sha, 'public_question_counts': counts,
              'input_unit_types': codec.mapping['obi_input_types'], 'prediction_targets': len(codec.units),
              'model_vocab_ids': codec.vocab_size, 'weights_status': weights_status,
              'gpu_smoke_executed': False, 'formal_training_started': False,
              'formal_readiness': 'NOT_ASSESSED_REQUIRES_INDEPENDENT_REVIEW_AND_REAL_WEIGHT_GPU_SMOKE'}
    if args.output:
        write(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
