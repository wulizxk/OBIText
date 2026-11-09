"""Read-only formal readiness checks; writes one new diagnostic report only."""
import argparse
from datetime import datetime
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT.parent / 'shared')]
from ed2 import formal_controller as controller
from ed2.io import read, write, sha, source_fingerprint


def tests(path):
    cases = ET.parse(path).getroot().findall('.//testcase')
    assert cases and all(c.find('failure') is None and c.find('error') is None for c in cases)
    return {'passed': sum(c.find('skipped') is None for c in cases),
            'skipped': sum(c.find('skipped') is not None for c in cases), 'sha256': sha(path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--full-dev-report', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve existing evidence; choose a new report')
    code = source_fingerprint(ROOT)
    freeze = read(ROOT / 'verification/source_freeze_v150.json')
    assert code == freeze['code_sha256']
    controller.check_frozen(code)
    controller.require_gpu_smoke(code)
    controller.require_training_gpus_free()
    assert not any((ROOT / 'outputs').iterdir()), 'Formal outputs must remain empty'
    full = read(args.full_dev_report)
    assert full['status'] == 'PASS' and full['code_sha256'] == code
    assert full['config_sha256'] == sha(ROOT / 'configs/ed2a.json')
    assert full['data_sha256'] == freeze['data_manifest_sha256']
    assert full['test_answers_used'] is False and full['training_resumed'] is False
    assert [c['questions'] for c in full['cases']] == [15024, 15024]
    result = full['cases'][-1]
    assert all(result[k] for k in ('all_metrics_identical', 'all_raw_predictions_identical',
                                   'all_scored_predictions_identical'))
    evidence = ROOT / 'verification/validation_speed_probe_20260908'
    cpu, gpu = tests(evidence / 'cpu_tests.xml'), tests(evidence / 'gpu_tests.xml')
    assert cpu['passed'] == 214 and gpu['passed'] == 8 and gpu['skipped'] == 0
    assert controller.main(['--plan'], expected_code_sha=code) == 0
    report = {'status': 'PASS_READY_NOT_RESTARTED',
              'checked_beijing': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
              'code_sha256': code, 'config_sha256': full['config_sha256'],
              'data_sha256': full['data_sha256'], 'cpu_tests': cpu, 'gpu_tests': gpu,
              'real_weight_gpu_smoke': 'PASS',
              'gpu_smoke_sha256': sha(ROOT / 'verification/gpu_smoke_v150/smoke_report.json'),
              'paired_full_dev_report': str(args.full_dev_report),
              'paired_full_dev_report_sha256': sha(args.full_dev_report),
              'full_dev_equality': 'ALL_RAW_AND_SCORED_PREDICTIONS_AND_METRICS_EXACT',
              'reference_seconds': full['cases'][0]['seconds_including_cold_capture'],
              'accelerated_seconds': result['seconds_including_cold_capture'],
              'speedup': result['speedup'], 'formal_controller_checks': 'PASS',
              'formal_outputs_empty': True, 'gpu_compute_jobs_empty': True,
              'formal_training_restarted': False, 'formal_test_started': False}
    write(args.output, report)
    print(report, flush=True)


if __name__ == '__main__':
    main()
