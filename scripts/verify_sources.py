"""Verify release source identities without importing neural libraries."""
import hashlib
import json
from release_utils import REPO, read, sha


def main():
    root = REPO / 'protocol_v140_work'
    registry = read(REPO / 'provenance/training_source_fingerprints.json')
    for code, info in registry.items():
        project = root / info['project']
        files = [project / 'run.py', project / 'requirements.txt',
                 root / 'shared/data_adapter_v116.py', root / 'shared/benchmark_protocol_v140.py']
        files += sorted((project / 'src').rglob('*.py'))
        if code == 'ED2':
            files += [project / 'scripts/run_formal_earlystop.py']
        elif code == 'ED3':
            files += sorted((project / 'scripts').glob('*.py'))
        payload = {str(p.relative_to(root)): sha(p) for p in files}
        actual = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        if actual != info['current_source_sha256']:
            raise ValueError('Release source changed: ' + code)
        print(code + ': matched')
    inventory = REPO / 'SHA256SUMS.json'
    if inventory.exists():
        for name, expected in read(inventory).items():
            if sha(REPO / name) != expected:
                raise ValueError('Release file changed: ' + name)
        print('Release inventory: matched')
    baseline = root / 'B1_B2_B6_baselines'
    files = [baseline / 'run.py', baseline / 'requirements.txt', *sorted((baseline / 'src').rglob('*.py'))]
    payload = {str(p.relative_to(baseline)): sha(p) for p in files}
    actual = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False,
                                      allow_nan=False, separators=(',', ':')).encode()).hexdigest()
    baseline_expected = read(REPO / 'provenance/baseline_source_fingerprints.json')['current_source_sha256']
    if actual != baseline_expected:
        raise ValueError('Release source changed: deterministic baselines')
    print('All neural and deterministic release source identities matched.')


if __name__ == '__main__':
    main()
