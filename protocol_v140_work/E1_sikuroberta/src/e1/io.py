import hashlib
import json
import os
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def rows(path):
    with Path(path).open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(temp, path)


def write_rows(path, values):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    with temp.open('w', encoding='utf-8') as stream:
        for item in values:
            stream.write(json.dumps(item, ensure_ascii=False, allow_nan=False) + '\n')
    os.replace(temp, path)


def verify_data(root):
    from benchmark_protocol_v140 import verify_training_data
    return verify_training_data(root)


def source_fingerprint(root):
    files = [root / 'run.py', root / 'requirements.txt', root.parent / 'shared/data_adapter_v116.py', root.parent / 'shared/benchmark_protocol_v140.py']
    files += sorted((root / 'src').rglob('*.py'))
    return digest({str(p.relative_to(root.parent)): sha(p) for p in files})
