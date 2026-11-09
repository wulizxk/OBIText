import gzip
import hashlib
import json
import os
from pathlib import Path
import platform
from datetime import datetime, timezone

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    with Path(path).open(encoding='utf-8') as stream:
        return json.load(stream)


def rows(path):
    with Path(path).open(encoding='utf-8') as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    allow_nan=False, separators=(',', ':')).encode()).hexdigest()


def write(path, value):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f'Refusing overwrite: {path}')
    # The parent is a newly reserved, exclusive output directory.
    tmp = path.with_name(path.name + '.partial')
    with tmp.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.flush(); os.fsync(stream.fileno())
    os.replace(tmp, path)


def write_gzip(path, value):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    tmp = path.with_name(path.name + '.partial')
    with tmp.open('xb') as raw:
        with gzip.GzipFile(filename='', fileobj=raw, mode='wb', mtime=0, compresslevel=3) as stream:
            stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False,
                                    separators=(',', ':')).encode())
        raw.flush(); os.fsync(raw.fileno())
    os.replace(tmp, path)


def read_gzip(path):
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        return json.load(stream)


def source_hash():
    files = [ROOT / 'run.py', ROOT / 'requirements.txt', *sorted((ROOT / 'src').rglob('*.py'))]
    return digest({str(p.relative_to(ROOT)): sha(p) for p in files})


def runtime():
    return {'python': platform.python_version(), 'numpy': np.__version__,
            'device': 'cpu', 'numeric_precision': 'float64', 'threads': 1}


def reserve(path, data):
    path, data = Path(path).resolve(), Path(data).resolve()
    if path == data or data in path.parents:
        raise ValueError('Outputs may not be written beneath frozen data')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.mkdir(exist_ok=False)
    return path


def verify_files(root, expected):
    root = Path(root)
    for name, value in expected.items():
        path = (root / name).resolve()
        if root.resolve() not in path.parents or sha(path) != value:
            raise ValueError(f'Artifact checksum mismatch: {name}')
