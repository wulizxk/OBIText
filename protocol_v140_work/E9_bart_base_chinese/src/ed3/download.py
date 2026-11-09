"""Pinned SHA-verified HTTPS downloads, independent of mirror HEAD headers."""
import os
from pathlib import Path
import tempfile

from .io import read, sha, write


def fetch(folder, cfg):
    import requests
    from .protocol import validate_formal_config
    validate_formal_config(cfg)
    endpoint = os.environ.get('HF_ENDPOINT', 'https://huggingface.co').rstrip('/')
    if endpoint not in {'https://huggingface.co', 'https://hf-mirror.com'}:
        raise ValueError('Use an approved public-model download endpoint')
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    hashes = {'config.json': cfg['native_config_sha256'], 'vocab.txt': cfg['native_vocab_sha256'],
              'model.safetensors': cfg['native_weight_sha256']}
    cache = folder / '.cache'
    cache.mkdir(exist_ok=True)
    for name, expected in hashes.items():
        target = folder / name
        if target.exists():
            if sha(target) != expected:
                raise ValueError('Refusing to overwrite a different model file: ' + name)
            continue
        url = f'{endpoint}/{cfg["model_repo"]}/resolve/{cfg["model_revision"]}/{name}'
        print('FETCH_PUBLIC', name, cfg['model_revision'], flush=True)
        with requests.get(url, stream=True, timeout=(15, 60)) as response:
            response.raise_for_status()
            with tempfile.NamedTemporaryFile(dir=cache, prefix=name + '.', delete=False) as stream:
                partial = Path(stream.name)
                for block in response.iter_content(1024 * 1024):
                    stream.write(block)
        if sha(partial) != expected:
            raise ValueError('Downloaded file does not match frozen SHA; retained in cache: ' + name)
        if target.exists():
            raise FileExistsError('A competing download published this target: ' + name)
        partial.replace(target)
    provenance = {'repo': cfg['model_repo'], 'revision': cfg['model_revision'],
                  'hashes': hashes, 'weight_file': 'model.safetensors'}
    if (folder / 'provenance.json').exists() and read(folder / 'provenance.json') != provenance:
        raise ValueError('Refusing to overwrite a different pretrained provenance')
    write(folder / 'provenance.json', provenance)
    from .model import verify_weights
    verify_weights(folder, cfg)
    print('PRETRAINED_DOWNLOAD_VERIFIED', hashes, flush=True)
