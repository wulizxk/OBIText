"""Download only the pinned public GPT2-Chinese-Ancient checkpoint."""
import argparse
import os
from pathlib import Path

from .io import read, sha, write


def fetch(folder, cfg):
    os.environ.setdefault('HF_HUB_DISABLE_XET', '1')
    from huggingface_hub import HfApi, hf_hub_download
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    info = HfApi().model_info(cfg['model_repo'], revision=cfg['model_revision'], files_metadata=True, timeout=30)
    if info.sha != cfg['model_revision']:
        raise ValueError('Resolved revision differs from frozen tokenizer revision')
    siblings = {s.rfilename: s for s in info.siblings}
    weight = next((n for n in ('model.safetensors', 'pytorch_model.bin') if n in siblings), None)
    if weight is None:
        raise ValueError('No supported single-file weights in pinned checkpoint')
    hashes = {}
    for name in ('config.json', 'vocab.txt', weight):
        path = hf_hub_download(cfg['model_repo'], name, revision=cfg['model_revision'], local_dir=folder)
        actual = sha(path)
        lfs = siblings[name].lfs
        if lfs is not None and actual != lfs.sha256:
            raise ValueError('Downloaded LFS weight checksum mismatch')
        hashes[name] = actual
    if hashes['config.json'] != cfg['native_config_sha256']:
        raise ValueError('Native config checksum mismatch')
    if hashes['vocab.txt'] != cfg['native_vocab_sha256']:
        raise ValueError('Native vocabulary checksum mismatch')
    write(folder / 'provenance.json', {'repo': cfg['model_repo'], 'revision': info.sha,
                                      'hashes': hashes, 'weight_file': weight})
    # Re-run the same offline gate used by training so a stale competing weight
    # file or an incomplete provenance manifest cannot survive deployment.
    from .model import verify_weights
    verify_weights(folder, cfg)
    print('PRETRAINED_DOWNLOAD_VERIFIED', hashes, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    fetch(args.output, read(args.config))
