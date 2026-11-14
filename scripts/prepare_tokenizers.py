"""Rebuild OBIText tokenizers from pinned upstream assets and frozen unit IDs."""

import argparse
import json
from pathlib import Path
import shutil
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'protocol_v140_work/shared'))
from benchmark_protocol_v140 import data_manifest, sha_file


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def prepare(root, asset, spec, native_root):
    from transformers import AddedToken, AutoTokenizer

    dest = root / 'tokenizers' / asset
    mapping = read(dest / 'unit_mapping.json')
    expected = spec['rebuilt_files']
    if all((dest / name).is_file() and sha_file(dest / name) == digest
           for name, digest in expected.items()):
        print(asset + ': verified', flush=True)
        return
    with tempfile.TemporaryDirectory(prefix='.prepare-', dir=root / 'tokenizers') as tmp:
        staging = Path(tmp)
        native = staging / 'native'
        native.mkdir()
        for name, digest in spec['upstream_files'].items():
            if native_root:
                source = native_root / asset / name
            else:
                from huggingface_hub import hf_hub_download
                source = Path(hf_hub_download(spec['repo'], name, revision=spec['revision']))
            if sha_file(source) != digest:
                raise ValueError(f'{asset}: upstream checksum mismatch: {name}')
            shutil.copyfile(source, native / name)
        tok = AutoTokenizer.from_pretrained(
            native, local_files_only=True, use_fast=True, trust_remote_code=False)
        units = mapping['unit_to_model_id']
        controls = mapping['control_to_model_id']
        tok.add_tokens([AddedToken(unit, normalized=False) for unit in sorted(units)])
        tok.add_special_tokens({'additional_special_tokens': list(controls)})
        if tok.pad_token_id is None:
            tok.pad_token = '<OBI_PAD>'
        if (len(tok) != mapping['extended_vocab_size']
                or any(tok.encode(unit, add_special_tokens=False) != [i] for unit, i in units.items())
                or any(tok.convert_tokens_to_ids(unit) != i for unit, i in controls.items())):
            raise ValueError(f'{asset}: reconstructed model vocabulary differs')
        rebuilt = staging / 'rebuilt'
        tok.save_pretrained(rebuilt)
        for name, digest in expected.items():
            if sha_file(rebuilt / name) != digest:
                raise ValueError(f'{asset}: reconstructed file checksum mismatch: {name}')
        for name in expected:
            (rebuilt / name).replace(dest / name)
    print(asset + ': rebuilt and verified', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', required=True, type=Path)
    p.add_argument('--native-root', type=Path, help='Optional local upstream files grouped by asset')
    args = p.parse_args()
    root = args.data.resolve()
    manifest, _ = data_manifest(root)
    inventory = root / 'tokenizer_sources.json'
    if manifest.get('edition') != 'public':
        p.error('This command is for the OBIText v1.5.0 public edition.')
    if sha_file(inventory) != manifest['files']['tokenizer_sources.json']:
        raise ValueError('Tokenizer source inventory checksum mismatch')
    for asset, spec in read(inventory).items():
        mapping_file = root / 'tokenizers' / asset / 'unit_mapping.json'
        name = mapping_file.relative_to(root).as_posix()
        if sha_file(mapping_file) != manifest['files'][name]:
            raise ValueError(f'{asset}: public unit mapping checksum mismatch')
        prepare(root, asset, spec, args.native_root)


if __name__ == '__main__':
    main()
