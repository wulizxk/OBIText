import hashlib

import pytest

from ed2 import download


def fixture_config(cfg):
    contents = {'config.json': b'{}', 'spiece.model': b'piece-fixture', 'pytorch_model.bin': b'weight-fixture'}
    hashes = {name: hashlib.sha256(value).hexdigest() for name, value in contents.items()}
    return dict(cfg, native_config_sha256=hashes['config.json'], native_vocab_sha256=hashes['spiece.model'],
                native_weight_sha256=hashes['pytorch_model.bin']), contents


def test_downloader_rejects_bad_bytes_before_publishing(cfg, tmp_path, monkeypatch):
    import requests
    from ed2 import protocol
    config, _ = fixture_config(cfg)
    monkeypatch.setattr(protocol, 'validate_formal_config', lambda c: c)

    class Response:
        def __enter__(self): return self
        def __exit__(self, *_): pass
        def raise_for_status(self): pass
        def iter_content(self, _): yield b'wrong-download'

    monkeypatch.setattr(requests, 'get', lambda *a, **k: Response())
    with pytest.raises(ValueError, match='frozen SHA'):
        download.fetch(tmp_path, config)
    assert not (tmp_path / 'config.json').exists()
    assert not (tmp_path / 'provenance.json').exists()


def test_downloader_reuses_only_sha_matching_files_and_pins_url(cfg, tmp_path, monkeypatch):
    import requests
    from ed2 import protocol
    config, contents = fixture_config(cfg)
    monkeypatch.setattr(protocol, 'validate_formal_config', lambda c: c)
    urls = []

    class Response:
        def __init__(self, content): self.content = content
        def __enter__(self): return self
        def __exit__(self, *_): pass
        def raise_for_status(self): pass
        def iter_content(self, _): yield self.content

    def get(url, **kwargs):
        urls.append(url)
        assert config['model_revision'] in url
        return Response(contents[url.rsplit('/', 1)[-1]])

    monkeypatch.setattr(requests, 'get', get)
    download.fetch(tmp_path, config)
    assert len(urls) == 3
    download.fetch(tmp_path, config)
    assert len(urls) == 3
    (tmp_path / 'config.json').write_bytes(b'changed')
    with pytest.raises(ValueError, match='Refusing to overwrite'):
        download.fetch(tmp_path, config)


def test_downloader_rejects_unapproved_endpoint_before_writes(cfg, tmp_path, monkeypatch):
    monkeypatch.setenv('HF_ENDPOINT', 'http://untrusted.invalid')
    with pytest.raises(ValueError, match='endpoint'):
        download.fetch(tmp_path / 'unused', cfg)
    assert not (tmp_path / 'unused').exists()
