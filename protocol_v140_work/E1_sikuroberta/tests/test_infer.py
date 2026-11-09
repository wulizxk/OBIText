"""CPU-only user-wrapper checks; no real checkpoint or Test answers required."""
import copy
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch
from transformers import BertConfig, BertForMaskedLM

from e1.decoding import predict_encoded_batch
from e1 import engine

ROOT = Path(__file__).resolve().parents[1]
INFER_SCRIPT = Path(os.environ.get("E1_INFERENCE_SCRIPT", str(ROOT / "infer.py")))
spec = importlib.util.spec_from_file_location("e1_user_infer", INFER_SCRIPT)
infer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(infer)


@pytest.fixture
def predictor(codec, cfg):
    torch.manual_seed(29)
    config = BertConfig(vocab_size=codec.vocab_size, hidden_size=16, num_hidden_layers=1,
                        num_attention_heads=2, intermediate_size=32, max_position_embeddings=128,
                        hidden_dropout_prob=0, attention_probs_dropout_prob=0)
    config._attn_implementation = "eager"
    value = infer.E1Predictor.__new__(infer.E1Predictor)
    value.cfg = copy.deepcopy(cfg)
    value.codec = codec
    value.device = torch.device("cpu")
    value.model = BertForMaskedLM(config).eval()
    value.load_seconds = 0.0
    value.metadata = {"seed": 42, "best_epoch": 27, "device": "cpu",
                      "effective_precision": "fp32", "training_precision": "fp16",
                      "fixture": "tiny_random_model_not_research_result"}
    return value


@pytest.mark.parametrize("text", ["王其[MASK]于□…", "王其[mask]于□…", "王其<mask>于□…"])
def test_aliases_and_natural_damage(text):
    assert infer.normalize_masks(text) == "王其[MASK]于□…"


@pytest.mark.parametrize("text", ["", "王其田", "王□…", "王[MASK]其[MASK]", "王" + "[MASK]" * 6,
                                 "王 [MASK]", "王[MASK]\n", "王[MASK]\t吉"])
def test_reject_invalid_syntax(text):
    with pytest.raises(ValueError):
        infer.normalize_masks(text)


@pytest.mark.parametrize("length", [1, 2, 3, 4, 5])
def test_top10_reuses_original_search_and_preserves_first_five(predictor, length):
    text = "王其" + "[MASK]" * length + "于□…"
    before = copy.deepcopy(predictor.cfg)
    q = predictor.codec.parse_text(text)
    legacy = predict_encoded_batch(predictor.model, predictor.codec, [predictor.codec.encode(q)],
                                   predictor.cfg, predictor.device)[0]
    out = predictor.predict(text)
    assert predictor.cfg == before
    assert len(out["top_k"]) == 10
    keep = 10 if length == 1 else 5
    assert [r["answer_units"] for r in out["top_k"][:keep]] == [
        [predictor.codec.units[i] for i in path] for path in legacy["paths"]]
    assert [r["score"] for r in out["top_k"][:keep]] == legacy["scores"]
    assert out["completed_text"] == "王其" + out["answer"] + "于□…"
    assert out["input"]["damage_unit_count"] == 2
    assert out["input"]["gap_length"] == length
    assert out["input"]["mask_unit_positions_1based"] == list(range(3, 3 + length))
    assert out["decoding"]["target_vocab_size"] == 4116
    assert len({tuple(row["answer_units"]) for row in out["top_k"]}) == 10
    if length == 1:
        assert out["decoding"]["exact_global_top_k"]
        assert out["top_k"][0]["probability"] == math.exp(legacy["scores"][0])
        assert 0 < out["decoding"]["top_k_probability_mass"] < 1
    else:
        assert not out["decoding"]["exact_global_top_k"]
        assert all(row["probability"] is None for row in out["top_k"])
        assert out["decoding"]["top_k_probability_mass"] is None
    assert any("FP32" in note and "FP16" in note for note in out["notes"])
    assert "模型答案：" in infer.render_text(out)
    assert json.loads(json.dumps(out, ensure_ascii=False, allow_nan=False))["answer"] == out["answer"]


@pytest.mark.parametrize("text,side", [("[MASK]王其田", "initial"), ("王其田[MASK]", "final")])
def test_boundaries(predictor, text, side):
    assert predictor.predict(text)["input"]["boundary"] == side


def test_protected_contexts_and_named_undeciphered_unit(predictor):
    units = predictor.codec.mapping["unit_to_model_id"]
    lig = next(t for t in units if t.startswith("<LIG-U"))
    unk = next(t for t in predictor.codec.units if t.startswith("[UNK-U"))
    result = predictor.predict(f"王{unk}{lig}[MASK]□…<ANNOTATION_BOUNDARY>吉")
    assert result["input"]["ligature_unit_count"] == 1
    assert result["input"]["has_zhaoji"]
    assert lig in result["completed_text"] and unk in result["completed_text"]
    assert result["completed_text"].endswith("□…<ANNOTATION_BOUNDARY>吉")
    assert all(t in predictor.codec.units for row in result["top_k"] for t in row["answer_units"])


@pytest.mark.parametrize("text", ["[MASK]", "王<ANNOTATION_BOUNDARY>[MASK]", "王🐉[MASK]", "王" * 128 + "[MASK]"])
def test_frozen_codec_rules_still_enforced(predictor, text):
    with pytest.raises(ValueError):
        predictor.predict(text)


@pytest.mark.parametrize("k", [0, 11, -1, 1.5, True])
def test_top_k_range(cfg, k):
    with pytest.raises(ValueError):
        infer.decode_config(cfg, k)


def test_request_top3_does_not_change_search(predictor):
    a = predictor.predict("王[MASK][MASK]", 10)
    b = predictor.predict("王[MASK][MASK]", 3)
    assert b["top_k"] == a["top_k"][:3]


def test_current_v150_defaults():
    assert infer.DEFAULT_DATA == ROOT.parent / "shared/data/v1.5.0"
    assert infer.DEFAULT_CHECKPOINT == ROOT / "outputs/e1a_v150_s42/best.pt"


def test_probability_note_uses_actual_prediction_vocabulary(predictor):
    result = predictor.predict("王[MASK]")
    assert any(f"全部{len(predictor.codec.units)}个" in note for note in result["notes"])
    assert not any("3511" in note for note in result["notes"])


@pytest.mark.parametrize("text", ["□[MASK]", "[MASK]…", "□[MASK]<ANNOTATION_BOUNDARY>吉"])
def test_damage_and_annotation_do_not_satisfy_minimum_context(predictor, text):
    with pytest.raises(ValueError, match="正文文字单位"):
        predictor.predict(text)


def test_ligature_satisfies_minimum_context(predictor):
    lig = next(t for t in predictor.codec.mapping["unit_to_model_id"] if t.startswith("<LIG-U"))
    assert predictor.predict(lig + "[MASK]")["input"]["ligature_unit_count"] == 1


def test_unsupported_frozen_beam_rejected(cfg):
    with pytest.raises(ValueError):
        infer.decode_config(dict(cfg, beam_width=20), 10)


def test_non_finite_scores_rejected(predictor):
    q = predictor.codec.parse_text("王[MASK]")
    with pytest.raises(ValueError):
        infer.make_report("王[MASK]", "王[MASK]", q, predictor.codec.encode(q),
                          {"paths": [[0]], "scores": [np.nan]}, predictor.codec.units, 1, {})


def test_json_exclusive_create(tmp_path):
    path = tmp_path / "result.json"
    infer.write_json_new(path, {"answer": "王"})
    assert json.loads(path.read_text())["answer"] == "王"
    with pytest.raises(FileExistsError):
        infer.write_json_new(path, {"answer": "其"})
    assert json.loads(path.read_text())["answer"] == "王"


def test_runtime_gate_is_not_bypassed(monkeypatch, tmp_path):
    monkeypatch.setattr(engine, "load_checkpoint", lambda _: {"kind": "best"})
    def reject(*args):
        raise ValueError("Checkpoint/runtime mismatch")
    monkeypatch.setattr(engine, "inspect_checkpoint", reject)
    with pytest.raises(ValueError, match="runtime mismatch"):
        infer.E1Predictor(tmp_path / "best.pt")


def test_non_best_rejected_before_model_inspection(monkeypatch, tmp_path):
    monkeypatch.setattr(engine, "load_checkpoint", lambda _: {"kind": "training"})
    monkeypatch.setattr(engine, "inspect_checkpoint", lambda *_: pytest.fail("must not inspect last.pt"))
    with pytest.raises(ValueError, match="best.pt"):
        infer.E1Predictor(tmp_path / "last.pt")


def test_no_gold_reader_or_training_used(predictor, monkeypatch):
    monkeypatch.setattr(engine, "benchmark", lambda *_: pytest.fail("must not read benchmark answers"))
    monkeypatch.setattr(engine, "train", lambda *_: pytest.fail("must not train"))
    predictor.predict("王[MASK]")


def test_help_without_model_loading():
    r = subprocess.run([sys.executable, "-B", str(INFER_SCRIPT), "--help"],
                       capture_output=True, text=True, timeout=20)
    assert r.returncode == 0 and "--top-k" in r.stdout and "--checkpoint" in r.stdout


def test_cli_invalid_input_does_not_load_model(monkeypatch):
    monkeypatch.setattr(infer, "E1Predictor", lambda *_: pytest.fail("invalid syntax must fail before loading"))
    assert infer.main(["--text", "王其田"]) == 2


def test_cli_json_and_saved_output(monkeypatch, predictor, tmp_path, capsys):
    monkeypatch.setattr(infer, "E1Predictor", lambda *_: predictor)
    path = tmp_path / "prediction.json"
    assert infer.main(["--text", "王[MASK]", "--json", "--output", str(path)]) == 0
    value = json.loads(capsys.readouterr().out)
    assert value == json.loads(path.read_text()) and len(value["top_k"]) == 10


def test_interactive_bad_input_then_valid_query(monkeypatch, predictor, capsys):
    count = []
    def load(*_):
        count.append(1)
        return predictor
    lines = iter(["王其田", "王[MASK]", ":quit"])
    monkeypatch.setattr(infer, "E1Predictor", load)
    monkeypatch.setattr("builtins.input", lambda: next(lines))
    assert infer.main(["--interactive"]) == 0
    captured = capsys.readouterr()
    assert "输入错误" in captured.err and "模型答案：" in captured.out and len(count) == 1
