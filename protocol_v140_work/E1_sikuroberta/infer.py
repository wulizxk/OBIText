#!/usr/bin/env python3
"""User-facing E1 inference; additive wrapper around the frozen E1 decoder.

This file is outside the training-source fingerprint. It does not alter the
training config, checkpoint, formal evaluator or decoder search procedure.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT.parent / "shared/data/v1.5.0"
DEFAULT_CHECKPOINT = ROOT / "outputs/e1a_v150_s42/best.pt"
sys.path[:0] = [str(ROOT / "src"), str(ROOT.parent / "shared")]
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def normalize_masks(text):
    if not isinstance(text, str) or not text:
        raise ValueError("请输入含 [MASK] 的规范化释文。")
    if any(c.isspace() for c in text):
        raise ValueError("输入不能含空格或换行；兆记请使用 <ANNOTATION_BOUNDARY> 标记，不要直接删除其边界。")
    value = re.sub(r"\[mask\]|<mask>", "[MASK]", text, flags=re.IGNORECASE)
    count = value.count("[MASK]")
    span = re.search(r"(?:\[MASK\])+", value)
    if not 1 <= count <= 5 or span is None or span.group().count("[MASK]") != count:
        raise ValueError("只支持一个连续的1–5字缺口，例如 [MASK] 或 [MASK][MASK]；□/…不是待预测位置。")
    return value


def decode_config(saved, top_k):
    if type(top_k) is not int or not 1 <= top_k <= 10:
        raise ValueError("top-k必须是1至10的整数。")
    if saved.get("beam_width") != 10 or saved.get("return_sequences") != 5:
        raise ValueError("要求原E1协议：beam_width=10、return_sequences=5。")
    # Only expose all ten ALREADY retained beams. Never widen/prune differently.
    # The first five candidates remain the original evaluator's output.
    return dict(saved, return_sequences=10)


def require_visible_context(question):
    """Require an actual main-text unit, not just damage or suffix annotations."""
    context = question["prefix_units"] + question["suffix_units"]
    if "<ANNOTATION_BOUNDARY>" in context:
        context = context[:context.index("<ANNOTATION_BOUNDARY>")]
    if not any(t not in {"[LACUNA_1]", "[LACUNA_N]"} for t in context):
        raise ValueError("遮盖后须至少保留一个正文文字单位；缺损符号和注记不计入可用语境。")


def make_report(text, normalized, question, encoded, prediction, units, top_k, metadata):
    length = question["gap_length"]
    paths, scores = prediction["paths"], prediction["scores"]
    if len(paths) != len(scores) or len(paths) < top_k:
        raise ValueError("解码器没有返回请求数量的候选。")
    candidates = []
    seen = set()
    for rank, (path, raw_score) in enumerate(zip(paths[:top_k], scores[:top_k]), 1):
        score = float(raw_score)
        if (len(path) != length or not math.isfinite(score) or score > 1e-5
                or any(type(index) is not int or not 0 <= index < len(units) for index in path)
                or tuple(path) in seen):
            raise ValueError("非法、重复或非有限的模型候选。")
        seen.add(tuple(path))
        tokens = [units[index] for index in path]
        answer = "".join(tokens)
        candidates.append({
            "rank": rank, "answer": answer, "answer_units": tokens,
            "unit_types": ["undeciphered_glyph" if t.startswith("[UNK-U") else "character" for t in tokens],
            "completed_text": normalized.replace("[MASK]" * length, answer, 1),
            "score": score, "mean_log_score": score / length,
            "probability": math.exp(score) if length == 1 else None})
    prefix, suffix = question["prefix_units"], question["suffix_units"]
    input_units = prefix + ["[MASK]"] * length + suffix
    boundary = "initial" if not prefix else "final" if not suffix else "internal"
    notes = ["答案为模型候选，不是经专家确认的正确答案；本输出不是正式Test指标。",
             "□表示单字缺失、…表示不定长缺失，均只保留为上下文；只补全显式[MASK]。"]
    if length == 1:
        notes.append(f"probability由全部{len(units)}个允许目标的softmax给出，未在Top-k内部重新归一化；不是正确率或校准置信度。")
    else:
        notes.append("多字Top-10是beam=10从左向右条件补全留下的候选，不是所有字符组合的精确全局Top-10。")
        notes.append("score为累计pseudo-log-probability，越大越优；不将其转换成整段答案置信度。")
    if metadata.get("effective_precision") == "fp32" and metadata.get("training_precision") == "fp16":
        notes.append("本次CPU FP32推理可能与正式GPU FP16结果有数值或排序差异；不混入正式评测。")
    if any(t.startswith("[UNK-U") for t in input_units) or any(
            t.startswith("[UNK-U") for c in candidates for t in c["answer_units"]):
        notes.append("[UNK-U…]是可区分的未释字形单元，不是缺失符号或原生[UNK]。")
    first = candidates[0]
    return {
        "schema": "e1_user_inference_v1", "kind": "user_inference_not_formal_evaluation",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "answer": first["answer"], "answer_units": first["answer_units"],
        "completed_text": first["completed_text"], "top_k": candidates,
        "input": {"original_text": text, "normalized_mask_text": normalized,
                  "units": input_units, "unit_count": len(input_units),
                  "gap_length": length, "boundary": boundary,
                  "mask_unit_positions_1based": list(range(len(prefix) + 1, len(prefix) + length + 1)),
                  "position_definition": "1-based input-unit index, not Unicode offset or WordPiece index",
                  "model_token_count_including_specials": len(encoded["input_ids"]),
                  "damage_unit_count": sum(t in {"[LACUNA_1]", "[LACUNA_N]"} for t in input_units),
                  "ligature_unit_count": sum(t.startswith("<LIG-") for t in input_units),
                  "has_zhaoji": "<ANNOTATION_BOUNDARY>" in input_units},
        "decoding": {"target_vocab_size": len(units), "beam_width": 10, "top_k": top_k,
                     "fill_order": "left_to_right", "formal_multi_return_sequences": 5,
                     "score_type": "restricted_log_probability" if length == 1 else "pseudo_log_probability",
                     "exact_global_top_k": length == 1,
                     "top_k_probability_mass": sum(c["probability"] for c in candidates) if length == 1 else None},
        "model": metadata, "notes": notes}


class E1Predictor:
    """Load one checkpoint once; reuse for multiple user queries, without gold."""
    def __init__(self, checkpoint=None, data=None, device="cpu", threads=1, project=ROOT):
        import torch
        from e1.engine import inspect_checkpoint, load_checkpoint
        from e1.runtime import evaluation_environment
        if type(threads) is not int or threads < 1:
            raise ValueError("threads必须是正整数。")
        torch.set_num_threads(threads)
        self.project = Path(project).resolve()
        self.checkpoint = Path(checkpoint or self.project / "outputs/e1a_v150_s42/best.pt").resolve()
        self.data = Path(data or self.project.parent / "shared/data/v1.5.0").resolve()
        self.device = torch.device(device)
        if self.device.type not in {"cpu", "cuda"}:
            raise ValueError("仅支持cpu或cuda:0等CUDA设备，不自动回退精度。")
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("请求CUDA但CUDA不可用；可显式改用--device cpu。")
        started = time.perf_counter()
        checkpoint = load_checkpoint(self.checkpoint)
        if checkpoint.get("kind") != "best":
            raise ValueError("请使用Dev选择的best.pt，不使用last.pt或smoke checkpoint。")
        checkpoint, self.cfg, self.codec, self.model = inspect_checkpoint(
            self.project, self.data, self.checkpoint, self.device, checkpoint)
        env = evaluation_environment(self.device, self.cfg)
        identity = checkpoint["identity"]
        self.metadata = {
            "checkpoint": str(self.checkpoint), "checkpoint_kind": checkpoint["kind"],
            "seed": identity["seed"], "best_epoch": checkpoint["epoch"],
            "model_repo": self.cfg["model_repo"], "model_revision": self.cfg["model_revision"],
            "dataset_version": self.cfg["dataset_version"],
            "training_precision": self.cfg["precision"],
            "effective_precision": env["effective_precision"], "device": str(self.device),
            "runtime": env, "frozen_code_sha256": identity["code_sha256"],
            "inference_wrapper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "data_sha256": identity["data_sha256"],
            "validation": "original_runtime_data_and_training_source_checks_passed"}
        self.load_seconds = time.perf_counter() - started

    def predict(self, text, top_k=10):
        from e1.decoding import predict_encoded_batch
        normalized = normalize_masks(text)
        cfg = decode_config(self.cfg, top_k)
        question = self.codec.parse_text(normalized)
        require_visible_context(question)
        encoded = self.codec.encode(question)
        started = time.perf_counter()
        prediction = predict_encoded_batch(self.model, self.codec, [encoded], cfg, self.device)[0]
        result = make_report(text, normalized, question, encoded, prediction,
                             self.codec.units, top_k, dict(self.metadata))
        result["timing_seconds"] = {"model_load": self.load_seconds,
                                    "prediction": time.perf_counter() - started}
        return result


def render_text(result):
    single = result["input"]["gap_length"] == 1
    model = result["model"]
    lines = [f"输入：{result['input']['original_text']}",
             f"模型答案：{result['answer']}", f"补全卜辞：{result['completed_text']}",
             f"缺口：{result['input']['gap_length']}个单元；位置：{result['input']['mask_unit_positions_1based']}",
             f"模型：SikuRoBERTa，数据={model.get('dataset_version')}，seed={model.get('seed')}，最佳epoch={model.get('best_epoch')}；{model.get('device')} / {model.get('effective_precision')}",
             f"Top-{len(result['top_k'])}（{'全部目标词表排名' if single else 'beam=10候选排名'}）：",
             "排名\t候选\t" + ("模型概率\tlog概率" if single else "累计pseudo-log评分")]
    for row in result["top_k"]:
        score = f"{row['probability']:.4%}\t{row['score']:.6f}" if single else f"{row['score']:.6f}"
        lines.append(f"{row['rank']}\t{row['answer']}\t{score}")
    lines.extend("说明：" + note for note in result["notes"])
    return "\n".join(lines)


def write_json_new(path, result):
    path = Path(path)
    payload = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    # Exclusive create: never overwrite a checkpoint, result or previous query.
    with path.open("x", encoding="utf-8") as stream:
        stream.write(payload)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="SikuRoBERTa / OBIText v1.5.0：输入含[MASK]的规范化释文，输出补全候选（默认CPU单线程）",
        epilog="连续缺失两个单位请写[MASK][MASK]；使用OBIText训练后的best.pt和对应冻结分词器，不使用原始预训练权重替代。")
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("--text", help="含一个连续1–5个[MASK]的规范化释文")
    inputs.add_argument("--interactive", action="store_true", help="只加载一次模型，连续输入；输入:quit退出")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT,
                        help="v1.5.0训练后的best.pt；默认outputs/e1a_v150_s42/best.pt")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA,
                        help="冻结数据目录；默认../shared/data/v1.5.0")
    parser.add_argument("--device", default="cpu", help="默认cpu；有资源时显式指定cuda:0")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--top-k", type=int, choices=range(1, 11), default=10)
    parser.add_argument("--json", action="store_true", help="标准输出使用JSON")
    parser.add_argument("--output", type=Path, help="另存单次查询的完整JSON；拒绝覆盖现有文件")
    args = parser.parse_args(argv)
    if args.interactive and args.output:
        parser.error("--output仅用于单次查询，不能与--interactive一起使用。")
    if args.threads < 1:
        parser.error("--threads必须大于0。")
    if args.output and (args.output.exists() or args.output.is_symlink()):
        parser.error("输出文件已存在，拒绝覆盖。")
    if args.device == "cpu":
        # Apply before importing torch; do not take GPU resources by default.
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ.setdefault("OMP_NUM_THREADS", str(args.threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(args.threads))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(args.threads))
    try:
        if not args.interactive:
            if args.text is None:
                print("请输入含[MASK]的卜辞：", file=sys.stderr)
                args.text = input()
            normalize_masks(args.text)
        print("正在加载SikuRoBERTa最佳检查点并校验运行环境、数据与训练源码……", file=sys.stderr)
        predictor = E1Predictor(args.checkpoint, args.data, args.device, args.threads)

        def emit(text):
            result = predictor.predict(text, args.top_k)
            if args.output:
                write_json_new(args.output, result)
            print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) if args.json else render_text(result))

        if not args.interactive:
            emit(args.text)
        else:
            while True:
                print("卜辞（:quit退出）：", file=sys.stderr)
                try:
                    text = input()
                except EOFError:
                    break
                if text == ":quit":
                    break
                try:
                    emit(text)
                except (ValueError, KeyError) as exc:
                    print("输入错误：" + str(exc), file=sys.stderr)
        return 0
    except (ValueError, KeyError, RuntimeError, OSError, EOFError) as exc:
        print("推理失败：" + str(exc), file=sys.stderr)
        return 2
    except ModuleNotFoundError as exc:
        print(f"推理失败：当前Python缺少依赖{exc.name!r}，请使用训练该检查点的Python环境。", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
