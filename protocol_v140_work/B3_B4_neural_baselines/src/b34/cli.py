import argparse
import json
import statistics
from pathlib import Path

from benchmark_protocol_v140 import validate_formal_suite_metrics

from .io import read, verify_data, write


def summarize(paths, output):
    from .runtime import require_evaluation_report

    reports = [read(Path(path) / "metrics.json") for path in paths]
    if len(reports) != 3 or {report["seed"] for report in reports} != {42, 3407, 2026}:
        raise ValueError("Formal summary requires exactly seeds 42, 3407 and 2026")
    environments = [require_evaluation_report(report) for report in reports]
    if any(environment != environments[0] for environment in environments):
        raise ValueError("Evaluation environments differ")
    identities = [
        {key: value for key, value in report["identity"].items() if key != "seed"}
        for report in reports
    ]
    if any(identity != identities[0] for identity in identities):
        raise ValueError("Cannot aggregate different experiment identities")
    for report in reports:
        validate_formal_suite_metrics(report.get("suites", {}))
        if not report.get("selection_sha256"):
            raise ValueError("Formal Test report lacks a frozen selection receipt")
    if len({report["selection_sha256"] for report in reports}) != 1:
        raise ValueError("The three Test reports were not authorized by one frozen selection")

    def flatten(value, prefix=""):
        result = {}
        for key, item in value.items():
            name = prefix + "/" + key
            if isinstance(item, dict):
                result.update(flatten(item, name))
            elif isinstance(item, (float, int)) and not isinstance(item, bool):
                result[name] = item
        return result

    flat = [flatten(report["suites"]) for report in reports]
    if any(set(item) != set(flat[0]) for item in flat):
        raise ValueError("Evaluated metric sets differ")
    summary = {
        "seeds": sorted(report["seed"] for report in reports),
        "n": 3,
        "formal_three_seed_complete": True,
        "metrics": {
            key: {
                "mean": statistics.mean(item[key] for item in flat),
                "sample_std": statistics.stdev(item[key] for item in flat),
            }
            for key in flat[0]
        },
        "identity_without_seed": identities[0],
        "evaluation_runtime": environments[0],
    }
    if Path(output).exists():
        raise FileExistsError("Summary output exists")
    write(output, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main(project):
    parser = argparse.ArgumentParser(description="B3/B4 non-pretrained neural restoration baselines")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", type=Path, required=True)
    common.add_argument(
        "--data", type=Path, default=project.parent / "shared/data/v1.5.0"
    )
    common.add_argument("--device", default="cuda:0")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("verify", parents=[common])
    train_parser = subparsers.add_parser("train", parents=[common])
    train_parser.add_argument("--seed", type=int, required=True, choices=[42, 3407, 2026])
    train_parser.add_argument("--output", type=Path, required=True)
    train_parser.add_argument("--resume", action="store_true")
    smoke_parser = subparsers.add_parser("smoke", parents=[common])
    smoke_parser.add_argument("--output", type=Path, required=True)
    validate_parser = subparsers.add_parser("validate", parents=[common])
    validate_parser.add_argument("--checkpoint", type=Path, required=True)
    validate_parser.add_argument("--output", type=Path, required=True)
    freeze_parser = subparsers.add_parser("freeze", parents=[common])
    freeze_parser.add_argument("--runs", nargs=3, required=True)
    freeze_parser.add_argument("--output", type=Path, required=True)
    test_parser = subparsers.add_parser("test", parents=[common])
    test_parser.add_argument("--checkpoint", type=Path, required=True)
    test_parser.add_argument("--selection", type=Path, required=True)
    test_parser.add_argument("--output", type=Path, required=True)
    test_parser.add_argument("--confirm-final", action="store_true")
    summary_parser = subparsers.add_parser("summarize")
    summary_parser.add_argument("--evaluations", nargs=3, required=True)
    summary_parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "summarize":
        return summarize(args.evaluations, args.output)
    cfg = read(args.config)
    from .engine import (
        evaluate_checkpoint,
        freeze_selection,
        train,
        validate_config,
    )

    validate_config(cfg)
    if args.command == "verify":
        from .data import codec_for

        checksum = verify_data(args.data)
        codec = codec_for(args.data, cfg)
        print(
            json.dumps(
                {
                    "status": "PASS",
                    "experiment": cfg["experiment"],
                    "data_manifest_sha256": checksum,
                    "input_vocab_size": codec.vocab_size,
                    "prediction_targets": len(codec.units),
                    "initialization": "random",
                    "pretrained_weights_loaded": False,
                    "dev_questions": 15024,
                },
                indent=2,
            )
        )
        return
    import torch

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    if args.command == "train":
        return train(project, args.data, args.output, cfg, args.seed, device, args.resume)
    if args.command == "smoke":
        from .smoke import smoke

        return smoke(project, args.data, args.output, cfg, device)
    if args.command == "validate":
        return evaluate_checkpoint(
            project, args.data, args.checkpoint, args.output, device, ["dev_full"]
        )
    if args.command == "freeze":
        return freeze_selection(project, args.data, args.runs, args.output)
    if args.command == "test":
        return evaluate_checkpoint(
            project,
            args.data,
            args.checkpoint,
            args.output,
            device,
            cfg["test_suites"],
            selection=args.selection,
            confirm_final=args.confirm_final,
        )
