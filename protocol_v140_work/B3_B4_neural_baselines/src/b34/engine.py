import math
import os
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from benchmark_protocol_v140 import FORMAL_TEST_SUITES, compatible_source, evaluation_config

from .data import SpanSampler, benchmark, codec_for
from .decoding import predict_encoded_batch
from .io import digest, read, sha, source_fingerprint, verify_data, write, write_rows
from .metrics import aggregate, better, score_prediction
from .model import (
    autocast,
    build_model,
    model_from_spec,
    model_spec,
    optimizer_for,
    per_sample_losses,
)
from .run_state import (
    SCHEMA_VERSION,
    complete_run,
    reconcile_records,
    require_completed_run,
    stage_best,
    validate_history,
)
from .runtime import (
    configure_execution,
    evaluation_environment,
    require_runtime,
    runtime_versions,
)


def seed_all(seed):
    configure_execution(seed)


def validate_config(cfg):
    common = {
        "dataset_version": "v1.5.0",
        "initialization": "random",
        "dapt": False,
        "training_square_bracket_restorations_maskable": True,
        "training_boundary_masks_enabled": True,
        "protected_target_types": ["damage", "ligature", "annotation_suffix", "non_main"],
        "max_epochs": 50,
        "samples_per_epoch": 66511,
        "effective_batch_size": 64,
        "weight_decay": 0.01,
        "gradient_clip": 1.0,
        "warmup_epochs": 1,
        "schedule": "linear",
        "precision": "fp16",
        "max_input_length": 128,
        "pad_multiple": 8,
        "eval_question_batch_size": 128,
        "eval_forward_batch_size": 256,
        "beam_width": 10,
        "return_sequences": 5,
        "single_mask_return_sequences": 10,
        "fill_order": "left_to_right",
        "loss_reduction": "mean_of_per_sample_mean_mask_cross_entropy",
        "prediction_space": "shared_4116_targets",
        "checkpoint_selection": ["macro5_em1", "macro5_hit5", "earlier_epoch"],
        "validation_suite": "dev_full",
        "test_suites": list(FORMAL_TEST_SUITES),
        "early_stopping": True,
        "early_stopping_patience": 5,
        "seeds": [42, 3407, 2026],
    }
    for key, expected in common.items():
        if cfg.get(key) != expected:
            raise ValueError(f"Unsupported formal protocol change: {key}")
    if "epochs" in cfg:
        raise ValueError("Use max_epochs plus early stopping, not a fixed epochs field")
    if cfg["experiment"] == "B3":
        expected = {
            "model_family": "bilstm_mlm",
            "input_embedding_dim": 256,
            "lstm_hidden_size_per_direction": 256,
            "lstm_layers": 2,
            "dropout": 0.2,
            "learning_rate": 1e-3,
            "micro_batch_size": 64,
        }
    elif cfg["experiment"] == "B4":
        expected = {
            "model_family": "bert_mlm_random_init",
            "vocab_size": 33195,
            "hidden_size": 768,
            "num_hidden_layers": 12,
            "num_attention_heads": 12,
            "intermediate_size": 3072,
            "learning_rate": 5e-5,
            "micro_batch_size": 32,
        }
    else:
        raise ValueError("Only B3 and B4 are implemented by this project")
    for key, value in expected.items():
        if cfg.get(key) != value:
            raise ValueError(f"Frozen {cfg['experiment']} setting changed: {key}")


def rng_state(sampler):
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": [
            numpy_state[0],
            numpy_state[1].tolist(),
            numpy_state[2],
            numpy_state[3],
            numpy_state[4],
        ],
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "sampler": sampler.rng.getstate(),
    }


def restore_rng(state, sampler):
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state(
        (
            numpy_state[0],
            np.array(numpy_state[1], dtype=np.uint32),
            numpy_state[2],
            numpy_state[3],
            numpy_state[4],
        )
    )
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        if len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("Resume with the same CUDA_VISIBLE_DEVICES count")
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])
    sampler.rng.setstate(state["sampler"])


def save_checkpoint(path, state):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)


def load_checkpoint(path):
    return torch.load(path, map_location="cpu", weights_only=True)


def scheduler_for(optimizer, cfg):
    updates_per_epoch = math.ceil(cfg["samples_per_epoch"] / cfg["effective_batch_size"])
    total = cfg["max_epochs"] * updates_per_epoch
    warmup = round(cfg["warmup_epochs"] * updates_per_epoch)

    def scale(step):
        if step < warmup:
            return (step + 1) / warmup
        return max(0.0, (total - step) / (total - warmup))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, scale)


def update(model, codec, questions, cfg, device, optimizer, scaler, target_ids=None):
    model.train()
    if target_ids is None:
        target_ids = torch.tensor(codec.target_ids, dtype=torch.long, device=device)
    cpu_rng = torch.get_rng_state()
    gpu_rng = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
    for attempt in range(8):
        optimizer.zero_grad(set_to_none=True)
        loss_sum = torch.zeros((), dtype=torch.float32, device=device)
        for start in range(0, len(questions), cfg["micro_batch_size"]):
            batch = codec.collate(questions[start : start + cfg["micro_batch_size"]], device)
            with autocast(device, cfg):
                losses = per_sample_losses(model, cfg, batch, target_ids)
                loss = losses.sum() / len(questions)
            scaler.scale(loss).backward()
            loss_sum += losses.detach().float().sum()
        scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), cfg["gradient_clip"], error_if_nonfinite=False
        )
        if bool(torch.isfinite(loss_sum) & torch.isfinite(norm)):
            scaler.step(optimizer)
            scaler.update()
            return float(loss_sum), float(norm)
        if not bool(torch.isfinite(loss_sum)):
            raise FloatingPointError("Non-finite training loss; no checkpoint committed")
        if not scaler.is_enabled() or scaler.get_scale() <= 1:
            raise FloatingPointError("Non-finite unscaled gradients; no optimizer step committed")
        lower_scale = max(1.0, scaler.get_scale() * 0.5)
        scaler.update(new_scale=lower_scale)
        torch.set_rng_state(cpu_rng)
        if gpu_rng is not None:
            torch.cuda.set_rng_state(gpu_rng, device)
        print(f"AMP_RETRY attempt={attempt + 1} scale={lower_scale}", flush=True)
    raise FloatingPointError("AMP overflow persisted after 8 attempts")


def prediction_batches(public, encoded, cfg):
    batches = []
    for gap_length in sorted({question["gap_length"] for question in public}):
        indices = sorted(
            (index for index, question in enumerate(public) if question["gap_length"] == gap_length),
            key=lambda index: (len(encoded[index]["input_ids"]), index),
        )
        batches.extend(
            indices[start : start + cfg["eval_question_batch_size"]]
            for start in range(0, len(indices), cfg["eval_question_batch_size"])
        )
    return batches


def predict_questions(model, codec, public, cfg, device, encoded=None):
    model.eval()
    encoded = encoded if encoded is not None else [codec.encode(question) for question in public]
    if len(encoded) != len(public):
        raise ValueError("Encoded question count changed")
    values = [None] * len(public)
    target_ids = torch.tensor(codec.target_ids, dtype=torch.long, device=device)
    completed = 0
    for batch_index, indices in enumerate(prediction_batches(public, encoded, cfg)):
        predictions = predict_encoded_batch(
            model, codec, [encoded[index] for index in indices], cfg, device, target_ids
        )
        for index, prediction in zip(indices, predictions):
            values[index] = prediction
        completed += len(indices)
        if batch_index and batch_index % 100 == 0:
            print(f"EVAL {completed}/{len(public)}", flush=True)
    if any(value is None for value in values):
        raise RuntimeError("Evaluation failed to produce every prediction")
    return values


def score_questions(public, raw_predictions, answers, units):
    scored = [
        score_prediction(question, prediction, answers[question["example_id"]], units)
        for question, prediction in zip(public, raw_predictions)
    ]
    return aggregate(scored), scored


def evaluate(model, codec, public, answers, cfg, device, encoded=None):
    raw = predict_questions(model, codec, public, cfg, device, encoded)
    return score_questions(public, raw, answers, codec.units)


def identity(project, data, cfg, seed):
    return {
        "cfg": cfg,
        "seed": seed,
        "data_sha256": verify_data(data),
        "code_sha256": source_fingerprint(project),
        "runtime": runtime_versions(),
        "execution_policy": configure_execution(),
    }


def restore_model(checkpoint, device):
    model = model_from_spec(checkpoint["model_spec"])
    model.load_state_dict(checkpoint["model"], strict=True)
    if checkpoint["model_spec"]["model_family"] == "bert_mlm_random_init":
        model.tie_weights()
    return model.to(device)


def train(project, data, output, cfg, seed, device, resume=False):
    validate_config(cfg)
    output = Path(output)
    ident = identity(project, data, cfg, seed)
    if output.exists() and not resume:
        raise FileExistsError("Output exists; choose a new run or explicitly --resume")
    if resume and not (output / "last.pt").is_file():
        raise FileNotFoundError("No epoch-boundary last.pt to resume")
    output.mkdir(parents=True, exist_ok=True)
    import fcntl

    with (output / ".run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        _train_locked(project, data, output, cfg, seed, device, resume, ident)


def _train_locked(project, data, output, cfg, seed, device, resume, ident):
    seed_all(seed)
    codec = codec_for(data, cfg)
    sampler = SpanSampler(data, seed)
    public, answers = benchmark(data, "dev_full")
    encoded = [codec.encode(question) for question in public]
    counts = Counter(str(question["gap_length"]) for question in public)
    protocol = read(Path(data) / "benchmark/protocol.json")
    if dict(counts) != protocol["validation_length_counts"]:
        raise ValueError("Full-validation length counts changed")
    checkpoint = load_checkpoint(output / "last.pt") if resume else None
    if checkpoint and checkpoint["identity"] != ident:
        raise ValueError("Resume identity mismatch")
    if checkpoint:
        if checkpoint.get("schema_version") != SCHEMA_VERSION or checkpoint.get("kind") != "training":
            raise ValueError("Unsupported resume checkpoint")
        reconcile_records(
            output,
            ident,
            checkpoint["epoch"],
            checkpoint["step"],
            checkpoint["history"],
            checkpoint["best"],
        )
        model = restore_model(checkpoint, device)
        epoch, step = checkpoint["epoch"], checkpoint["step"]
        history, best = checkpoint["history"], checkpoint["best"]
    else:
        model = build_model(codec, cfg).to(device)
        epoch, step, history, best = 0, 0, [], None
    optimizer = optimizer_for(model, cfg)
    scheduler = scheduler_for(optimizer, cfg)
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=device.type == "cuda" and cfg["precision"] == "fp16",
        init_scale=1024.0,
    )
    if checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        restore_rng(checkpoint["rng"], sampler)
        del checkpoint
    write(
        output / "run.json",
        {
            "kind": "formal",
            "identity": ident,
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "trainable_parameter_count": sum(
                parameter.numel() for parameter in model.parameters() if parameter.requires_grad
            ),
            "initialization": "random",
            "pretrained_weights_loaded": False,
            "resume_granularity": "completed_epoch",
            "test_used_for_selection": False,
            "training_execution": evaluation_environment(device, cfg),
        },
    )

    def state():
        validate_history(cfg, epoch, step, history, best)
        return {
            "kind": "training",
            "schema_version": SCHEMA_VERSION,
            "identity": ident,
            "model_spec": model_spec(model, cfg),
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "rng": rng_state(sampler),
            "epoch": epoch,
            "step": step,
            "history": history,
            "best": best,
        }

    if not resume:
        save_checkpoint(output / "last.pt", state())
        reconcile_records(output, ident, epoch, step, history, best)
    if epoch == cfg["max_epochs"]:
        return complete_run(output, ident, epoch, step, history, best, "max_epochs")
    if epoch and epoch - best["epoch"] == cfg["early_stopping_patience"]:
        return complete_run(output, ident, epoch, step, history, best, "early_stopping")
    if epoch and epoch - best["epoch"] > cfg["early_stopping_patience"]:
        raise ValueError("Resume checkpoint exceeded early-stopping patience")

    target_ids = torch.tensor(codec.target_ids, dtype=torch.long, device=device)
    stopping_reason = None
    for next_epoch in range(epoch + 1, cfg["max_epochs"] + 1):
        started = time.time()
        loss_sum = 0.0
        length_counts = Counter()
        for offset in range(0, cfg["samples_per_epoch"], cfg["effective_batch_size"]):
            questions = [
                sampler.sample()
                for _ in range(
                    min(cfg["effective_batch_size"], cfg["samples_per_epoch"] - offset)
                )
            ]
            length_counts.update(question["gap_length"] for question in questions)
            loss, _ = update(
                model, codec, questions, cfg, device, optimizer, scaler, target_ids
            )
            loss_sum += loss
            step += 1
            scheduler.step()
            if step % 100 == 0:
                completed = offset + len(questions)
                print(
                    f"TRAIN epoch={next_epoch} samples={completed}/{cfg['samples_per_epoch']} "
                    f"loss={loss_sum/completed:.5f} lr={scheduler.get_last_lr()[0]:.8f}",
                    flush=True,
                )
        train_seconds = time.time() - started
        validation_started = time.time()
        metrics, predictions = evaluate(model, codec, public, answers, cfg, device, encoded)
        validation_seconds = time.time() - validation_started
        epoch = next_epoch
        improved = better(metrics, best["metrics"] if best else None)
        if improved:
            candidate = {
                "kind": "best",
                "schema_version": SCHEMA_VERSION,
                "identity": ident,
                "model_spec": model_spec(model, cfg),
                "model": model.state_dict(),
                "epoch": epoch,
                "metrics": metrics,
            }
            best = stage_best(output, candidate, predictions, save_checkpoint)
        history.append(
            {
                "epoch": epoch,
                "step": step,
                "train_loss": loss_sum / cfg["samples_per_epoch"],
                "sampled_lengths": dict(length_counts),
                "validation": metrics,
                "seconds": time.time() - started,
                "train_seconds": train_seconds,
                "validation_seconds": validation_seconds,
                "new_best": improved,
            }
        )
        save_checkpoint(output / "last.pt", state())
        reconcile_records(output, ident, epoch, step, history, best)
        print(
            f"EPOCH_DONE epoch={epoch} Macro5-EM@1={metrics['macro5_em1']:.6f} "
            f"Macro5-Hit@5={metrics['macro5_hit5']:.6f} best_epoch={best['epoch']}",
            flush=True,
        )
        if epoch < cfg["max_epochs"] and epoch - best["epoch"] == cfg["early_stopping_patience"]:
            stopping_reason = "early_stopping"
            break
    complete_run(output, ident, epoch, step, history, best, stopping_reason or "max_epochs")


def inspect_checkpoint(project, data, path, device, checkpoint=None):
    checkpoint = load_checkpoint(path) if checkpoint is None else checkpoint
    cfg = evaluation_config(checkpoint["identity"]["cfg"])
    validate_config(cfg)
    require_runtime(checkpoint["identity"])
    if checkpoint["identity"]["data_sha256"] != verify_data(data):
        raise ValueError("Checkpoint/data mismatch")
    if not compatible_source(project, checkpoint["identity"]["code_sha256"], source_fingerprint(project)):
        raise ValueError("Evaluation code differs from training code")
    codec = codec_for(data, cfg)
    model = restore_model(checkpoint, device)
    model.eval()
    return checkpoint, cfg, codec, model


def freeze_selection(project, data, runs, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError("Selection output already exists")
    if len(runs) != 3:
        raise ValueError("Freeze requires exactly three completed seed runs")
    records = []
    identities = []
    for run in map(Path, runs):
        checkpoint, marker = require_completed_run(run / "best.pt")
        cfg = evaluation_config(checkpoint["identity"]["cfg"])
        validate_config(cfg)
        if checkpoint["identity"]["data_sha256"] != verify_data(data):
            raise ValueError("Selection run uses different data")
        if not compatible_source(project, checkpoint["identity"]["code_sha256"], source_fingerprint(project)):
            raise ValueError("Selection run uses different code")
        identity_without_seed = {
            key: value for key, value in checkpoint["identity"].items() if key != "seed"
        }
        identities.append(identity_without_seed)
        records.append(
            {
                "seed": checkpoint["identity"]["seed"],
                "run": str(run.resolve()),
                "checkpoint": str((run / "best.pt").resolve()),
                "checkpoint_sha256": sha(run / "best.pt"),
                "completion_sha256": sha(run / "TRAINING_COMPLETE.json"),
                "best_epoch": checkpoint["epoch"],
                "dev_metrics": checkpoint["metrics"],
            }
        )
    if {record["seed"] for record in records} != {42, 3407, 2026}:
        raise ValueError("Selection requires seeds 42, 3407 and 2026")
    if any(identity != identities[0] for identity in identities):
        raise ValueError("Cannot freeze non-equivalent runs")
    output.mkdir(parents=True)
    payload = {
        "schema": 1,
        "status": "frozen_before_test",
        "experiment": identities[0]["cfg"]["experiment"],
        "identity_without_seed": identities[0],
        "records": sorted(records, key=lambda item: item["seed"]),
        "test_used_for_selection": False,
    }
    write(output / "selection.json", payload)
    write(
        output / "SELECTION_COMPLETE.json",
        {"status": "complete", "selection_sha256": sha(output / "selection.json")},
    )
    return payload


def require_selection(selection, checkpoint_path, checkpoint):
    selection = Path(selection)
    complete = read(selection / "SELECTION_COMPLETE.json")
    if complete.get("status") != "complete" or complete.get("selection_sha256") != sha(
        selection / "selection.json"
    ):
        raise ValueError("Invalid frozen selection receipt")
    payload = read(selection / "selection.json")
    if payload.get("status") != "frozen_before_test" or payload.get("test_used_for_selection") is not False:
        raise ValueError("Selection was not frozen before Test")
    identity_without_seed = {
        key: value for key, value in checkpoint["identity"].items() if key != "seed"
    }
    if payload.get("identity_without_seed") != identity_without_seed:
        raise ValueError("Selection/checkpoint identity mismatch")
    matching = [
        record for record in payload["records"] if record["seed"] == checkpoint["identity"]["seed"]
    ]
    if len(matching) != 1 or matching[0]["checkpoint_sha256"] != sha(checkpoint_path):
        raise ValueError("Checkpoint was not frozen in the three-seed selection")
    return sha(selection / "selection.json")


def evaluate_checkpoint(
    project, data, path, output, device, suites, selection=None, confirm_final=False
):
    output = Path(output)
    if output.exists():
        raise FileExistsError("Choose a new evaluation output directory")
    if not suites or len(suites) != len(set(suites)):
        raise ValueError("Empty or duplicate evaluation suites")
    is_test = any(suite.startswith("test") for suite in suites)
    allowed = {"dev_full", *FORMAL_TEST_SUITES}
    if not set(suites) <= allowed:
        raise ValueError("Unsupported evaluation suite")
    completion = None
    selection_hash = None
    if is_test:
        if not confirm_final or tuple(suites) != FORMAL_TEST_SUITES or selection is None:
            raise ValueError("Formal Test requires frozen three-seed selection and the current manuscript suites")
        import fcntl

        with (Path(path).parent / ".run.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
            checkpoint, completion = require_completed_run(path)
            selection_hash = require_selection(selection, path, checkpoint)
            return _evaluate_checkpoint(
                project,
                data,
                path,
                output,
                device,
                suites,
                checkpoint,
                completion,
                selection_hash,
            )
    return _evaluate_checkpoint(project, data, path, output, device, suites)


def _evaluate_checkpoint(
    project,
    data,
    path,
    output,
    device,
    suites,
    checkpoint=None,
    completion=None,
    selection_hash=None,
):
    checkpoint, cfg, codec, model = inspect_checkpoint(project, data, path, device, checkpoint)
    output.mkdir(parents=True)
    public_by_suite = {}
    raw_by_suite = {}
    # Critical Test boundary: finish every prediction before opening any gold file.
    for suite in suites:
        public, _ = benchmark(data, suite, include_answers=False)
        public_by_suite[suite] = public
        raw_by_suite[suite] = predict_questions(model, codec, public, cfg, device)
    results = {}
    for suite in suites:
        _, answers = benchmark(data, suite, include_answers=True)
        metrics, scored = score_questions(
            public_by_suite[suite], raw_by_suite[suite], answers, codec.units
        )
        results[suite] = metrics
        write_rows(output / f"{suite}_predictions.jsonl", scored)
    report = {
        "seed": checkpoint["identity"]["seed"],
        "epoch": checkpoint["epoch"],
        "identity": checkpoint["identity"],
        "checkpoint_sha256": sha(path),
        "suites": results,
        "evaluation_runtime": evaluation_environment(device, cfg),
        "completion_sha256": digest(completion) if completion else None,
        "selection_sha256": selection_hash,
        "all_predictions_completed_before_gold_open": True,
    }
    write(output / "metrics.json", report)
    print(results, flush=True)
    return report
