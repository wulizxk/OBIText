from pathlib import Path

import torch

from .data import SpanSampler, benchmark, codec_for
from .engine import (
    evaluate,
    identity,
    inspect_checkpoint,
    restore_rng,
    rng_state,
    save_checkpoint,
    seed_all,
    update,
)
from .io import write
from .model import build_model, model_spec, optimizer_for
from .runtime import evaluation_environment


def smoke(project, data, output, cfg, device):
    output = Path(output)
    if output.exists():
        raise FileExistsError("Smoke output exists")
    ident = identity(project, data, cfg, 42)
    output.mkdir(parents=True)
    seed_all(42)
    codec = codec_for(data, cfg)
    sampler = SpanSampler(data, 42)
    model = build_model(codec, cfg).to(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    optimizer = optimizer_for(model, cfg)
    scaler = torch.amp.GradScaler(
        "cuda", enabled=device.type == "cuda" and cfg["precision"] == "fp16", init_scale=1024.0
    )
    questions = [sampler.sample() for _ in range(cfg["effective_batch_size"])]
    tracked = next(model.parameters()).detach().clone()
    losses = []
    for _ in range(2):
        loss, _ = update(model, codec, questions, cfg, device, optimizer, scaler)
        losses.append(loss / len(questions))
    if torch.equal(tracked, next(model.parameters()).detach()):
        raise AssertionError("Model parameters did not update")
    public, answers = benchmark(data, "dev_full")
    chosen = [next(question for question in public if question["gap_length"] == length) for length in range(1, 6)]
    chosen_answers = {question["example_id"]: answers[question["example_id"]] for question in chosen}
    metrics, predictions = evaluate(model, codec, chosen, chosen_answers, cfg, device)
    state = rng_state(sampler)
    expected = [sampler.sample() for _ in range(8)]
    restore_rng(state, sampler)
    if expected != [sampler.sample() for _ in range(8)]:
        raise AssertionError("Sampler RNG restoration failed")
    checkpoint = {
        "kind": "smoke",
        "identity": ident,
        "model_spec": model_spec(model, cfg),
        "model": model.state_dict(),
        "epoch": 0,
    }
    save_checkpoint(output / "smoke.pt", checkpoint)
    _, _, restored_codec, restored_model = inspect_checkpoint(
        project, data, output / "smoke.pt", device
    )
    _, restored_predictions = evaluate(
        restored_model, restored_codec, chosen, chosen_answers, cfg, device
    )
    if predictions != restored_predictions:
        raise AssertionError("Checkpoint round trip changed predictions")
    report = {
        "kind": "smoke_only_not_paper_metrics",
        "status": "PASS",
        "experiment": cfg["experiment"],
        "optimizer_updates": 2,
        "losses": losses,
        "gap_lengths_checked": [1, 2, 3, 4, 5],
        "checkpoint_predictions_identical": True,
        "sampler_restore_identical": True,
        "test_gold_opened": False,
        "device": str(device),
        "peak_cuda_memory_mib": (
            torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None
        ),
        "identity": ident,
        "evaluation_runtime": evaluation_environment(device, cfg),
        "diagnostic_metrics": metrics,
    }
    write(output / "smoke_report.json", report)
    print(report, flush=True)
