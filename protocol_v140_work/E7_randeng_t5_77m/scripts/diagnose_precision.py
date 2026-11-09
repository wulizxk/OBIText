"""Compare forward/backward precision on one Train microbatch; never optimize.

This is a standalone diagnostic, not a change to the frozen FP16 protocol.
"""
import argparse
import json
import math
import os
from pathlib import Path
import sys

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT.parent / 'shared'))

import torch
from ed1.data import Codec, SpanSampler
from ed1.engine import seed_all
from ed1.io import read, write, source_fingerprint
from ed1.model import load_pretrained, per_sample_losses
from ed1.runtime import runtime_versions


def safe_max(tensor):
    if tensor is None:
        return None
    value = float(tensor.detach().abs().max())
    return value if math.isfinite(value) else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Diagnostic report already exists')
    cfg = read(ROOT / 'configs/ed1a.json')
    seed_all(42)
    codec = Codec(ROOT.parent / 'shared/data/v1.5.0', cfg)
    sampler = SpanSampler(ROOT.parent / 'shared/data/v1.5.0', 42)
    model, _ = load_pretrained(ROOT.parent / 'shared/pretrained/ED1', codec, cfg)
    model.to('cuda').train()
    questions = [sampler.sample() for _ in range(64)]
    batch = codec.collate(questions[:32], torch.device('cuda'))
    candidates = torch.tensor(codec.candidate_ids, device='cuda')
    cpu_state, gpu_state = torch.get_rng_state(), torch.cuda.get_rng_state()
    reports = {}
    monitored_name = 'decoder.block.1.layer.2.DenseReluDense.wo'
    for name, dtype in [('fp16', torch.float16), ('fp32', torch.float32), ('bf16', torch.bfloat16)]:
        torch.set_rng_state(cpu_state)
        torch.cuda.set_rng_state(gpu_state)
        model.zero_grad(set_to_none=True)
        bad, monitored = [], {}

        def make_hook(label):
            def hook(module, inputs, output):
                if not torch.is_tensor(output) or not output.is_floating_point():
                    return
                if len(bad) >= 8 and label != monitored_name:
                    return
                finite = bool(torch.isfinite(output).all())
                if not finite or label == monitored_name:
                    first = inputs[0] if inputs and torch.is_tensor(inputs[0]) else None
                    record = {'module': label, 'dtype': str(output.dtype), 'shape': list(output.shape),
                              'nonfinite_count': int((~torch.isfinite(output)).sum()),
                              'input_max_abs': safe_max(first), 'output_max_abs': safe_max(output)}
                    if not finite:
                        bad.append(record)
                    if label == monitored_name:
                        monitored.update(record)
            return hook

        hooks = [module.register_forward_hook(make_hook(label)) for label, module in model.named_modules()
                 if isinstance(module, torch.nn.Linear) or module.__class__.__name__ == 'MT5LayerNorm']
        try:
            with torch.autocast('cuda', dtype=dtype, enabled=name != 'fp32'):
                losses = per_sample_losses(model, batch, candidates)
                loss = losses.mean()
            finite = bool(torch.isfinite(losses).all())
            result = {'all_sample_losses_finite': finite, 'mean_loss': float(loss) if finite else None,
                      'first_nonfinite_modules': bad, 'monitored_layer': monitored}
            if finite:
                loss.backward()
                invalid = [label for label, parameter in model.named_parameters()
                           if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all())]
                result['all_gradients_finite'] = not invalid
                result['invalid_gradient_parameters'] = invalid[:8]
            reports[name] = result
        finally:
            for hook in hooks:
                hook.remove()
        print(name, json.dumps(result, allow_nan=False), flush=True)
    write(args.output, {'kind': 'precision_diagnostic_only', 'optimizer_updates': 0,
                        'formal_protocol_changed': False, 'test_used': False, 'seed': 42,
                        'same_train_microbatch': 32, 'source_sha256': source_fingerprint(ROOT),
                        'runtime': runtime_versions(), 'bf16_supported': torch.cuda.is_bf16_supported(),
                        'results': reports})


if __name__ == '__main__':
    main()
