"""One execution policy for training, standalone evaluation and prediction."""
import os
import platform
import random

import numpy as np
import torch


def configure_execution(seed=None):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    if os.environ['CUBLAS_WORKSPACE_CONFIG'] not in (':4096:8', ':16:8'):
        raise ValueError('Use deterministic CUBLAS_WORKSPACE_CONFIG=:4096:8 or :16:8 before CUDA initialization')
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    return execution_policy()


def execution_policy():
    return {'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
            'cudnn_benchmark': torch.backends.cudnn.benchmark,
            'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32,
            'matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32,
            'cublas_workspace_config': os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
            'attention_implementation': 'eager'}


def runtime_versions():
    import transformers
    import tokenizers
    return {'python': platform.python_version(), 'torch': str(torch.__version__),
            'cuda_runtime': torch.version.cuda, 'transformers': transformers.__version__,
            'tokenizers': tokenizers.__version__, 'numpy': np.__version__}


def require_runtime(identity):
    actual = runtime_versions()
    if identity.get('runtime') != actual:
        raise ValueError('Checkpoint/runtime mismatch; restore the training environment. '
                         f'Expected {identity.get("runtime")!r}; actual {actual!r}')
    policy = configure_execution(identity['seed'])
    if identity.get('execution_policy') != policy:
        raise ValueError('Checkpoint/execution-policy mismatch; do not silently change numerical settings')
    return actual


def effective_precision(device, cfg):
    """CPU is FP32 diagnostic only; never silently downgrade CUDA precision."""
    device = torch.device(device)
    precision = cfg.get('precision')
    if precision not in {'fp32', 'fp16', 'bf16'}:
        raise ValueError('Unsupported precision: ' + repr(precision))
    if device.type not in {'cpu', 'cuda'}:
        raise ValueError('Only CPU diagnostics and CUDA execution are supported')
    return precision if device.type == 'cuda' else 'fp32'


def require_cuda_precision(device, cfg):
    device = torch.device(device)
    precision = effective_precision(device, cfg)
    if device.type != 'cuda' or not torch.cuda.is_available():
        raise RuntimeError('CUDA precision requires an available CUDA device')
    if device.index is not None and not 0 <= device.index < torch.cuda.device_count():
        raise RuntimeError(f'CUDA device index is unavailable: {device.index}')
    if precision == 'bf16':
        # Query the requested device, not whichever CUDA device was current.
        with torch.cuda.device(device):
            if not torch.cuda.is_bf16_supported(including_emulation=False):
                raise RuntimeError('Native CUDA BF16 is required; no FP16/FP32 fallback is permitted')
    return precision


def evaluation_environment(device, cfg):
    device = torch.device(device)
    hardware = {'type': device.type}
    if device.type == 'cuda':
        # A device index is intentionally not part of the compatibility key:
        # different seeds can run on different cards of the same hardware class.
        hardware.update(name=torch.cuda.get_device_name(device),
                        capability=list(torch.cuda.get_device_capability(device)))
    return {'versions': runtime_versions(), 'execution_policy': execution_policy(),
            'hardware': hardware,
            'effective_precision': effective_precision(device, cfg)}


def require_evaluation_report(report):
    env = report.get('evaluation_runtime')
    identity = report['identity']
    if not isinstance(env, dict) or not {'versions', 'execution_policy', 'hardware', 'effective_precision'} <= set(env):
        raise ValueError('Evaluation report lacks actual evaluation runtime; regenerate it with the frozen code')
    if env['versions'] != identity.get('runtime') or env['execution_policy'] != identity.get('execution_policy'):
        raise ValueError('Evaluation report runtime/policy differs from its training identity')
    if report['seed'] != identity['seed']:
        raise ValueError('Evaluation report seed differs from its training identity')
    expected_precision = effective_precision(env['hardware']['type'], identity['cfg'])
    if env['effective_precision'] != expected_precision:
        raise ValueError('Evaluation report precision is inconsistent with the device/config')
    return env
