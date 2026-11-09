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
            'effective_precision': 'fp16' if device.type == 'cuda' and cfg['precision'] == 'fp16' else 'fp32'}


def require_evaluation_report(report):
    env = report.get('evaluation_runtime')
    identity = report['identity']
    if not isinstance(env, dict) or not {'versions', 'execution_policy', 'hardware', 'effective_precision'} <= set(env):
        raise ValueError('Evaluation report lacks actual evaluation runtime; regenerate it with the frozen code')
    if env['versions'] != identity.get('runtime') or env['execution_policy'] != identity.get('execution_policy'):
        raise ValueError('Evaluation report runtime/policy differs from its training identity')
    if report['seed'] != identity['seed']:
        raise ValueError('Evaluation report seed differs from its training identity')
    expected_precision = 'fp16' if env['hardware']['type'] == 'cuda' and identity['cfg']['precision'] == 'fp16' else 'fp32'
    if env['effective_precision'] != expected_precision:
        raise ValueError('Evaluation report precision is inconsistent with the device/config')
    return env
