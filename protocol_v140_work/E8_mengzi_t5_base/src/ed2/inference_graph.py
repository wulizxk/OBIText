"""Exact-shape CUDA replay scoped to one immutable model evaluation.

No batch repadding, KV cache, precision change, beam change or global model
cache. Small/rare shapes stay eager. Each graph has its own memory pool;
returned outputs are cloned so later replays cannot overwrite live results.
"""
import time
import torch

BACKEND = 'cuda_graph_exact_v1'
MAX_ENTRIES = 24
CAPTURE_AFTER = 3


class EvaluationGraphSession:
    def __init__(self, model, cfg, device):
        backend = cfg.get('evaluation_backend', 'eager')
        if backend not in ('eager', BACKEND):
            raise ValueError('Unknown evaluation backend: ' + repr(backend))
        self.model, self.cfg = model, cfg
        self.device = torch.device(device)
        self.enabled = backend == BACKEND and self.device.type == 'cuda'
        self.cache, self.seen = {}, {}
        self.closed = False
        if self.enabled and model.training:
            raise RuntimeError('CUDA replay is evaluation-only; call model.eval() first')
        self.versions = self._versions() if self.enabled else ()
        self.stats = {'backend': backend, 'enabled': self.enabled,
                      'captures': 0, 'replays': 0, 'eager_calls': 0,
                      'capture_seconds': 0.0, 'max_entries': MAX_ENTRIES,
                      'capture_bitwise_checks_passed': 0}

    def _versions(self):
        return tuple((id(p), p._version) for p in self.model.parameters())

    def __enter__(self):
        if self.closed:
            raise RuntimeError('Cannot reuse a closed evaluation session')
        return self

    def __exit__(self, kind, value, traceback):
        try:
            if self.enabled:
                torch.cuda.synchronize(self.device)
                if kind is None and self._versions() != self.versions:
                    raise RuntimeError('Model parameters changed during graph evaluation')
        finally:
            self.cache.clear()
            self.closed = True

    def call(self, kind, fn, args):
        if self.closed:
            raise RuntimeError('Cannot reuse a closed evaluation session')
        if not self.enabled:
            return fn(*args)
        if self.model.training or torch.is_grad_enabled():
            raise RuntimeError('CUDA replay requires eval mode and disabled gradients')
        if not all(isinstance(x, torch.Tensor) and x.device.type == 'cuda'
                   and x.device == args[0].device and not x.requires_grad for x in args):
            raise ValueError('Graph inputs must be same-device CUDA inference tensors')
        # Preserve strides and batch shapes: do not silently convert a view to
        # a different matrix layout, which may select different GEMM kernels.
        if not all(x.is_contiguous() for x in args):
            self.stats['eager_calls'] += 1
            return fn(*args)
        key = (kind, tuple((tuple(x.shape), x.dtype, tuple(x.stride()), x.device.index) for x in args))
        self.seen[key] = self.seen.get(key, 0) + 1
        if key not in self.cache:
            if self.seen[key] < CAPTURE_AFTER or len(self.cache) >= MAX_ENTRIES:
                self.stats['eager_calls'] += 1
                return fn(*args)
            started = time.perf_counter()
            static = [x.detach().clone() for x in args]
            precision = self.cfg['precision']
            dtype = {'bf16': torch.bfloat16, 'fp32': torch.float32}[precision]

            def invoke():
                # Capture casts themselves; never retain an autocast weight
                # cache owned by a caller's short-lived context or another epoch.
                with torch.no_grad(), torch.autocast('cuda', dtype=dtype,
                                                     enabled=precision != 'fp32', cache_enabled=False):
                    return fn(*static)

            stream = torch.cuda.Stream(device=self.device)
            stream.wait_stream(torch.cuda.current_stream(self.device))
            with torch.cuda.stream(stream):
                for _ in range(2):
                    warm = invoke()
            torch.cuda.current_stream(self.device).wait_stream(stream)
            del warm
            reference = fn(*args)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                output = invoke()
            graph.replay()
            if not torch.equal(reference, output):
                raise RuntimeError('CUDA graph capture differs from eager output: ' + repr(key))
            self.cache[key] = graph, static, output
            self.stats['captures'] += 1
            self.stats['capture_bitwise_checks_passed'] += 1
            self.stats['capture_seconds'] += time.perf_counter() - started
        graph, static, output = self.cache[key]
        for target, source in zip(static, args):
            target.copy_(source)
        graph.replay()
        self.stats['replays'] += 1
        return output.clone()
