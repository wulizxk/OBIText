"""Evaluation-local FP32 output-row reuse; no cache survives an evaluation call."""
import torch


class EvaluationProjectionCache:
    def __init__(self, model):
        self.model = model
        self.weight = self.bias = None
        self.versions = None
        self.entries = {}
        self.hits = self.misses = self.peak_bytes = 0
        self.closed = False

    def __enter__(self):
        if self.closed:
            raise ValueError('Cannot reopen an evaluation projection cache')
        return self

    def __exit__(self, *_):
        self.entries.clear()
        self.model = self.weight = self.bias = None
        self.closed = True

    def select(self, model, candidate_ids):
        if self.closed or model is not self.model:
            raise ValueError('Projection cache belongs to a different/finished evaluation')
        if model.training or torch.is_grad_enabled():
            raise ValueError('Projection cache is evaluation/no-grad only')
        weight, bias = model.get_output_embeddings().weight, model.final_logits_bias
        if weight.dtype != torch.float32 or bias.dtype != torch.float32:
            raise ValueError('Cached output masters/bias must remain FP32')
        if (candidate_ids.ndim != 1 or candidate_ids.dtype != torch.long
                or candidate_ids.device != weight.device):
            raise ValueError('Invalid evaluation projection indices')
        versions = (weight._version, bias._version)
        if self.weight is None:
            self.weight, self.bias, self.versions = weight, bias, versions
        elif weight is not self.weight or bias is not self.bias or versions != self.versions:
            raise ValueError('Output weights changed during a cached evaluation')
        key = id(candidate_ids)
        if key in self.entries:
            ids, version, selected_weight, selected_bias = self.entries[key]
            if ids is not candidate_ids or version != candidate_ids._version:
                raise ValueError('Candidate indices changed during a cached evaluation')
            self.hits += 1
            return selected_weight, selected_bias
        # Exact same index_select and contiguous FP32 values as project_hidden;
        # only their lifetime changes. Never fuse bias or alter GEMM dimensions.
        selected_weight = weight.detach().index_select(0, candidate_ids)
        selected_bias = bias.detach()[0].index_select(0, candidate_ids)
        self.entries[key] = (candidate_ids, candidate_ids._version, selected_weight, selected_bias)
        self.misses += 1
        self.peak_bytes += (selected_weight.numel() * selected_weight.element_size()
                            + selected_bias.numel() * selected_bias.element_size())
        return selected_weight, selected_bias

    def report(self):
        return {'enabled': True, 'hits': self.hits, 'misses': self.misses,
                'peak_bytes': self.peak_bytes, 'closed': self.closed,
                'scope': 'one_evaluation_no_grad_FP32_rows_same_GEMM_and_bias_path'}
