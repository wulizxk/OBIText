import json
import re
from pathlib import Path
from benchmark_protocol_v140 import benchmark_path

from data_adapter_v116 import SpanSampler, encode_example
from .io import read, rows, sha


TOKENIZER_SOURCE_REPO = 'SIKU-BERT/sikuroberta'
TOKENIZER_SOURCE_REVISION = 'bb25260d5c321924fe4fb353c09191c0aaf5c5c6'


class Codec:
    def __init__(self, root, cfg):
        from transformers import AutoTokenizer
        self.root = Path(root)
        # SIKU-BERT/sikubert and SIKU-BERT/sikuroberta publish the exact same
        # 29,791-line native vocabulary. Reuse the frozen E1 extension instead
        # of creating a second mapping for the same token IDs.
        folder = self.root / 'tokenizers' / cfg['tokenizer_asset']
        self.mapping = read(folder / 'unit_mapping.json')
        self.tokenizer = AutoTokenizer.from_pretrained(folder, local_files_only=True, use_fast=True,
                                                       trust_remote_code=False)
        self.vocab_size = len(self.tokenizer)
        self.units = read(self.root / 'target_vocab.json')['tokens']
        self.target_ids = [self.mapping['target_unit_to_model_id'][t] for t in self.units]
        self.target_index = {t: i for i, t in enumerate(self.units)}
        self.native_to_target = {native: i for i, native in enumerate(self.target_ids)}
        self.cfg = cfg
        if len(self.units) != 4116 or len(set(self.target_ids)) != 4116:
            raise ValueError('E3-A requires exactly 4,116 distinct prediction targets')
        if self.vocab_size != self.mapping['model_resize_required_to'] or self.vocab_size != 33195:
            raise ValueError('Unexpected frozen E1 tokenizer extension size')
        if self.mapping['original_vocab_size'] != 29791 or self.mapping['original_config_vocab_size'] != 29791:
            raise ValueError('Unexpected native tokenizer size')
        if (self.mapping['checkpoint_repo'], self.mapping['checkpoint_revision']) != (
                TOKENIZER_SOURCE_REPO, TOKENIZER_SOURCE_REVISION):
            raise ValueError('Unexpected provenance for the reused frozen E1 tokenizer asset')
        if sha(folder / 'vocab.txt') != cfg['native_vocab_sha256']:
            raise ValueError('SikuBERT native vocabulary differs from the frozen E1 tokenizer')
        if self.tokenizer.mask_token_id != self.mapping['native_mask_token_id']:
            raise ValueError('Native mask token ID changed')

    def encode(self, question, supervised=False):
        if not question['prefix_units'] and not question['suffix_units']:
            raise ValueError('At least one context unit is required')
        result = encode_example(question, 'encoder', self.tokenizer, self.mapping, supervised)
        if len(result['input_ids']) > self.cfg['max_input_length']:
            raise ValueError('Input exceeds limit; truncation is forbidden')
        positions = [i for i, x in enumerate(result['input_ids']) if x == self.tokenizer.mask_token_id]
        if len(positions) != question['gap_length']:
            raise ValueError('Unexpected mask token in context')
        result['positions'] = positions
        return result

    def collate(self, questions, device):
        import torch
        encoded = [self.encode(q, True) for q in questions]
        return training_batch(encoded, self.native_to_target, self.tokenizer.pad_token_id,
                              self.cfg['pad_multiple'], device)

    def parse_text(self, text):
        # This accepts already-normalized user input, not an alternative raw-data cleaner.
        text = text.replace('□', '[LACUNA_1]').replace('…', '[LACUNA_N]')
        pattern = r'\[MASK\]|\[UNK-U[0-9A-F]+\]|\[LACUNA_[1N]\]|<LIG-U[0-9A-F]+>|<ANNOTATION_BOUNDARY>|.'
        units = re.findall(pattern, text)
        if ''.join(units) != text:
            raise ValueError('Input contains unsupported whitespace/newlines')
        mask = [i for i, t in enumerate(units) if t == '[MASK]']
        if not mask or len(mask) > 5 or mask != list(range(mask[0], mask[-1] + 1)):
            raise ValueError('Provide one contiguous span of 1..5 [MASK] tokens')
        if '<ANNOTATION_BOUNDARY>' in units[:mask[0]]:
            raise ValueError('A 兆记 position cannot be a prediction target')
        for t in units:
            if t != '[MASK]' and t not in self.mapping['unit_to_model_id']:
                raise ValueError('Unsupported input unit; use normalized data: ' + repr(t))
        return {'example_id': 'USER', 'source_id': 'USER', 'group_id': 'USER',
                'gap_length': len(mask), 'prefix_units': units[:mask[0]], 'suffix_units': units[mask[-1] + 1:]}


def pad_inputs(values, pad_id, multiple, device):
    import torch
    if not values:
        raise ValueError('Cannot pad an empty batch')
    width = max(len(v) for v in values)
    width = (width + multiple - 1) // multiple * multiple
    # Construct the batch once on CPU, then transfer it once. Creating one CUDA
    # tensor per row is especially expensive during beam decoding, where this
    # function is called thousands of times per full-Dev evaluation.
    ids = torch.tensor([list(value) + [pad_id] * (width - len(value)) for value in values],
                       dtype=torch.long)
    lengths = torch.tensor([len(value) for value in values], dtype=torch.long)
    attention = (torch.arange(width).unsqueeze(0) < lengths.unsqueeze(1)).long()
    return ids.to(device), attention.to(device)


def training_batch(encoded, native_to_target, pad_id, multiple, device):
    import torch
    ids, attention = pad_inputs([q['input_ids'] for q in encoded], pad_id, multiple, device)
    owners, positions, targets, counts = [], [], [], []
    for i, q in enumerate(encoded):
        selected = [(j, x) for j, x in enumerate(q['labels']) if x != -100]
        if not selected:
            raise ValueError('Training sample has no supervised mask')
        counts.append(len(selected))
        for j, native in selected:
            owners.append(i); positions.append(j); targets.append(native_to_target[native])
    return {'input_ids': ids, 'attention_mask': attention,
            'owners': torch.tensor(owners, dtype=torch.long, device=device),
            'positions': torch.tensor(positions, dtype=torch.long, device=device),
            'targets': torch.tensor(targets, dtype=torch.long, device=device),
            'counts': torch.tensor(counts, dtype=torch.float32, device=device)}


def benchmark(root, suite):
    root = Path(root)
    public = rows(benchmark_path(root, suite, 'public'))
    answers = {g['example_id']: g['gold_units'] for g in rows(benchmark_path(root, suite, 'gold'))}
    if len(public) != len(answers) or {q['example_id'] for q in public} != set(answers):
        raise ValueError('Unpaired benchmark files')
    return public, answers
