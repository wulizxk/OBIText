import json
import re
from pathlib import Path
from benchmark_protocol_v140 import benchmark_path

from data_adapter_v116 import SpanSampler, encode_example
from .io import read, rows, sha


TOKENIZER_SOURCE_REPO = 'uer/gpt2-chinese-ancient'
TOKENIZER_SOURCE_REVISION = '3b264872995b09b5d9873e458f3d03a221c00669'


class Codec:
    def __init__(self, root, cfg):
        from transformers import AutoTokenizer
        self.root = Path(root)
        folder = self.root / 'tokenizers' / cfg['tokenizer_asset']
        self.mapping = read(folder / 'unit_mapping.json')
        self.tokenizer = AutoTokenizer.from_pretrained(folder, local_files_only=True, use_fast=True,
                                                       trust_remote_code=False)
        self.vocab_size = len(self.tokenizer)
        self.units = read(self.root / 'target_vocab.json')['tokens']
        self.target_ids = [self.mapping['target_unit_to_model_id'][t] for t in self.units]
        self.end_gap_id = self.mapping['control_to_model_id']['<END_GAP>']
        self.candidate_ids = self.target_ids + [self.end_gap_id]
        self.target_index = {t: i for i, t in enumerate(self.units)}
        self.native_to_candidate = {native: i for i, native in enumerate(self.candidate_ids)}
        self.cfg = cfg
        if len(self.units) != 4116 or len(set(self.target_ids)) != 4116:
            raise ValueError('D1-A requires exactly 4,116 distinct character targets')
        if self.end_gap_id in set(self.target_ids):
            raise ValueError('<END_GAP> must not be a character prediction target')
        if self.vocab_size != self.mapping['model_resize_required_to'] or self.vocab_size != 29999:
            raise ValueError('Unexpected frozen D1 tokenizer extension size')
        if self.mapping['original_vocab_size'] != 25370 or self.mapping['original_config_vocab_size'] != 25370:
            raise ValueError('Unexpected native tokenizer size')
        if (self.mapping['checkpoint_repo'], self.mapping['checkpoint_revision']) != (
                TOKENIZER_SOURCE_REPO, TOKENIZER_SOURCE_REVISION):
            raise ValueError('Unexpected provenance for the frozen D1 tokenizer asset')
        if self.mapping.get('architecture') != 'decoder':
            raise ValueError('D1 requires the frozen decoder tokenizer mapping')
        if sha(folder / 'vocab.txt') != cfg['native_vocab_sha256']:
            raise ValueError('GPT2-Chinese-Ancient native vocabulary checksum changed')
        if self.tokenizer.pad_token_id != self.mapping['pad_token_id']:
            raise ValueError('Native padding token ID changed')

    def encode(self, question, supervised=False):
        if not question['prefix_units'] and not question['suffix_units']:
            raise ValueError('At least one context unit is required')
        result = encode_example(question, 'decoder', self.tokenizer, self.mapping, supervised)
        if len(result['input_ids']) > self.cfg['max_input_length']:
            raise ValueError('Input exceeds limit; truncation is forbidden')
        controls = self.mapping['control_to_model_id']
        prompt_tail = [controls[f'<LEN_{question["gap_length"]}>'], controls['<MID>']]
        prompt_length = len(result['input_ids']) if not supervised else next(
            i for i, value in enumerate(result['labels']) if value != -100)
        if result['input_ids'][prompt_length - 2:prompt_length] != prompt_tail:
            raise ValueError('Malformed decoder FIM prompt')
        result['prompt_length'] = prompt_length
        result['gap_length'] = question['gap_length']
        if supervised:
            labels = [value for value in result['labels'] if value != -100]
            if len(labels) != question['gap_length'] + 1 or labels[-1] != self.end_gap_id:
                raise ValueError('Decoder supervision must be the answer followed by <END_GAP>')
            if any(value not in self.native_to_candidate for value in labels):
                raise ValueError('Decoder supervision escaped the shared character targets plus <END_GAP>')
        return result

    def collate(self, questions, device):
        import torch
        encoded = [self.encode(q, True) for q in questions]
        return training_batch(encoded, self.native_to_candidate, self.tokenizer.pad_token_id,
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


def training_batch(encoded, native_to_candidate, pad_id, multiple, device):
    import torch
    ids, attention = pad_inputs([q['input_ids'] for q in encoded], pad_id, multiple, device)
    owners, positions, targets, counts = [], [], [], []
    for i, q in enumerate(encoded):
        # In a causal LM, hidden state j-1 predicts the supervised token at j.
        selected = [(j - 1, x) for j, x in enumerate(q['labels']) if x != -100]
        if not selected:
            raise ValueError('Training sample has no supervised answer')
        if any(j < 0 for j, _ in selected):
            raise ValueError('A causal target cannot precede the FIM prompt')
        counts.append(len(selected))
        for j, native in selected:
            if native not in native_to_candidate:
                raise ValueError('Unexpected decoder target ID')
            owners.append(i); positions.append(j); targets.append(native_to_candidate[native])
    return {'input_ids': ids, 'attention_mask': attention,
            'owners': torch.tensor(owners, dtype=torch.long, device=device),
            'positions': torch.tensor(positions, dtype=torch.long, device=device),
            'targets': torch.tensor(targets, dtype=torch.long, device=device),
            'counts': torch.tensor(counts, dtype=torch.float32, device=device)}


def benchmark_public(root, suite):
    root = Path(root)
    public = rows(benchmark_path(root, suite, 'public'))
    ids = [q['example_id'] for q in public]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate public benchmark example IDs')
    return public


def benchmark_gold(root, suite, public):
    root = Path(root)
    answers = {g['example_id']: g['gold_units'] for g in rows(benchmark_path(root, suite, 'gold'))}
    if len(public) != len(answers) or {q['example_id'] for q in public} != set(answers):
        raise ValueError('Unpaired benchmark files')
    return answers


def benchmark(root, suite):
    """Compatibility helper for diagnostics; formal paths use the gold-late loaders."""
    public = benchmark_public(root, suite)
    return public, benchmark_gold(root, suite, public)
