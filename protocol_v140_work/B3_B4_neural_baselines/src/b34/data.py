import json
import re
from pathlib import Path
from benchmark_protocol_v140 import benchmark_path

from data_adapter_v116 import SpanSampler

from .io import read, rows


SPECIAL = {"PAD": 0, "MASK": 1, "BOS": 2, "EOS": 3}


def has_visible_main_context(question):
    for token in question["prefix_units"] + question["suffix_units"]:
        if token == "<ANNOTATION_BOUNDARY>":
            break
        if token not in {"[LACUNA_1]", "[LACUNA_N]", "[MASK]"}:
            return True
    return False


class CharacterCodec:
    """One atomic corpus unit per model input ID; no subword segmentation."""

    def __init__(self, root, cfg):
        self.root = Path(root)
        input_units = read(self.root / "input_units.json")["tokens"]
        self.units = read(self.root / "target_vocab.json")["tokens"]
        if len(input_units) != 4546 or len(self.units) != 4116:
            raise ValueError("Frozen v1.5.0 vocabulary sizes changed")
        if len(input_units) != len(set(input_units)) or len(self.units) != len(set(self.units)):
            raise ValueError("Duplicate units in frozen vocabulary")
        self.unit_to_id = {unit: index + len(SPECIAL) for index, unit in enumerate(input_units)}
        if not set(self.units) <= set(self.unit_to_id):
            raise ValueError("Prediction target is absent from the input vocabulary")
        self.target_index = {unit: index for index, unit in enumerate(self.units)}
        self.target_ids = [self.unit_to_id[unit] for unit in self.units]
        self.pad_id = SPECIAL["PAD"]
        self.mask_id = SPECIAL["MASK"]
        self.vocab_size = len(input_units) + len(SPECIAL)
        self.cfg = cfg

    def encode(self, question, supervised=False):
        if not has_visible_main_context(question):
            raise ValueError("At least one visible main-text context unit is required")
        gap_length = question["gap_length"]
        if not 1 <= gap_length <= 5:
            raise ValueError("Gap length must be 1..5")
        context = question["prefix_units"] + question["suffix_units"]
        missing = [unit for unit in context if unit not in self.unit_to_id]
        if missing:
            raise ValueError("Unknown normalized input unit: " + repr(missing[0]))
        prefix = [self.unit_to_id[unit] for unit in question["prefix_units"]]
        suffix = [self.unit_to_id[unit] for unit in question["suffix_units"]]
        positions = list(range(1 + len(prefix), 1 + len(prefix) + gap_length))
        input_ids = [SPECIAL["BOS"], *prefix, *([self.mask_id] * gap_length), *suffix, SPECIAL["EOS"]]
        if len(input_ids) > self.cfg["max_input_length"]:
            raise ValueError("Input exceeds limit; truncation is forbidden")
        result = {"input_ids": input_ids, "positions": positions}
        if supervised:
            gold = question.get("gold_units")
            if not isinstance(gold, list) or len(gold) != gap_length:
                raise ValueError("Supervised question lacks a complete gold span")
            try:
                result["targets"] = [self.target_index[unit] for unit in gold]
            except KeyError as error:
                raise ValueError("Gold unit is outside the prediction space") from error
        return result

    def collate(self, questions, device):
        encoded = [self.encode(question, supervised=True) for question in questions]
        return training_batch(encoded, self.pad_id, self.cfg["pad_multiple"], device)


class BertCodec:
    """Exact E1 tokenizer/mapping interface, but used without pretrained weights."""

    def __init__(self, root, cfg):
        from transformers import AutoTokenizer

        self.root = Path(root)
        folder = self.root / "tokenizers/E1"
        self.mapping = read(folder / "unit_mapping.json")
        self.tokenizer = AutoTokenizer.from_pretrained(
            folder, local_files_only=True, use_fast=True, trust_remote_code=False
        )
        self.units = read(self.root / "target_vocab.json")["tokens"]
        self.target_ids = [self.mapping["target_unit_to_model_id"][unit] for unit in self.units]
        self.target_index = {unit: index for index, unit in enumerate(self.units)}
        self.pad_id = self.tokenizer.pad_token_id
        self.mask_id = self.tokenizer.mask_token_id
        self.vocab_size = len(self.tokenizer)
        self.cfg = cfg
        if len(self.units) != 4116 or len(set(self.target_ids)) != 4116:
            raise ValueError("Frozen target mapping changed")
        if self.vocab_size != cfg["vocab_size"]:
            raise ValueError("B4 must use the exact E1 expanded tokenizer size")

    def encode(self, question, supervised=False):
        from data_adapter_v116 import encode_example

        if not has_visible_main_context(question):
            raise ValueError("At least one visible main-text context unit is required")
        encoded = encode_example(question, "encoder", self.tokenizer, self.mapping, supervised)
        if len(encoded["input_ids"]) > self.cfg["max_input_length"]:
            raise ValueError("Input exceeds limit; truncation is forbidden")
        positions = [index for index, value in enumerate(encoded["input_ids"]) if value == self.mask_id]
        if len(positions) != question["gap_length"]:
            raise ValueError("Unexpected mask token in context")
        result = {"input_ids": encoded["input_ids"], "positions": positions}
        if supervised:
            labels = [value for value in encoded["labels"] if value != -100]
            native_to_target = {native: index for index, native in enumerate(self.target_ids)}
            try:
                result["targets"] = [native_to_target[value] for value in labels]
            except KeyError as error:
                raise ValueError("Gold unit is outside the prediction space") from error
        return result

    def collate(self, questions, device):
        encoded = [self.encode(question, supervised=True) for question in questions]
        return training_batch(encoded, self.pad_id, self.cfg["pad_multiple"], device)


def codec_for(root, cfg):
    if cfg["model_family"] == "bilstm_mlm":
        return CharacterCodec(root, cfg)
    if cfg["model_family"] == "bert_mlm_random_init":
        return BertCodec(root, cfg)
    raise ValueError("Unknown model family")


def pad_inputs(values, pad_id, multiple, device):
    import torch

    if not values:
        raise ValueError("Cannot pad an empty batch")
    width = max(len(value) for value in values)
    width = (width + multiple - 1) // multiple * multiple
    ids = torch.tensor(
        [list(value) + [pad_id] * (width - len(value)) for value in values], dtype=torch.long
    )
    lengths = torch.tensor([len(value) for value in values], dtype=torch.long)
    attention = (torch.arange(width).unsqueeze(0) < lengths.unsqueeze(1)).long()
    return ids.to(device), attention.to(device)


def training_batch(encoded, pad_id, multiple, device):
    import torch

    ids, attention = pad_inputs([item["input_ids"] for item in encoded], pad_id, multiple, device)
    owners, positions, targets, counts = [], [], [], []
    for owner, item in enumerate(encoded):
        if len(item["positions"]) != len(item["targets"]):
            raise ValueError("Mask/target count mismatch")
        counts.append(len(item["targets"]))
        for position, target in zip(item["positions"], item["targets"]):
            owners.append(owner)
            positions.append(position)
            targets.append(target)
    return {
        "input_ids": ids,
        "attention_mask": attention,
        "owners": torch.tensor(owners, dtype=torch.long, device=device),
        "positions": torch.tensor(positions, dtype=torch.long, device=device),
        "targets": torch.tensor(targets, dtype=torch.long, device=device),
        "counts": torch.tensor(counts, dtype=torch.float32, device=device),
    }


def benchmark(root, suite, include_answers=True):
    root = Path(root)
    public = rows(benchmark_path(root, suite, 'public'))
    if not include_answers:
        return public, None
    answers = {
        item["example_id"]: item["gold_units"]
        for item in rows(benchmark_path(root, suite, 'gold'))
    }
    if len(public) != len(answers) or {item["example_id"] for item in public} != set(answers):
        raise ValueError("Unpaired benchmark files")
    return public, answers
