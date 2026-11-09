"""Torch-independent data interface for the frozen OBIText experiment contract."""
import json
import random
from pathlib import Path


def read_rows(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def has_visible_main_context(prefix_units, suffix_units):
    """Return whether context retains a visible main-text unit under v1.5.0."""
    for token in list(prefix_units) + list(suffix_units):
        if token == "<ANNOTATION_BOUNDARY>":
            break
        if token not in {"[LACUNA_1]", "[LACUNA_N]", "[MASK]"}:
            return True
    return False


class SpanSampler:
    def __init__(self, data_dir, seed):
        root = Path(data_dir)
        self.rows = {row["sequence_id"]: row for row in read_rows(root / "splits/train.jsonl")}
        self.pools = {
            length: read_rows(root / f"pools/train_span{length}.jsonl")
            for length in range(1, 6)
        }
        if not all(self.pools.values()):
            raise ValueError("A training length has no legal spans")
        self.rng = random.Random(seed)

    def sample(self):
        length = self.rng.choice([1, 2, 3, 4, 5])
        item = self.rng.choice(self.pools[length])
        start = self.rng.choice(item["starts"])
        row = self.rows[item["source_id"]]
        if start not in row["eligible_starts"][str(length)]:
            raise ValueError("Training pool disagrees with the frozen row eligibility")
        if not all(unit["targetable"] for unit in row["units"][start : start + length]):
            raise ValueError("Training pool attempts to mask a protected position")
        prefix = row["model_tokens"][:start]
        suffix = row["model_tokens"][start + length :]
        remaining_visible = (row["visible_context_mask"][:start]
                             + row["visible_context_mask"][start + length:])
        if not any(remaining_visible) or not has_visible_main_context(prefix, suffix):
            raise ValueError("A sampled span must retain visible main-text context")
        return {
            "source_id": row["sequence_id"],
            "gap_length": length,
            "start_unit": start,
            "prefix_units": prefix,
            "suffix_units": suffix,
            "gold_units": row["model_tokens"][start : start + length],
        }


def encode_example(question, architecture, tokenizer, mapping, supervised=False):
    """Encode an internal or one-sided boundary gap without truncation.

    All source units have attention=1, including damage, ligatures and 兆记.
    A boundary question may have an empty prefix or suffix; both cannot be empty.
    """
    length = question["gap_length"]
    if length not in range(1, 6):
        raise ValueError("Gap length must be 1..5")
    if not has_visible_main_context(question["prefix_units"], question["suffix_units"]):
        raise ValueError("At least one visible main-text context unit is required")
    unit_ids = mapping["unit_to_model_id"]
    controls = mapping["control_to_model_id"]
    prefix = [unit_ids[token] for token in question["prefix_units"]]
    suffix = [unit_ids[token] for token in question["suffix_units"]]
    if supervised:
        target_ids = mapping["target_unit_to_model_id"]
        gold = question["gold_units"]
        if len(gold) != length:
            raise ValueError("Gold length mismatch")
        answer = [target_ids[token] for token in gold]
    if architecture == "encoder":
        mask = tokenizer.mask_token_id
        if mask is None:
            mask = controls["<OBI_MASK>"]
        content = prefix + [mask] * length + suffix
        inputs = tokenizer.build_inputs_with_special_tokens(content)
        probe = tokenizer.build_inputs_with_special_tokens([-index - 1 for index in range(len(content))])
        content_positions = [index for index, value in enumerate(probe) if value < 0]
        if len(content_positions) != len(content) or len(probe) != len(inputs):
            raise ValueError("Cannot locate wrapped source units")
        result = {"input_ids": inputs, "attention_mask": [1] * len(inputs)}
        if supervised:
            labels = [-100] * len(inputs)
            for index, target in zip(
                content_positions[len(prefix) : len(prefix) + length], answer
            ):
                labels[index] = target
            result["labels"] = labels
    elif architecture == "decoder":
        prompt = (
            [controls["<PRE>"]]
            + prefix
            + [controls["<SUF>"]]
            + suffix
            + [controls[f"<LEN_{length}>"] , controls["<MID>"]]
        )
        inputs = prompt
        result = {}
        if supervised:
            labels = answer + [controls["<END_GAP>"]]
            inputs = prompt + labels
            result["labels"] = [-100] * len(prompt) + labels
        result.update(input_ids=inputs, attention_mask=[1] * len(inputs))
    elif architecture == "encoder_decoder":
        content = [controls[f"<LEN_{length}>"]] + prefix + [controls["<GAP>"]] + suffix
        inputs = tokenizer.build_inputs_with_special_tokens(content)
        result = {"input_ids": inputs, "attention_mask": [1] * len(inputs)}
        if supervised:
            result["labels"] = answer + [controls["<END_GAP>"]]
    else:
        raise ValueError(architecture)
    return result
