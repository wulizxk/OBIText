# OBIText data

Prepared public edition of the frozen OBIText v1.5.0 resource and contextual
completion benchmark. Files use UTF-8 encoding. JSONL files contain one JSON
record per line.

## Contents

Paths in the following table are relative to `v1.5.0/`.

| Path | Contents |
|---|---|
| `full_resource/normalized_resource_records.jsonl` | 81,941 resource records, with normalized units, source identifiers, textual status, glyph identity, ligature components, and restoration flags |
| `splits/train.jsonl` | 66,511 task sequences |
| `splits/validation.jsonl` | 4,689 task sequences |
| `splits/test.jsonl` | 7,675 task sequences |
| `group_assignments.json` | The 36,078 groups and their frozen split assignments |
| `input_units.json` | The 4,546 input units and structural markers |
| `target_vocab.json` | The 4,116 prediction targets and their training frequencies |
| `pools/train_span1.jsonl` through `train_span5.jsonl` | Legal starts used by the dynamic training sampler |
| `benchmark/` | Task protocol and 15,024 validation questions with reference answers |
| `evaluation/` | Frozen evaluation questions, reference answers, and their checksums |
| `tokenizers/*/unit_mapping.json` | Fixed model-specific unit IDs and control-token IDs |
| `tokenizer_sources.json` | Pinned upstream tokenizer revisions and file checksums for rebuilding the expanded tokenizers |
| `glyphs/undeciphered/` | 1,572 undeciphered-glyph PNG renderings and their identifier index |
| `DATA_SHA256SUMS.json` | Checksums for the public data edition |

The resource contains 81,726 aligned records and 215 records whose original
transcriptions could not be aligned. For the latter, `alignment_success` is
`false`, `model_tokens` is `null`, and `raw_transcription` preserves their
available text. The completion benchmark uses the separate task sequences in
`splits/`.

## Record fields

Resource records retain `sequence_id`, `source_plate_id`, `sentence_id`,
`alignment_success`, and `model_tokens`. Aligned records also contain `units`.
Each unit preserves its source `glyph`, normalized `token`, `status`, `role`,
`restored` flag, and `targetable` flag. Ligature units additionally preserve
`components`, the ordered component readings recorded in the original data.

Task records contain `sequence_id`, `group_id`, `split`, `model_tokens`,
`targetable_mask`, `visible_context_mask`, and `eligible_starts`. Their `units`
retain textual status, protection, restoration, and annotation information.
`annotation_regions` records each region's type and zero-based unit positions.
`sources` preserves source collection and plate identifiers together with
`full_resource_sequence_id` links to the full resource. Source variants remain
linked to their task sequence.

Question `start_unit` values and annotation positions are zero-based.
`prefix_units` and `suffix_units` supply the visible contexts, and `gap_length`
specifies the number of missing normalized units. Reference answers are in the
paired `gold.jsonl` files and are joined by `example_id`.

## Evaluation sets

| Folder under `evaluation/` | Items | Target construction |
|---|---:|---|
| `random/` | 7,675 | 1,535 questions at each gap length from 1 to 5 |
| `expert/` | 1,599 | Complete editorial-restoration spans of length 1 to 5 |
| `prefix/` | 4,873 | One unit at the sequence beginning |
| `suffix/` | 4,412 | One unit at the sequence end |
| `all_position/` | 49,907 | Every legal single-unit prediction position |

Frequency-stratified evaluation uses the all-position questions. Training
frequencies of 1–20, 21–99, and at least 100 occurrences yield 1,051, 2,305,
and 46,551 items respectively. Frequencies are counted at eligible single-unit
prediction positions in the training split. This frozen test split contains
zero items whose target has zero training occurrences.

## Using the data with the code

The `OBIText_data/` directory is included at the repository root. Run the
following from that root. Rebuilding tokenizers downloads the exact pinned upstream
files, adds the frozen units and control tokens, and verifies the resulting
IDs and checksums.

```bash
python -m pip install -r requirements.txt
python scripts/prepare_tokenizers.py --data OBIText_data/v1.5.0
```

Follow the accompanying code README to install the neural training dependencies
and run model verification, training, and evaluation.

The tokenizer-building command also accepts `--native-root` for an existing
local cache whose subdirectories use the asset names from
`tokenizer_sources.json`. SikuBERT uses the same tokenizer asset as SikuRoBERTa
(`E1`). Expanded unit mappings preserve the vocabulary size and ID assignments
required by the saved checkpoints, including registered auxiliary units.

This public edition removes repeated raw transcriptions, intermediate alignment
evidence, duplicated tokenizer assets, and processing-only fields. Model inputs,
target eligibility, split assignments, question order, and reference answers
retain their frozen values. The accompanying code accepts both the original
and public data manifests and preserves compatibility with the recorded
checkpoint identities.

## Undeciphered glyphs

An identifier such as `[UNK-U1048FF]` preserves a distinct source glyph.
`source_code_point` in `glyphs/undeciphered/index.csv` refers to the original
font encoding and supplies the code point to use for a matching rendering.
The index also reports resource occurrence and record counts and whether the
identifier belongs to the prediction vocabulary.

The directory contains 1,572 PNG renderings for inspecting the glyphs, with one
image per identifier. For example, `[UNK-U1048FF]` maps to `U1048FF.png` through
the index's `image_filename` field. Each image preserves the existing 80-by-80
pixel RGBA rendering from the Guolian oracle-bone font. Text-model training and
scoring use the normalized identifiers.

## Release terms

Copyright (c) 2026 OBIText contributors.

The OBIText data are licensed under Creative Commons Attribution-NonCommercial
4.0 International (CC BY-NC 4.0). You may share and adapt the licensed material
for noncommercial purposes, with appropriate attribution, a license link, and
an indication of changes. See [LICENSE](LICENSE) and the
[official license terms](https://creativecommons.org/licenses/by-nc/4.0/).

The license applies to the rights that the OBIText contributors are authorized
to grant in the compilation, normalization, annotations, and distributed data.
Source materials and the Guolian font retain the rights of their respective
holders. The source identifiers retained in the records support attribution
and traceability. The collected resources include OBIMD, Guoxue Dashi, the
Pre-Qin Oracle Bone, Bronze, and Bamboo and Wooden Slip Lexical Database, and
*Yinxu Jiaguwen Moshi Quanbian*.

The code is available under the MIT License at the repository root.
