# OBIText

Code release candidate for **OBIText: A Dataset and Benchmark for Contextual Text
Completion of Oracle Bone Inscriptions**.

This package contains 14 baseline/model implementations, frozen training
configurations, input adaptation, candidate generation, evaluation and
score-summary utilities. It uses the prepared task data and frozen test questions
supplied separately by the authors. Data preparation and test-set construction
are outside this code release.
It accompanies the frozen task resource. Historical directory names
containing `v140`, `v116` or `E7`–`E9` are retained because saved checkpoints bind
their source identities to those paths.

## License

The OBIText code in this repository is released under the [MIT License](LICENSE).
The data package is distributed separately by request under its own terms of use.

## Data access

Contact the author at **wulizxk@stu.aynu.edu.cn** to
request the OBIText data package and confirm its terms of use. The package
contains the full resource, the prepared task data, and the frozen
evaluation questions and answers.

Extract `OBIText_data/` next to the code repository. From the repository root,
set the data path in the shell you use to run the examples. If you store the
package elsewhere, set `OBITEXT_DATA` to the corresponding `v1.5.0/` directory.

```bash
OBITEXT_DATA="../OBIText_data/v1.5.0"
```

Keep the directory layout and `DATA_SHA256SUMS.json` supplied with the package.
The data identity check verifies the files used by training and validation.
The project-level `test` command reads the paired `public.jsonl` and
`gold.jsonl` files from these directories under the data root.

| Folder | Evaluation | Questions |
|---|---|---:|
| `evaluation/random/` | Random gaps | 7,675 |
| `evaluation/expert/` | Expert restorations | 1,599 |
| `evaluation/prefix/` | Sequence-initial single-unit gaps | 4,873 |
| `evaluation/suffix/` | Sequence-final single-unit gaps | 4,412 |
| `evaluation/all_position/` | All legal single-unit positions | 49,907 |

The standalone evaluator accepts the same frozen files and the
`evaluation/manifest.json` supplied with the data package.
`provenance/evaluation_suites.json` records the expected per-length counts and
question and answer hashes.

## Getting started

Use the recorded Python, PyTorch, CUDA and dependency versions in
`provenance/recorded_runtime.json` when evaluating existing checkpoints. The
original neural experiments used Python 3.11.16 and PyTorch 2.7.1+cu128.
Install PyTorch for the matching CUDA platform before the other neural dependencies.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-neural.txt
```

```bash
python scripts/verify_sources.py
python scripts/evaluate_questions.py --help
python scripts/summarize_scores.py --help
```

For SikuRoBERTa, verify the data, fetch the pinned pretrained weights, and
train the three seeds independently.

```bash
python protocol_v140_work/E1_sikuroberta/run.py verify --data "$OBITEXT_DATA"
python protocol_v140_work/E1_sikuroberta/run.py fetch-model
python protocol_v140_work/E1_sikuroberta/run.py train --data "$OBITEXT_DATA" --seed 42 --output results_local/E1_s42
python protocol_v140_work/E1_sikuroberta/run.py train --data "$OBITEXT_DATA" --seed 3407 --output results_local/E1_s3407
python protocol_v140_work/E1_sikuroberta/run.py train --data "$OBITEXT_DATA" --seed 2026 --output results_local/E1_s2026
```

With the five evaluation pairs installed under the data root, the project
`test` command runs them in a fixed order:

```bash
python protocol_v140_work/E1_sikuroberta/run.py test --data "$OBITEXT_DATA" --confirm-final \
  --checkpoint results_local/E1_s42/best.pt --output results_local/E1_s42_test
```

To evaluate one suite, pass its question and answer files and the shared
evaluation manifest. This example uses the random-gap suite.

```bash
python scripts/evaluate_questions.py --model E1 \
  --data "$OBITEXT_DATA" \
  --questions "$OBITEXT_DATA/evaluation/random/public.jsonl" \
  --gold "$OBITEXT_DATA/evaluation/random/gold.jsonl" \
  --manifest "$OBITEXT_DATA/evaluation/manifest.json" \
  --checkpoint results_local/E1_s42/best.pt --output results_local/E1_s42_random
```

For the deterministic baselines, run `fit`, `dev`, and `freeze` with the
B1/B2/B6 project CLI. Pass `--data "$OBITEXT_DATA"` before each subcommand.
Use the resulting fit and selection directories with `--fit` and `--selection`
in the standalone evaluator. Each project's `--help` describes any additional
checkpoint selection or training-completion arguments.

## Running the project tests

Some tests read the frozen task data at
`protocol_v140_work/shared/data/v1.5.0/`. To use the separate data package,
create a symbolic link at that location from the repository root.

```bash
mkdir -p protocol_v140_work/shared/data
ln -s "$(cd "$OBITEXT_DATA" && pwd -P)" protocol_v140_work/shared/data/v1.5.0
```

Run each test suite from its project directory with the Python environment
active. For example, the following command runs the SikuRoBERTa tests.

```bash
(cd protocol_v140_work/E1_sikuroberta && python -m pytest tests)
```

CUDA-specific tests require a compatible GPU.

## Scoring and aggregation

Random gaps contain 1,535 questions at each length from 1 through 5. Expert
restoration targets are complete annotated spans of length 1–5. The boundary
suites mask exactly one unit. Report each suite separately because the suites
can share source records and prediction positions.

The all-position evaluation supplies the 1–20, 21–99 and at least 100 training
frequency strata. Pass the frozen `target_vocab.json` with `--target-vocab` to
`scripts/summarize_scores.py` to compute these strata. The script accepts one
or several scored files for the same ordered questions and reference answers.

## Included models

| Code | Model | Project directory under `protocol_v140_work` |
|---|---|---|
| B1 | Unigram frequency | B1_B2_B6_baselines |
| B2 | Bidirectional 5-gram | B1_B2_B6_baselines |
| B6 | BM25 retrieval | B1_B2_B6_baselines |
| B3 | BiLSTM | B3_B4_neural_baselines |
| B4 | BERT, random initialization | B3_B4_neural_baselines |
| E1 | SikuRoBERTa | E1_sikuroberta |
| E2 | GuwenBERT-base | E2_guwenbert_base |
| E3 | SikuBERT | E3_sikubert |
| D1 | GPT2-Chinese-Ancient | D1_gpt2_chinese_ancient |
| D2 | Qwen2.5-0.5B | D2_qwen2_5_0_5b |
| D3 | Qwen3-0.6B-Base | D3_qwen3_0_6b_base |
| ED1 | Randeng-T5-77M | E7_randeng_t5_77m |
| ED2 | Mengzi-T5-base | E8_mengzi_t5_base |
| ED3 | Chinese BART-base | E9_bart_base_chinese |

The manuscript's fixed core comparison uses B2, B6, B4, E1, D1 and ED3. Neural
scores are averaged across three separate runs (seeds 42, 3407, 2026), not an
ensemble. Deterministic baselines have one run. All-position frequency strata
contain 1,051, 2,305 and 46,551 items respectively; zero-frequency targets have
no items in this frozen split.
