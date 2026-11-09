# E8_mengzi_t5_base

This project implements the experiment identified in the repository-level
model table. See the root README for the recorded runtime, data layout and
training and evaluation commands.

Run `python run.py --help` for the project CLI. The `test` command evaluates
random gaps, expert restorations, initial gaps, final gaps and all legal
single-unit positions using the supplied frozen question files. Use
`scripts/evaluate_questions.py` from the repository root to evaluate an
individual manuscript suite.
