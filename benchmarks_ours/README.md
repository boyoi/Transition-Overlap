# Benchmarks

This directory contains the benchmark loaders and the end-to-end evaluation pipeline included in the anonymous release.

The public release keeps only the components required for the end-to-end experiments:

- dataset loaders in `data_sets/`
- shared evaluation helpers in `evals/utils.py`
- the end-to-end pipeline in `evals/e2e/`

## Supported Datasets

The end-to-end evaluation currently supports:

- `2wikimqa`
- `musique`
- `samsum`
- `multi_news`
- `hotpotqa`
- `needle`

LongBench-based datasets are loaded from the public Hugging Face dataset `THUDM/LongBench`.

## Install

From the repository root:

```bash
pip install -e .
pip install -r benchmarks_ours/requirements.txt
```

## Run

The main entry point is:

```bash
bash benchmarks_ours/evals/e2e/run_test_v2.sh
```

Example:

```bash
MODEL=Qwen/Qwen3-4B \
DATASET=2wikimqa \
APPROACH=SSO \
CUDA_VISIBLE_DEVICES=0 \
bash benchmarks_ours/evals/e2e/run_test_v2.sh
```

Outputs are written to `benchmarks_ours/evals/e2e/results/`.
