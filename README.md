## Anonymous Release

This repository is a minimal anonymous code release for the long-context cache-reuse experiments in the submission.

The release contains:

- the modified `vllm/` source tree used by the experiments
- the required `csrc/`, `cmake/`, and packaging files for building from source
- the benchmark dataset loaders in `benchmarks_ours/data_sets/`
- the end-to-end evaluation pipeline in `benchmarks_ours/evals/e2e/`

The release intentionally excludes:

- git history and remotes
- CI configuration, docs, examples, and tests
- local editor/assistant configs
- build artifacts, compiled shared objects, and cached bytecode
- private absolute paths and machine-specific defaults
- previously generated evaluation outputs
- local copies of large benchmark archives when they can be fetched from public sources

## Setup

Create a Python environment with the dependencies required by your platform, then install this repository from source.

```bash
pip install -r requirements/cuda.txt
pip install -e .
pip install -r benchmarks_ours/requirements.txt
```

Depending on your system, you may need additional CUDA, compiler, or FlashAttention-related prerequisites that are normally required by vLLM source builds.

## Data

The end-to-end benchmark currently uses the following datasets:

- `2wikimqa`
- `musique`
- `samsum`
- `multi_news`
- `hotpotqa`
- `needle`

The LongBench-based datasets are loaded from the public Hugging Face dataset `THUDM/LongBench`. The `needle` benchmark uses the bundled `PaulGrahamEssays` text files in this repository.

## Main Entry Point

The main experiment entry point is:

```bash
bash benchmarks_ours/evals/e2e/run_test_v2.sh
```

Useful environment variables:

- `MODEL`: Hugging Face model ID or local model path
- `DATASET`: one of the datasets listed above
- `APPROACH`: one of `fr`, `naive`, `cacheblend-*`, `epic-*`, `CSO`, `SSO`
- `CUDA_VISIBLE_DEVICES`: GPU selection
- `TOT_NUM_DATA`: number of evaluation examples
- `VLLM_MAX_MODEL_LEN`
- `VLLM_TENSOR_PARALLEL_SIZE`

Example:

```bash
MODEL=Qwen/Qwen3-4B \
DATASET=2wikimqa \
APPROACH=SSO \
CUDA_VISIBLE_DEVICES=0 \
bash benchmarks_ours/evals/e2e/run_test_v2.sh
```

Results are written to:

```bash
benchmarks_ours/evals/e2e/results/
```
