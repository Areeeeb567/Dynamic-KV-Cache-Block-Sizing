# Dynamic KV Cache Block Sizing - Reproduction of vLLM PagedAttention

## Overview

This repository contains the reproducibility materials for the Base Article Selection deliverable (October 7, 2026) for Group G29. The base paper is "Efficient Memory Management for Large Language Model Serving with PagedAttention" by Kwon et al., SOSP 2023.

The reproduction runs vLLM with different KV cache block sizes on a Tesla T4 GPU to verify the reported throughput trends and to document hardware limitations that affect reproducibility.

## Repository Structure

```
reproducibility/
├── main.ipynb
├── README.md
├── data/
│   └── download_instructions.md
├── results/
│   ├── vllm_commit.txt
│   ├── vllm_version.txt
│   ├── requirements.txt
│   ├── nvidia_smi.txt
│   ├── throughput_block8.log
│   ├── throughput_block16.log
│   ├── throughput_block32.log
│   ├── azure_trace_summary.txt
│   └── azure_trace_with_classes.csv
└── report/
    └── BaseArticleSelection.pdf
```

All code is contained in `main.ipynb`. The notebook is organised into four sections: Setup, Throughput Benchmark, Azure Trace Analysis, and Packaging. Each section is described below.

## Environment

- GPU: Tesla T4, compute capability 7.5, Google Colab
- Driver: 580.82.07
- CUDA: 13.0
- vLLM version: 0.31.0
- vLLM commit: 2a54f6b625f28109180b072b704c0b0d372a277d
- Model: TinyLlama/TinyLlama-1.1B-Chat-v1.0
- Data type: float16
- Python: 3.13 (Google Colab default)

## Setup

Open `main.ipynb` in Google Colab or a local Jupyter environment with GPU access. Run the cells in Section 1 (Setup). This section installs vLLM, clones the vLLM repository to record the exact commit, captures the environment, and downloads the ShareGPT dataset.

The equivalent shell commands are:

```bash
pip install vllm
git clone https://github.com/vllm-project/vllm.git
cd vllm && git rev-parse HEAD > /content/vllm_commit.txt
vllm --version > /content/vllm_version.txt
pip freeze > /content/requirements.txt
nvidia-smi > /content/nvidia_smi.txt
wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json -O /content/sharegpt.json
```

See `data/download_instructions.md` for dataset sources and licensing notes.

## Reproduction Steps

### Section 2: Throughput Benchmark

Run the cells in Section 2 of `main.ipynb`. This runs the current vLLM CLI command `vllm bench throughput` for block sizes 8, 16, and 32 using 300 ShareGPT prompts, seed 0, and max model length 2048. Each run writes a log to `results/throughput_block{8,16,32}.log`. Failures are caught and logged rather than crashing the notebook.

Key command for each block size:

```bash
vllm bench throughput \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --dataset-name sharegpt \
  --dataset-path /content/sharegpt.json \
  --num-prompts 300 \
  --block-size <8|16|32> \
  --max-model-len 2048 \
  --seed 0
```

### Section 3: Azure LLM Inference Trace Analysis

Run the cells in Section 3 of `main.ipynb`. This downloads the Azure LLM Inference Trace, computes context and generated token distributions, arrival burstiness, and the request class split (short, medium, long). Outputs are written to `results/azure_trace_summary.txt` and `results/azure_trace_with_classes.csv`.

### Section 4: Packaging

Run the cells in Section 4 of `main.ipynb`. This collects all environment files, logs, and summaries into a single zip file for download and archival.

## Results

### Throughput Reproduction

300 ShareGPT prompts, seed 0, TinyLlama 1.1B, Tesla T4, float16, max model length 2048.

| Block Size | Requests/sec | Total Tokens/sec | Output Tokens/sec | Notes |
|------------|--------------|------------------|-------------------|-------|
| 16         | 6.55         | 3269.59         | 1677.31           | Baseline |
| 32         | 6.21         | 3099.12          | 1589.86           | About 5 percent slower than block size 16. Triton kernel JIT recompilation observed in log. |
| 8          | N/A          | N/A              | N/A               | Failed. No supported attention backend for block size 8 on compute capability 7.5. |

Original paper results (A100 GPUs, larger models) are not directly comparable due to hardware and model scale differences. Only relative trends across block sizes are meaningful here.

### Azure Trace Summary

- Context tokens: mean 1154.70, p90 2734.50
- Generated tokens: mean 211.13, p90 424.00
- Arrivals per second: mean 5.53
- Burst factor (max/mean): 3.44
- Request class distribution: long 60.5 percent, medium 36.7 percent, short 2.8 percent

## Known Limitations

- Hardware mismatch: Original paper uses A100 GPUs and models at 13B parameters and above. This reproduction uses a Tesla T4 and TinyLlama 1.1B.
- Block size 8 cannot be benchmarked on T4 because vLLM attention backend selection rejects it for compute capability below 8.0. The T4 is compute capability 7.5.
- Benchmark tool interface changed: `benchmark_throughput.py` is deprecated in the vLLM commit used and replaced by `vllm bench throughput`. This reproduction uses the new CLI.
- Only 300 prompts and one seed were used for this initial reproduction. Multiple runs and higher prompt counts are planned for later phases.
- Memory fragmentation was not directly measured in this reproduction. Only throughput was captured. A discrete event simulator will address fragmentation in the next phase.

## Reproducibility Checklist

- Official repository: https://github.com/vllm-project/vllm
- Source code: Yes, full engine source and PagedAttention allocator.
- Experiment scripts: Included in `main.ipynb`.
- Datasets: ShareGPT (used in official benchmarks) and Azure LLM Inference Trace (for workload characterization).
- Configuration files: Block size and model parameters are set in the benchmark commands inside `main.ipynb`.
- Required software: Python 3.9+, PyTorch, CUDA. Versions pinned in `results/requirements.txt`.
- Hardware: Tesla T4 (Google Colab). Original paper used A100.
- Installation instructions: See Setup.
- Instructions to run experiments: See Reproduction Steps.
- Instructions to reproduce reported results: Full scale A100 results cannot be reproduced. Small scale relative trends are reported.

## Report

The full Base Article Selection report is in `report/BaseArticleSelection.pdf`. It includes detailed analysis, dataset documentation, and the reproduction comparison table.

## License

The reproduction materials in this repository are provided for academic reproducibility. The vLLM source code is subject to its own license. The ShareGPT dataset and Azure trace are subject to their respective licenses.
