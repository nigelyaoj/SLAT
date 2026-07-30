# SLAT: Segment-Level Adaptive Trimming for Efficient CoT Reasoning

SLAT is a research implementation built on the
[verl](https://github.com/verl-project/verl) framework for improving the
efficiency of chain-of-thought reasoning.
The original experiments were run on internal computing infrastructure, and we provide the public implementation and the released model checkpoints for further research.

## Released Models

The released checkpoints are available in the
[SLAT collection on Hugging Face](https://huggingface.co/collections/nigelyaoj/slat).

## Installation

Set up the environment by following the
*verl installation guide*. Make sure the environment
supports the rollout backend and distributed setup used by your experiment.

## Quick Start

The provided [`run.sh`](run.sh) contains an example GRPO training configuration with SLAT enabled.

Before launching it, update the environment-specific values:

- `train_files` and `test_files`
- `actor_rollout_ref.model.path`
- `trainer.default_local_dir`
- `trainer.n_gpus_per_node` and `trainer.nnodes`
- Weights & Biases credentials, or `trainer.logger` if W&B is not required

The included configuration targets 4 nodes with 8 GPUs per node and uses vLLM
for rollout generation. Launch it with:

```bash
chmod +x run.sh
./run.sh
```