# BES: Self-Improving Language Models with Bidirectional Evolutionary Search

<p align="center">
  <a href="https://arxiv.org/abs/PLACEHOLDER"><img src="https://img.shields.io/badge/paper-arxiv.PLACEHOLDER-B31B1B.svg" /></a>
  <img src="https://img.shields.io/badge/python-%3E%3D3.10-blue" />
  <a href='https://huggingface.co/collections/Xkev/bes'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Page-blue'></a>
  <a href='https://guoweixu.com/bes/'><img src='https://img.shields.io/badge/Project-Page-Green'></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" /></a>
</p>

## Overview

Search has been proposed as an effective method for self-improving language models and agentic systems, both for post-training sample generation and for inference. However, widely used methods such as best-of-N sampling and tree search face two fundamental limitations: they are guided by **sparse verification signals**, and they construct candidates primarily through **autoregressive expansion**, restricting exploration to regions with substantial model probability mass.

We propose **Bidirectional Evolutionary Search (BES)**, a search framework that couples *forward candidate evolution* with *backward goal decomposition*. The forward search augments standard expansion with evolution operators (combination, translocation, deletion, crossover) that recombine parts of existing trajectories into candidates that are difficult to reach from a single rollout. The backward search recursively decomposes the task objective into a tree of checkable sub-goals, producing dense intermediate feedback that prioritizes which forward candidates to grow.

<p align="center">
  <img src="assets/teaser.png" width="85%" />
</p>

## Experiments

We evaluate BES on both post-training and inference across LLM and agent settings. For post-training, we consider Logical Reasoning (LLM) and Multi-Hop Reasoning (Agent). For inference, we consider three representative open problem solving benchmarks: Circle Packing (Square), Circle Packing (Rectangle), and the Heilbronn Convex problem.

Each setting is self-contained under its own directory, with its own README, data, and launchers:

| Directory |  Setting |
|---|---|
| [`logical/README.md`](logical/README.md) | RL post-training on Knights-and-Knaves with Gemma-3-1B-it (GRPO / MaxRL / BES) |
| [`multihop/README.md`](multihop/README.md) | RL post-training on MuSiQue with Llama-3.2-3B / Llama-3.1-8B (GRPO / Tree-GRPO / BES) |
| [`inference/README.md`](inference/README.md) | Inference-time open-problem solving on Circle Packing (Square / Rect) and Heilbronn (Convex), built on top of ShinkaEvolve |

## Citation

If you find this work useful, please cite:

```bibtex
@article{bes2026,
  title   = {PLACEHOLDER},
  author  = {PLACEHOLDER},
  journal = {PLACEHOLDER},
  year    = {2026},
}
```
