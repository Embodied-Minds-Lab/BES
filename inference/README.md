# Inference

Inference-time code for the three open-problem solving benchmarks in §5.2 of the paper. Built on top of [`ShinkaEvolve`](https://github.com/SakanaAI/ShinkaEvolve), we keep the `shinka` Python package name and the upstream Apache-2.0 license. The code specific to BES lives in `shinka/bidirectional_search/` (backward search) and `shinka/database/score.py` (scoring).

## 1. Setup

Requires Python >= 3.10.

```bash
conda create -n bes_inference python=3.11
conda activate bes_inference
pip install -e .
```

Put your API keys in a `.env` at the repo root (auto-loaded on `import shinka`):

```
OPENAI_API_KEY=...
GEMINI_API_KEY=...
```

Anthropic, AWS Bedrock, Azure OpenAI, DeepSeek, OpenRouter, and a local OpenAI-compatible backend are also supported (set the matching `*_API_KEY`).

## 2. Running the three benchmarks

Each benchmark is a self-contained directory under `tasks/`. The yamls use relative paths (`init_program_path: initial.py`, `results_dir: ./results/...`), so **launch from inside the task directory**.

### Circle Packing (Square)

Pack `n=26` non-overlapping circles in a unit square; maximize sum of radii.

```bash
cd tasks/circle_packing
python run_evo.py --config_path shinka_backward_adaptive_gpt5.yaml
```

### Circle Packing (Rect)

Pack `n=21` non-overlapping circles in a rectangle of perimeter 4 (`width + height = 2`, aspect ratio chosen per candidate); maximize sum of radii.

```bash
cd tasks/circle_packing_rect
python run_evo.py --config_path shinka_backward_adaptive_gpt5.yaml
```

### Heilbronn (Convex)

Place `n=13` points anywhere in the plane; maximize the minimum triangle area over all 3-point subsets, normalized by the convex hull area of the chosen points.

```bash
cd tasks/heilbronn_convex
python run_evo.py --config_path shinka_backward_adaptive_gpt5.yaml
```

### SLURM

Each task also ships a `run_gpt5_slurm.sh`. **Edit the env-activation lines at the top** for your cluster (commented placeholders), then:

```bash
sbatch tasks/<task_name>/run_gpt5_slurm.sh shinka_backward_adaptive_gpt5.yaml
```

## 3. Expected performance

Backbone: GPT-5. Higher is better; **bold** = best Avg per benchmark column.

| Strategy             | Circle Packing (Sq.) Avg / Best | Circle Packing (Rect) Avg / Best | Heilbronn (Convex) Avg / Best |
|:---------------------|:---------------------------------|:---------------------------------|:------------------------------|
| Human                | – / 2.634                        | – / 2.364                        | – / 0.0306                    |
| AlphaEvolve (closed) | – / 2.635                        | – / 2.3658                       | – / 0.0309                    |
| OpenEvolve           | 2.531 ± .018 / 2.541             | 2.267 ± .014 / 2.276             | 0.025 ± .005 / **0.027**      |
| GEPA                 | 2.613 ± .022 / 2.628             | 2.326 ± .023 / 2.354             | 0.025 ± .002 / **0.027**      |
| ShinkaEvolve         | 2.464 ± .083 / 2.541             | 2.335 ± .026 / 2.358             | 0.023 ± .005 / 0.026          |
| **BES (ours)**       | **2.623 ± .014 / 2.632**         | **2.349 ± .012 / 2.360**         | **0.026 ± .001 / 0.027**      |


## 4. Reproduction

The programs we found per task are checked in under `results/`:

```
results/
├── circle_packing/        1.py  2.py  3.py
├── circle_packing_rect/   1.py  2.py  3.py
└── heilbronn_convex/      1.py  2.py  3.py
```

To verify any individual program, run it through the corresponding task's evaluator. 

## 5. Acknowledgments

We thank the authors of [ShinkaEvolve](https://github.com/SakanaAI/ShinkaEvolve) for sharing their code and making it available for open-source use.