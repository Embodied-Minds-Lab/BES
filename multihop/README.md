# Multi-Hop Reasoning

Code for Section 5.1.2 ("Multi-Hop Reasoning") of the paper. Trains
Llama-3.2-3B-Instruct and Llama-3.1-8B-Instruct on the 3–4-hop solvable
subset of MuSiQue and evaluates on the full official MuSiQue validation
set, comparing GRPO, Tree-GRPO, and BES.

## 1. Install

```bash
conda create -n bes_multihop python=3.11
conda activate bes_multihop
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0
pip install vllm==0.8.5.post1
pip install -e .
pip install flash-attn --no-build-isolation
pip install transformers datasets pyserini faiss-gpu==1.7.3 uvicorn fastapi
```

**vLLM patch.** Apply
[vllm-project/vllm#23477](https://github.com/vllm-project/vllm/pull/23477) to
`vllm/device_allocator/cumem.py` in your env (or skip if vllm ≥ 0.10.1).
Without it, ~3–5 h of training may trigger a
`Fatal Python error: none_dealloc` in the C extension and Ray kills the
trainer with `Worker exit type: SYSTEM_ERROR`.

```diff
@@ class CuMemAllocator:
     def __init__(self):
         ...
         self.allocator_and_pools: Dict[str, Any] = {}
+        # keep strong refs so the C extension's borrowed refs survive GC
+        self.python_malloc_callback = self._python_malloc_callback
+        self.python_free_callback   = self._python_free_callback
-    def python_malloc_callback(self, allocation_handle): ...
+    def _python_malloc_callback(self, allocation_handle): ...
-    def python_free_callback(self, ptr): ...
+    def _python_free_callback(self, ptr): ...
```

## 2. Data

Download the Wikipedia E5 index + corpus, build the base parquets, then
split out the 3–4-hop subset and re-tag `data_source` by hop count.

```bash
SAVE=/path/to/data
DATA_DIR=$SAVE/datasets/multihop_musique
FLASHRAG_DIR=/path/to/FlashRAG_Dataset   # https://github.com/RUC-NLPIR/FlashRAG
mkdir -p $DATA_DIR

# 1. Wikipedia corpus + E5 index
python scripts/download.py --save_path $SAVE
cat $SAVE/part_* > $SAVE/e5_Flat.index
gzip -d $SAVE/wiki-18.jsonl.gz

# 2. MuSiQue parquets
python scripts/data_process/qa_search_train_merge.py \
    --local_dir $DATA_DIR --flashrag_dir $FLASHRAG_DIR --data_sources musique
python scripts/data_process/qa_search_test_merge.py \
    --local_dir $DATA_DIR --flashrag_dir $FLASHRAG_DIR --data_sources musique

# 3. 3-4 hop subset for training; full musique test set for eval
python scripts/prepare_hop_split_musique.py --src_dir $DATA_DIR
# → train.parquet, test.parquet, hard_train.parquet, hard_test.parquet
```

`hard_train.parquet` is the 3–4-hop training set (~5.5k examples);
`hard_test.parquet` is the official MuSiQue validation set.

## 3. Servers

Training relies on one or two long-running services that talk to the
trainer over HTTP:

- **Retriever** (always required) — E5 + FAISS over `wiki-18`; publishes its URL to `$DATA_ROOT/retriever.endpoint`.
- **Backward server** (BES only) — vLLM OpenAI-compatible endpoint serving models for subgoal decomposition; publishes its URL to `musique_backward_server_url.txt`.

`scripts/run_retriever_slurm.sh` and `scripts/run_backward_slurm.sh` are
the two **portable** SLURM templates — set `DATA_ROOT` / `BACKWARD_MODEL` /
`CONDA_ENV` near the top, then submit:

```bash
sbatch -A <account> -p <partition> scripts/run_retriever_slurm.sh   # always
sbatch -A <account> -p <partition> scripts/run_backward_slurm.sh    # only for --algo bes
```

## 4. Train

`scripts/run_bes_local.sh` and `scripts/run_bes_slurm.sh` accept
`--algo {grpo, tree-grpo, bes}` and `--model {3b, 8b}` (default
`--algo bes --model 8b`), covering all six training jobs below.

### Bash mode

Use this for development on one machine. It starts the
retriever, backward server, and trainer as background processes on the
same node (4 GPUs total; the backward server is skipped when not running
BES):

```bash
conda activate bes_multihop
bash scripts/run_bes_local.sh                     # default: BES, Llama-3.1-8B
bash scripts/run_bes_local.sh --algo grpo --model 3b
bash scripts/run_bes_local.sh --algo tree-grpo --model 8b
```


### SLURM mode

First submit the long-running servers (once; they are shared across
trainer runs):

```bash
sbatch -A <account> -p <partition> scripts/run_retriever_slurm.sh
sbatch -A <account> -p <partition> scripts/run_backward_slurm.sh    # only for --algo bes
```

Then submit any of the six training jobs:

```bash
bash scripts/run_bes_slurm.sh                     # default: BES, Llama-3.1-8B
bash scripts/run_bes_slurm.sh --algo grpo --model 3b
bash scripts/run_bes_slurm.sh --algo tree-grpo --model 8b
```


## 5. Expected results

Reported on the full MuSiQue validation set after 120 global steps. 

| Method | Accuracy (%) ↑ | # Valid Search ↑ | # Valid Actions ↑ | Finish Ratio ↑ |
|---|---:|---:|---:|---:|
| **Llama-3.2-3B-Instruct** | | | | |
| Base       | 4.0          | –    | –    | –    |
| + GRPO     | 2.1 (−1.9)   | 0.84 | 0.20 | 0.64 |
| + Tree-GRPO| 3.9 (−0.1)   | 1.50 | 2.14 | 0.64 |
| + BES      | **7.0 (+3.0)** | **2.31** | **3.29** | **0.97** |
| **Llama-3.1-8B-Instruct** | | | | |
| Base       | 6.6          | –    | –    | –    |
| + GRPO     | 5.6 (−1.0)   | 1.46 | 1.83 | 0.37 |
| + Tree-GRPO| 7.4 (+0.8)   | 0.65 | 1.36 | 0.71 |
| + BES      | **10.4 (+3.8)** | **2.11** | **3.05** | **0.94** |

## 6. Reproduction

Metrics for all six runs above are published as a
W&B report: <https://api.wandb.ai/links/xkev/c9gp7y4y>.

Both checkpoints trained by BES are released on HuggingFace Hub for
direct reuse / evaluation:

- 3B: [`Xkev/Llama-3.2-3B-Instruct-multihop-BES`](https://huggingface.co/Xkev/Llama-3.2-3B-Instruct-multihop-BES)
- 8B: [`Xkev/Llama-3.1-8B-Instruct-multihop-BES`](https://huggingface.co/Xkev/Llama-3.1-8B-Instruct-multihop-BES)

## 7. Troubleshooting

We developed and ran this code on the Harvard FAS-RC cluster, and we are fully
aware that environments differ a lot between machines and getting them lined up is genuinely hard. It took us a non-trivial amount of time to get this stack running ourselves, so if you run into problems trying to reproduce, that is completely normal.

Please open a GitHub issue with as much detail as you can (full log, command
line, partition / GPU type, conda env, vLLM and faiss versions). We will reply
as soon as we have time, usually within a few hours to a few days.

## 8. Acknowledgments

We thank the authors of [Tree-GRPO](https://github.com/AMAP-ML/Tree-GRPO) for sharing their code and making it available for open-source use.