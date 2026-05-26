# Logical Reasoning

Code for Section 5.1.1 ("Logical Reasoning") of the paper. Trains
Gemma-3-1B-IT on Knights-and-Knaves (K&K) puzzles, comparing GRPO,
MaxRL, and BES.

## 1. Install

Follow the [official instructions from verl](https://verl.readthedocs.io/en/latest/start/install.html) to install the dependencies.

## 2. Data

`data/` contains the K&K parquet files referenced by the training scripts:

```
data/
├── LICENSE                 (from mem-kk-logic)
├── train.parquet
└── eval.parquet
```

The SFT checkpoint we used is published at [`Xkev/gemma-3-1b-it-kk`](https://huggingface.co/Xkev/gemma-3-1b-it-kk)
and the RL launcher pulls it from HF Hub directly. 
If you want to redo the cold-start SFT on your own data, see Re-run SFT below.

## 3. Train

`verl/scripts/run_bes_local.sh` and `verl/scripts/run_bes_slurm.sh`
accept `--algo {grpo, maxrl, bes}` (default `--algo bes`), covering all
three training jobs.  All
three use [`Xkev/gemma-3-1b-it-kk`](https://huggingface.co/Xkev/gemma-3-1b-it-kk)
as the base model.

### Bash mode

Use this for development on one machine. It starts the
backward vLLM server (only for BES) and the trainer as background
processes on the same node:

```bash
conda activate <your-env>
bash verl/scripts/run_bes_local.sh                  # default: BES
bash verl/scripts/run_bes_local.sh --algo grpo
bash verl/scripts/run_bes_local.sh --algo maxrl
```


### SLURM mode

For BES, first submit the long-running backward vLLM server (it
publishes its URL to `verl/backward_server_url.txt` and stays up):

```bash
sbatch -A <account> -p <partition> verl/scripts/run_backward_slurm.sh
```

Then submit any of the three training jobs:

```bash
bash verl/scripts/run_bes_slurm.sh                  # default: BES
bash verl/scripts/run_bes_slurm.sh --algo grpo
bash verl/scripts/run_bes_slurm.sh --algo maxrl
```

### Re-run SFT

To re-create the SFT checkpoint from scratch on your own SFT data,
prepare a parquet with verl's SFT message schema (chat-formatted
`messages` column), edit the `TRAIN_DATA=` placeholder in
`verl/scripts/run_sft_slurm.sh`, then:

```bash
sbatch -A <account> -p <partition> verl/scripts/run_sft_slurm.sh
```

Defaults: `optim.lr=1e-5`, `trainer.total_epochs=3`, bf16, single GPU,
base `google/gemma-3-1b-it`. Point the RL launcher's `MODEL_PATH=` at
the output checkpoint to use your SFT in place of `Xkev/gemma-3-1b-it-kk`.

## 4. Reproduction

Both checkpoints used in the paper are released on HuggingFace Hub for
direct reuse / evaluation:

- SFT cold-start: [`Xkev/gemma-3-1b-it-kk`](https://huggingface.co/Xkev/gemma-3-1b-it-kk)
- BES post-trained: [`Xkev/gemma-3-1b-it-kk-bes`](https://huggingface.co/Xkev/gemma-3-1b-it-kk-bes)

## 5. Troubleshooting

We developed and ran this code on the Harvard FAS-RC cluster, and we are fully
aware that environments differ a lot between machines and getting them lined up is genuinely hard. It took us a non-trivial amount of time to get this stack running ourselves, so if you run into problems trying to reproduce, that is completely normal.

Please open a GitHub issue with as much detail as you can (full log, command
line, partition / GPU type, conda env, vLLM and faiss versions). We will reply
as soon as we have time, usually within a few hours to a few days.


## 6. Acknowledgments

We thank the authors of [mem-kk-logic](https://arxiv.org/abs/2410.23123)
for the K&K puzzle generator  and
[verl](https://github.com/volcengine/verl) for the RL post-training
framework on top of which the BES agent loops are implemented.
