# DASE7506 MP1 — Small Language Model Challenge

**Sun Ruoheng · 3036707434**

| | |
|---|---|
| **Test BPB (CPU, FP32)** | **1.5668** |
| Validation BPB | 1.5441 |
| Baseline test BPB | 2.1013 |
| Parameters | 5,260,032 |
| Checkpoint SHA-256 | `12aacf505d700bb595085a9199b7faba48c393164ae297a96a623bf4b56796db` |

Report: [`report/report.pdf`](report/report.pdf) (source: `report/report.md`, figures alongside).
Full experiment log: [`code/RUN_LOG_TEMPLATE.csv`](code/RUN_LOG_TEMPLATE.csv) — every run, with
its seed, configuration, commit, wall-clock time, score and checkpoint hash.

## The model

Width 256, 6 blocks, 8 heads, context 256. Rotary position embeddings (RoPE) instead of
learned absolute position embeddings, RMSNorm instead of LayerNorm, dropout 0.1 on the
residual branches during training only. Trained for 9,600 updates of batch 32 with peak
learning rate 0.0025 and seed 17, on CPU in FP32.

Measured on an idle MacBook Air (M5, 24 GB), three alternating passes per checkpoint:

| Budget | Measured | Limit |
|---|---|---|
| CPU scoring time | 11.54 s vs baseline 3.92 s = 2.94x | 5x |
| Peak evaluation RAM | 1.83 GiB | 4 GiB |
| Uncompressed inference assets | 20.1 MiB | 64 MiB |

## Install

Python 3.12. From `code/`:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\Activate.ps1
python -m pip install torch==2.7.1 # macOS: default PyPI index; Linux/Windows CPU: --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

Verified with Python 3.12 and PyTorch 2.7.1 on macOS (arm64, CPU).

## Evaluate the submitted checkpoint (no retraining)

Download `checkpoint.pt` from this repository's Releases page, then from `code/`:

```bash
python evaluate.py --checkpoint /path/to/checkpoint.pt \
  --device cpu --precision fp32 --split test
```

Expected: `bpb` 1.5668356938894281. Scoring is deterministic — repeated runs reproduce this
to all printed digits. `test_cpu_fp32.json` from my own run is attached to the same release
for comparison.

## Reproduce the training run

About 1 hour 52 minutes on the machine above.

```bash
python train.py --implementation student --config configs/rr_w256_d6_do01.json \
  --device cpu --threads 4 --seed 17 --steps 9600 --eval-every 1200 \
  --lr 0.0025 --run-dir runs/final-9600
python evaluate.py --checkpoint runs/final-9600/checkpoint.pt \
  --device cpu --precision fp32 --split test
```

## What was changed

| File | Change |
|---|---|
| `code/student.py` | My model. Switchable position encoding (`learned` / `rope`), normalization (`layernorm` / `rmsnorm`), feed-forward (`gelu` / `swiglu`) and dropout, so each mechanism can be ablated from config alone. With every switch at its default it reproduces `model.py` bit-for-bit. |
| `code/train.py` | One added option, `--lr` (default 0.001, i.e. the original behaviour), recorded in `metrics.json`. |
| `code/configs/*.json` | Configurations for the experiments; `rr_w256_d6_do01.json` is the submitted model. |
| `code/model.py`, `code/configs/baseline.json` | Unchanged, preserved for comparison. |
| `code/common.py`, `code/evaluate.py`, `code/data/` | Unchanged, as required. |

## Method summary

1. **Training recipe** (no inference cost): the baseline's 1,200 updates and peak learning
   rate of 0.001 were both conservative. 2,400 updates at 0.005: 2.0711 -> 1.7817 validation.
2. **RoPE** (headline mechanism): relative position built into the attention computation
   rather than learned from absolute slots. 1.7817 -> 1.7042, with 32,768 fewer parameters.
   The `learned` control reproduces the supplied baseline exactly, so the ablation isolates
   one factor at equal processed targets.
3. **RMSNorm** adopted (1.7042 -> 1.6859); **SwiGLU** tested and rejected as a tie (1.7050).
4. **Capacity**: six shapes timed from untrained checkpoints before training any. Depth beats
   width at equal cost; wider models need lower learning rates.
5. **Dropout 0.1** once the larger model began to overfit: 1.6248 -> 1.5872.
6. **9,600 updates** for the final run: 1.5441 validation, 1.5668 test.

Development used the validation split only. The method was frozen before the test split was
evaluated, once.

## Cost

21 training runs, 6 hours 39 minutes of CPU training in total (final run 1 h 52 m,
exploration 4 h 47 m). No run was initialised from another run's weights.

## AI assistance

I used AI (Claude, Anthropic) throughout, as a tutor and as a coding assistant. It explained
how the baseline works and what the concepts mean; it suggested which changes to try (RoPE,
RMSNorm, SwiGLU, dropout, scaling) and how to test them fairly, including the control
experiment, the noise measurement and the timing method; it provided code suggestions and
explanations, which I then reviewed, modified and integrated into `student.py` and the `--lr`
option; and it helped draft the report from my results. I ran every training run and every
measurement myself on my own laptop, and every number reported comes from my own logs. I
decided which experiments to run, which to skip, and which model to submit. I have read and
understood the submitted code and the claims in the report.

## Data

WikiText-2, as supplied with the assignment package; see `code/README.md` for the upstream
attribution and licence notices (CC BY-SA 3.0 and GFDL). The data, tokenizer and evaluator
are unchanged.
