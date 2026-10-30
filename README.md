# Financial Time-Series Generation & Risk Classification

**A reproducible research prototype for synthetic corporate financial time-series generation and enterprise risk classification.**

The project contains both generation and downstream recognition pipelines:

- **TimeVAE**: a conditional variational autoencoder with base, polynomial trend, and quarterly-seasonality branches (earlier neural generative baseline).
- **S0 / S1**: rule-aware **non-neural statistical generators**. The current S1 model performs tail-aware joint-parameter sampling, quarterly dynamic reconstruction, accounting-consistent decoding, and novelty screening.
- **Multi-head LSTM**: 15 independent per-financial-variable LSTM branches. Trend-V1 uses quarter-relative representations; Trend-V2 additionally models financial leverage and leverage changes.
- **Evaluation**: train-only normalization, enterprise-level splitting, nested cross-validation, fold-specific S1 refitting and generation, and generated-data quality audits.

> **Research limitations:** The 80 enterprises and their risk labels are *synthetic simulation data*, not actual company financial records or observed bankruptcy events. Reported numbers come from an internal out-of-fold (OOF) analysis of the synthetic training set, **not** from independent test performance. This prototype is not an operational credit-risk decision system.

## Data and prediction task

Each independent enterprise has **20 consecutive quarterly observations × 15 financial variables**, represented as `(20, 15)` in model code. Input spreadsheets may store five annual rows per company, each containing four quarters.

The 15 features are: `total_assets`, `total_liabilities`, `revenue`, `net_income`, `current_assets`, `current_liabilities`, `cash`, `long_term_debt`, `inventory`, `accounts_receivable`, `cogs`, `gross_profit`, `ebit`, `ebitda`, and `operating_cash_flow`.

Risk labels are binary (`normal`/`risk`). The simulated sample contains **64 normal and 16 risk enterprises**. Fixed, enterprise-disjoint splits are **48 train** (38 normal / 10 risk), **16 validation** (13 / 3), and **16 sealed test** (13 / 3). No quarters of a single enterprise may cross partitions.

Data, trained checkpoints, evaluation outputs, Excel spreadsheets, and generated enterprises are **intentionally excluded** from the repository. Provide your own authorized dataset with the existing `VAE/src/dataset.py` schema, or explicitly run the project's synthetic-data creation and split scripts to reconstruct a compatible experimental setup. Do not reuse previously trained checkpoints after changing the split.

## Repository layout

```text
.
├── VAE/
│   ├── src/         # TimeVAE, structured S0/S1, data, financial constraints, metrics
│   ├── scripts/     # data creation, splitting, training, generating, comparison
│   ├── tests/
│   └── ...
├── LSTM/
│   ├── src/         # 15-head models, trend variants, fold-specific generation, metrics
│   ├── scripts/     # training, nested CV, data augmentation, audit, reporting
│   ├── config/
│   ├── tests/
│   └── ...
├── config/          # TimeVAE training configuration
├── docs/            # Documentation and engineering notes (if present)
├── requirements.txt
└── README.md
```

This repository was assembled from the supplied project source snapshots and add-on packages (through October 30, 2026), **not by remotely reading the live server working tree**. Cross-check changed local files against the server before treating this as a byte-exact release. The TimeVAE V2 source is the latest complete audit snapshot available here; an experimental V3 trainer is stored separately in `experiments/timevae_v3/`. S2.0/S2.1 probe scripts are exploratory, not validated improvements to S1. Files under `data/`, `outputs/`, `logs/`, and `checkpoints/` are not published.

## Core architecture

### Generators

**TimeVAE** uses a compact latent representation (typically `latent_dim=8`) and reconstruction, temporal-difference, and KL terms. Its decoder combines base, trend, and seasonal components. TimeVAE is retained as an earlier model family and should not be confused with S1.

**StructuredFinanceS1** is a statistical generator, **not a TimeVAE checkpoint and not a deep generative neural network**. It fits joint financial drivers to train enterprises, generates quarterly trajectories, decodes the 15 features using financial relationships, and selects candidate trajectories by asset-size and nearest-enterprise-distance criteria. S1 is independently fit for every training fold in leakage-safe cross-validation.

### Risk classification

The baseline model uses **15 separate LSTM heads** (one per feature; these are not Transformer attention heads). Trend-V1 converts each financial series to a relative temporal representation. Trend-V2 adds leverage-level and leverage-change features. A sigmoid output provides a binary enterprise risk probability.

### Data augmentation comparison

- `real_only` (**L0**): original simulated training enterprises only.
- `generated_only` (**L1**): training-fold-specific S1 generated enterprises only.
- `real_plus_s1` (**L2**): real + S1 generated enterprises.

The nested CV pipeline uses company-disjoint outer folds and inner checkpoint/epoch selection. S1 and every scaler are refit **within the relevant training fold**, never on the held-out fold. Reported 5-fold results use the same OOF enterprises across the three regimes; the fixed threshold is 0.5.

## Setup

Environment used in the research: Ubuntu 24.04, Python 3.11, PyTorch, NVIDIA RTX 3090; most generation/training scripts require **physical GPU1 only**. The visible PyTorch device is `cuda:0` when `CUDA_VISIBLE_DEVICES` exposes only physical GPU1. Do not allow fallback to physical GPU0 or CPU for the guarded formal runs.

```bash
conda activate sers_ddpm
# From the cloned repository root:
python -m pip install -r requirements.txt
python -m pytest -v VAE/tests LSTM/tests
```

Check the actual CLI options in each script (`--help`) and review the YAML files before running. For formal GPU1 runs:

```bash
GPU1_UUID=$(nvidia-smi -i 1 --query-gpu=uuid --format=csv,noheader)
test "$GPU1_UUID" = 'GPU-b74f16a4-c211-d07a-2642-eb1e81a3c2dd' || { echo 'Required GPU1 unavailable'; exit 1; }
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$GPU1_UUID"
```

**Example entry points** (require valid data files and the original project configuration):

```bash
# Earlier neural generation baseline (TimeVAE):
python -m VAE.scripts.train --config config/vae.yaml

# S1 non-neural statistical generation, per-class counts:
python -m VAE.scripts.structured_generate_s1 \
  --risk-number 100 --normal-number 100 --output-name structured_s1_example

# Prepare data for existing LSTM pipelines:
python -m LSTM.scripts.prepare_data \
  --generated VAE/outputs/generated/structured_s1_example \
  --output LSTM/data/example

# Compare Trend-V1 and Trend-V2 with nested CV (train.xlsx only):
python -u -m LSTM.scripts.cv_trend_v1_v2

# Nested fold-specific S1 augmentation (train.xlsx only):
python -u -m LSTM.scripts.cv_trend_v2_s1_augmentation

# Audit saved fold-generated trajectories without retraining:
python -u -m LSTM.scripts.audit_foldwise_s1
```

Commands involving GPU must be invoked only after the GPU1 visibility checks. Formal output directories must be new and should never overwrite earlier experiments. Unit tests are not proof of model validity.

## Experimental results (simulated training set, OOF)

The following values were observed in the **48-enterprise internal nested five-fold** comparison (10 risk and 38 normal enterprises). These are exploratory results from one experiment protocol—not measurements on real companies or the sealed test set.

| Metric | L0: real only | L1: generated only | L2: real + S1 |
| --- | ---: | ---: | ---: |
| AUROC ↑ | 0.7947 | **0.8816** | 0.8658 |
| PR-AUC ↑ | 0.5803 | **0.7115** | 0.6815 |
| Risk-class F1 ↑ | 0.6087 | 0.6154 | **0.6667** |
| Balanced accuracy ↑ | 0.7711 | 0.7947 | **0.8211** |
| Brier score ↓ | 0.1658 | 0.1636 | **0.1434** |

At the fixed 0.5 threshold, the L2 OOF confusion matrix is **TP=8, FN=2, FP=6, TN=32** (risk class is positive), giving 80% risk recall and 83.33% overall accuracy. This is a very small synthetic sample; a single risk-company prediction changes recall by 10 percentage points.

Paired exploratory Bootstrap intervals for **L2 − L0** cross zero: AUROC `[-0.0040, 0.1684]`, PR-AUC `[-0.1198, 0.3339]`; the Brier difference interval is `[-0.0630, 0.0158]`. These data **do not establish a statistically significant or causally identified improvement**.

## S1 generation audit and limitations

Across five folds, two stages (inner and outer), and two labels, the saved audit covered **2,000 generated enterprise trajectories**. Under the audit's implemented checks:

| Audit item | Observed result |
| --- | --- |
| Financial-rule maximum violation rate | 0% |
| Exact copies of training companies | 0 |
| Exact duplicate generated companies | 0 |
| Below S1 nearest-neighbor novelty threshold | 234 / 2000 (**11.7%**) |
| Generated-to-real pairwise-distance ratio | 0.534–1.100 |
| Quality audit status | `REVIEW_REQUIRED` |

The diversity and novelty thresholds are heuristic, not guarantees of privacy, non-memorization, or real-world financial realism. A replay diagnostic found that S1's novelty **selection stage** shifted the risk-enterprise EBIT margin by approximately **−2.48 percentage points** on average relative to the prescreening candidate pool. An experimental selection-rule probe may be present in the repository, but **S2.1 is not validated as a replacement for S1**.

## Reproducibility and safety notes

- Fit normalizers and any synthetic generator exclusively on the applicable **train enterprises**.
- Keep `validation` for allowed model selection only; preserve `test` for a single final assessment.
- Under fold-wise augmentation, re-fit S1 independently on inner and outer training partitions. Never use an S1 dataset generated from all 48 train enterprises to evaluate a held-out subset of those 48.
- Quality constraints, non-copying, AUROC, and synthetic OOF improvement do **not** establish real-company credit-risk fitness.
- All enterprises in the described experiment are simulated. Do not describe the experiments as a verified actual bankruptcy predictor.
- Output metrics, training records, and model checkpoints are omitted from GitHub to keep the repository code-only and protect data provenance.

## Status

**Current frozen baseline:** S1 structured statistical generation + Trend-V2 15-head LSTM classification, with leakage-safe augmentation and quality audits. S2.1 selection improvements are exploratory only. No finalized independent test performance is claimed.
