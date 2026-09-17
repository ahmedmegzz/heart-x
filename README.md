# HEART-X

Hybrid and explainable dual-stream deep learning for ECG-based heart-disease diagnosis
(*Hibrit ve Açıklanabilir Derin Öğrenme Mimarisi ile EKG Tabanlı Çok Modlu Kalp Rahatsızlıkları Teşhis Sistemi*).

## Setup

```bash
cd /home/kimbra/Documents/GitHub/tez
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=src PYTHONUNBUFFERED=1
```

Data:

- `data/mit-bih-arrhythmia-database-1.0.0/`
- `data/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/`

## Pipeline overview

| Stage | Code |
|------|------|
| Patient-wise split + class reports | `src/heartx/data/patient_split.py`, `class_report.py` |
| 1D + STFT/CWT dual reps | `src/heartx/features/spectrogram.py` |
| CNN-LSTM + fusion (`concat`/`gated`/`attention`) | `src/heartx/models/` |
| MM-GradCAM | `src/heartx/xai/mm_gradcam.py` |
| Train (macro-F1 primary) | `scripts/train_heartx.py` |

## Prepare data (patient-wise)

```bash
python scripts/prepare_mitbih.py --max-beats-per-record 500
python scripts/prepare_ptbxl.py --max-records 2000
```

Writes `train_mask` / `val_mask` / `test_mask`, `class_distribution.csv|png`, and unique patient/record counts under `outputs/processed/`.

MIT-BIH: Chazal DS2 = test; DS1(+others) → GroupShuffleSplit train/val.  
PTB-XL: fold-10 patients = test; remaining patients → train/val.

```bash
python -m pytest tests/test_patient_split.py -q
```

## Train

```bash
python scripts/train_heartx.py \
  --model dual \
  --fusion gated \
  --loss focal \
  --class-weight balanced \
  --sampler none \
  --epochs 10
```

Key CLI:

| Flag | Values | Notes |
|------|--------|-------|
| `--model` | `1d`, `2d`, `clinical`, `1d_2d`, `1d_clinical`, `2d_clinical`, `dual` / `1d_2d_clinical`, `dual_noclinical` | Ablation building blocks |
| `--fusion` | `concat` (default), `gated`, `attention` | Feature-level fusion |
| `--loss` | `ce`, `focal` | |
| `--class-weight` | `none`, `balanced` | |
| `--sampler` | `none`, `random_oversample`, `smote` | SMOTE: 1D(+clinical) then **regenerate** spectrogram from synthetic 1D |

Checkpoint selection and LR schedule use **macro-F1** (accuracy is secondary).

### Fusion weight analysis

```bash
python scripts/analyze_fusion_weights.py \
  --checkpoint outputs/checkpoints/heartx_<run>.pt \
  --fusion gated
```

## Experiment runners

```bash
# 7-way modality ablation
python scripts/run_ablation.py --epochs 8

# Imbalance matrix → outputs/results/imbalance_comparison.csv
python scripts/run_imbalance_experiments.py --epochs 6
# or smaller:
python scripts/run_imbalance_experiments.py --quick --epochs 3

# External validation (binary normal/abnormal proxy MIT-BIH ↔ PTB-XL)
python scripts/external_validation.py --checkpoint outputs/checkpoints/heartx_<run>.pt
```

## Explainability

```bash
python scripts/explain_mm_gradcam.py
```

## Outputs

- `outputs/processed/` — npz, class distributions  
- `outputs/checkpoints/` — best macro-F1 weights  
- `outputs/results/` — per-run metrics, ablation / imbalance / external tables  
- `outputs/figures/` — confusion matrices, ablation plot, fusion weights, Grad-CAM  

## SMOTE policy (important)

SMOTE interpolates in **1D waveform + clinical** space only. Spectrogram inputs for synthetic samples are **recomputed via STFT/CWT** from the synthetic 1D signal so the dual streams stay physically consistent. Independent SMOTE in spectrogram pixel space is intentionally avoided.
