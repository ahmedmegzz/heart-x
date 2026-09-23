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

## MIT-BIH protocol (Chazal / advisor-aligned)

1. Exclude pacemaker recordings: **102, 104, 107, 217**
2. Keep only DS1 ∪ DS2
3. **DS2 = test only**
4. **Train/val** split patient-wise **within DS1 only**
5. Use **all** annotated AAMI beats (no per-record cap for final runs)

```bash
python scripts/prepare_mitbih.py --method stft
# optional smoke only:
python scripts/prepare_mitbih.py --max-beats-per-record 500
python -m pytest tests/test_patient_split.py tests/test_chazal_protocol.py -q
```

## Train

Primary metric: **macro-F1** (accuracy is secondary).

```bash
python scripts/train_heartx.py \
  --model 1d_2d_clinical \
  --fusion gated \
  --loss focal \
  --class-weight balanced \
  --sampler none \
  --epochs 10
```

| Flag | Values |
|------|--------|
| `--model` | `1d`, `2d`, `clinical`, `1d_2d`, `1d_clinical`, `2d_clinical`, `1d_2d_clinical` / `dual` |
| `--fusion` | `concat`, `gated`, `attention` |
| `--loss` | `ce`, `focal` |
| `--class-weight` | `none`, `balanced` |
| `--sampler` | `none`, `random_oversample`, `smote` (1D+clinical SMOTE; spectrogram regenerated) |

## Advisor experiment suite (multi-seed)

Runs ablation + fusion comparison + imbalance matrix over seeds, then writes mean±std tables:

```bash
python scripts/run_advisor_suite.py --epochs 8 --seeds 42 43 44
# faster subset:
python scripts/run_advisor_suite.py --epochs 5 --quick --seeds 42 43 44
```

Individual runners:

```bash
python scripts/run_ablation.py --epochs 8
python scripts/run_fusion_comparison.py --epochs 8
python scripts/run_imbalance_experiments.py --epochs 6
python scripts/analyze_fusion_weights.py --checkpoint outputs/checkpoints/<run>.pt --fusion gated
python scripts/external_validation.py --checkpoint outputs/checkpoints/<run>.pt
```

## Outputs

- `outputs/processed/mitbih/` — npz, `class_distribution.csv|png`, `meta.json`
- `outputs/results/advisor_suite/` — raw + mean±std CSVs, `advisor_summary.txt`
- `outputs/results/duzeltilmis_dagilim.txt` — corrected split report
- `outputs/figures/` — plots / confusion matrices / Grad-CAM

## SMOTE policy

SMOTE interpolates in **1D + clinical** space. Spectrograms for synthetic rows are **recomputed from synthetic 1D** (not independent 2D SMOTE).
