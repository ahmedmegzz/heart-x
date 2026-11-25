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

## MIT-BIH protocol (de Chazal)

The MIT-BIH Arrhythmia Database has 48 recordings. This study follows the inter-patient DS1/DS2 partition of de Chazal et al.

1. **Remove the pacemaker recordings from the whole study:** **102, 104, 107, 217**. They are not assigned to DS1 or to DS2.
2. Split the remaining 44 recordings into two disjoint sets of 22. A recording belongs to only one set.
   - **DS1** — model selection, training, and validation: 101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122, **124**, 201, 203, 205, 207, 208, 209, 215, 220, 223, 230.
   - **DS2** — test only: 100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210, 212, 213, 214, 219, 221, 222, 228, 231, 232, 233, 234.
3. **Record 124 is inside DS1.** It is used in patient-wise selection and in the DS1 refit. It is not in DS2.
4. **DS2 is evaluated once**, after the configuration is locked. It is not used to choose the model, the fusion, the imbalance method, or the seed.
5. Train/validation splits are **patient-wise and stay inside DS1**. Beats from one recording are not placed in both train and validation.
6. Final runs use **all** annotated AAMI beats (no per-record cap).

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

## Final model selection (DS1 only)

DS2 is not used to choose the model, the fusion, the imbalance method, or the seed.
Record 124 stays inside DS1 for that selection and for the final DS1 fit.
The three candidates are compared with the same patient-wise folds and the same seeds.
The imbalance method attached to the third candidate is the one with the best DS1
cross-validation macro-F1 on the attention model.

```bash
python scripts/select_final_model.py --epochs 4 --folds 4 --seeds 42 43 44
```

Tables land in `outputs/results/final_selection/`. The locked checkpoint is then
retrained on DS1 and evaluated once on DS2. External validation, fusion weights,
and MM-GradCAM are regenerated from that checkpoint.

## Outputs

- `outputs/processed/mitbih/` — npz, `class_distribution.csv|png`, `meta.json`
- `outputs/results/advisor_suite/` — raw + mean±std CSVs, `advisor_summary.txt`
- `outputs/results/duzeltilmis_dagilim.txt` — corrected split report
- `outputs/figures/` — plots / confusion matrices / Grad-CAM

## SMOTE policy

SMOTE interpolates in **1D + clinical** space. Spectrograms for synthetic rows are **recomputed from synthetic 1D** (not independent 2D SMOTE).
