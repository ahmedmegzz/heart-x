# HEART-X

Hybrid and explainable dual-stream deep learning for ECG-based heart-disease diagnosis
(thesis pipeline for *Hibrit ve Açıklanabilir Derin Öğrenme Mimarisi ile EKG Tabanlı Çok Modlu Kalp Rahatsızlıkları Teşhis Sistemi*).

## Pipeline (aşamalar)

| Stage | Description | Code |
|------|-------------|------|
| 2 | MIT-BIH / PTB-XL load, band-pass + notch, R-peak beats, clinical features | `src/heartx/data/` |
| 3 | Dual representation: 1D window + STFT/CWT spectrogram | `src/heartx/features/spectrogram.py` |
| 4 | CNN-LSTM (1D) + CNN (2D) dual-stream | `src/heartx/models/dual_stream.py` |
| 5 | Feature-level fusion with RR / QRS clinical features | `HeartXDualStream(clinical_dim=...)` |
| 6 | MM-GradCAM heatmaps | `src/heartx/xai/mm_gradcam.py` |
| 7 | Accuracy / F1 / sensitivity / specificity + baselines | `scripts/train_heartx.py` |

## Setup

```bash
cd /home/kimbra/Documents/GitHub/tez
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Data is expected under:

- `data/mit-bih-arrhythmia-database-1.0.0/`
- `data/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/`

## Quick start (MIT-BIH)

```bash
export PYTHONPATH=src
# Limit beats for a faster first run; omit flag for full dataset
python scripts/prepare_mitbih.py --max-beats-per-record 500
python scripts/train_heartx.py --model dual --epochs 10
python scripts/explain_mm_gradcam.py
```

Baselines / ablation:

```bash
python scripts/train_heartx.py --model 1d --epochs 10
python scripts/train_heartx.py --model 2d --epochs 10
python scripts/train_heartx.py --model dual_noclinical --epochs 10
```

PTB-XL subset:

```bash
python scripts/prepare_ptbxl.py --max-records 2000
```

Outputs land in `outputs/processed/`, `outputs/checkpoints/`, and `outputs/figures/`.
