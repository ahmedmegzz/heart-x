"""
Label mapping between MIT-BIH AAMI beat classes and PTB-XL superclasses.

Thesis note
-----------
These taxonomies are **not** equivalent. MIT-BIH AAMI labels describe
beat-level arrhythmia morphology (N/S/V/F/Q). PTB-XL superclasses describe
record-level diagnostic groups (NORM/MI/STTC/CD/HYP).

External validation therefore uses a **coarse proxy mapping** for binary /
reduced evaluation only:

- ``NORM`` (PTB-XL)  ↔  ``N`` (AAMI normal beat family)
- Non-NORM PTB-XL    ↔  ``abnormal`` proxy (collapsed S/V/F/Q on MIT-BIH)

Fine-grained cross-dataset transfer of S/V/F/Q is **not claimed**; results
must be reported as proxy external validation, not direct class transfer.
"""

from __future__ import annotations

from heartx.config import AAMI_CLASSES, PTBXL_CLASSES

# Binary proxy labels used for cross-dataset evaluation
PROXY_CLASSES = ["normal", "abnormal"]
PROXY_TO_IDX = {"normal": 0, "abnormal": 1}

# AAMI -> proxy
AAMI_TO_PROXY = {
    "N": "normal",
    "S": "abnormal",
    "V": "abnormal",
    "F": "abnormal",
    "Q": "abnormal",
}

# PTB-XL superclass -> proxy
PTBXL_TO_PROXY = {
    "NORM": "normal",
    "MI": "abnormal",
    "STTC": "abnormal",
    "CD": "abnormal",
    "HYP": "abnormal",
}


def aami_idx_to_proxy_idx(y_aami) -> list[int]:
    return [PROXY_TO_IDX[AAMI_TO_PROXY[AAMI_CLASSES[int(i)]]] for i in y_aami]


def ptbxl_idx_to_proxy_idx(y_ptb) -> list[int]:
    return [PROXY_TO_IDX[PTBXL_TO_PROXY[PTBXL_CLASSES[int(i)]]] for i in y_ptb]


def mapping_report() -> str:
    lines = [
        "MIT-BIH AAMI -> proxy:",
        *[f"  {k} -> {v}" for k, v in AAMI_TO_PROXY.items()],
        "PTB-XL superclass -> proxy:",
        *[f"  {k} -> {v}" for k, v in PTBXL_TO_PROXY.items()],
        "",
        "Limitation: beat-level arrhythmia classes cannot be mapped 1:1 onto",
        "PTB-XL diagnostic superclasses; evaluation is binary normal/abnormal proxy.",
    ]
    return "\n".join(lines)
