"""Unit tests for Chazal patient-wise MIT-BIH protocol constraints."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from heartx.config import (  # noqa: E402
    MITBIH_DS1,
    MITBIH_DS2,
    MITBIH_EXCLUDE_PACEMAKER,
)
from heartx.data.patient_split import patient_wise_split  # noqa: E402


def test_pacemaker_not_in_ds_lists():
    exclude = set(MITBIH_EXCLUDE_PACEMAKER)
    assert exclude == {"102", "104", "107", "217"}
    assert exclude.isdisjoint(set(MITBIH_DS1))
    assert exclude.isdisjoint(set(MITBIH_DS2))


def test_ds1_ds2_disjoint():
    assert set(MITBIH_DS1).isdisjoint(set(MITBIH_DS2))


def test_split_train_val_only_from_ds1_groups():
    """Synthetic beats labeled with DS1/DS2 record IDs."""
    rng = np.random.default_rng(0)
    groups = []
    y = []
    for r in MITBIH_DS1:
        n = 20
        groups.extend([r] * n)
        y.extend(rng.integers(0, 4, size=n).tolist())
    for r in MITBIH_DS2:
        n = 15
        groups.extend([r] * n)
        y.extend(rng.integers(0, 4, size=n).tolist())
    groups = np.array(groups)
    y = np.array(y)

    split = patient_wise_split(
        y, groups, test_groups=MITBIH_DS2, val_ratio=0.2, seed=0
    )
    split.assert_no_leakage()
    train_g = set(map(str, split.train_groups))
    val_g = set(map(str, split.val_groups))
    test_g = set(map(str, split.test_groups))
    assert train_g.issubset(set(MITBIH_DS1))
    assert val_g.issubset(set(MITBIH_DS1))
    assert test_g.issubset(set(MITBIH_DS2))
    assert train_g.isdisjoint(val_g)
