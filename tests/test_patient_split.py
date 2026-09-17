"""Unit tests for patient-wise splitting (no subject leakage)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from heartx.data.patient_split import (  # noqa: E402
    masks_from_split,
    patient_wise_split,
)


def test_no_group_overlap_fixed_test():
    rng = np.random.default_rng(0)
    groups = np.array([f"p{i // 10}" for i in range(100)])
    y = rng.integers(0, 5, size=100)
    test_groups = [f"p{i}" for i in range(7, 10)]
    split = patient_wise_split(y, groups, test_groups=test_groups, val_ratio=0.2, seed=0)
    split.assert_no_leakage()
    assert set(map(str, split.test_groups)).issubset(set(test_groups))
    assert set(map(str, split.train_groups)).isdisjoint(set(test_groups))
    assert set(map(str, split.val_groups)).isdisjoint(set(test_groups))


def test_no_group_overlap_random():
    rng = np.random.default_rng(1)
    groups = np.array([f"g{i // 5}" for i in range(200)])
    y = rng.integers(0, 3, size=200)
    split = patient_wise_split(y, groups, test_groups=None, val_ratio=0.15, test_ratio=0.2, seed=1)
    split.assert_no_leakage()
    n = len(y)
    tr, va, te = masks_from_split(n, split)
    assert tr.sum() + va.sum() + te.sum() == n
    assert not (tr & va).any()
    assert not (tr & te).any()
    assert not (va & te).any()


def test_masks_cover_all_once():
    groups = np.array(["a"] * 20 + ["b"] * 20 + ["c"] * 20 + ["d"] * 20)
    y = np.zeros(80, dtype=int)
    y[40:] = 1
    split = patient_wise_split(
        y, groups, test_groups=["d"], val_ratio=0.5, seed=2
    )
    tr, va, te = masks_from_split(80, split)
    assert (tr.astype(int) + va.astype(int) + te.astype(int) == 1).all()
