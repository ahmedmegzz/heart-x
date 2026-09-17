"""Patient / record-wise train-val-test splits (no subject leakage)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import GroupShuffleSplit


@dataclass
class SplitResult:
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    train_groups: np.ndarray
    val_groups: np.ndarray
    test_groups: np.ndarray

    def assert_no_leakage(self) -> None:
        tr = set(map(str, self.train_groups))
        va = set(map(str, self.val_groups))
        te = set(map(str, self.test_groups))
        assert tr.isdisjoint(va), f"train/val group overlap: {tr & va}"
        assert tr.isdisjoint(te), f"train/test group overlap: {tr & te}"
        assert va.isdisjoint(te), f"val/test group overlap: {va & te}"

    def summary(self) -> dict:
        return {
            "n_train": int(len(self.train_idx)),
            "n_val": int(len(self.val_idx)),
            "n_test": int(len(self.test_idx)),
            "n_train_groups": int(len(np.unique(self.train_groups))),
            "n_val_groups": int(len(np.unique(self.val_groups))),
            "n_test_groups": int(len(np.unique(self.test_groups))),
        }


def _group_label(y: np.ndarray, groups: np.ndarray) -> dict[str, int]:
    """Majority class label per group (for stratified group split)."""
    out: dict[str, int] = {}
    for g in np.unique(groups):
        mask = groups == g
        counts = np.bincount(y[mask].astype(int))
        out[str(g)] = int(counts.argmax()) if len(counts) else 0
    return out


def patient_wise_split(
    y: np.ndarray,
    groups: np.ndarray,
    test_groups: np.ndarray | list | set | None = None,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    ensure_test_classes: bool = True,
) -> SplitResult:
    """
    Split samples by patient/record group IDs.

    If ``test_groups`` is provided (e.g. MIT-BIH DS2 record IDs), those groups
    form the test set and the remaining groups are split into train/val with
    GroupShuffleSplit. Otherwise a two-stage group split is used.
    """
    y = np.asarray(y)
    groups = np.asarray(groups)
    n = len(y)
    if len(groups) != n:
        raise ValueError("y and groups must have the same length")

    idx_all = np.arange(n)

    if test_groups is not None:
        test_set = {str(g) for g in test_groups}
        test_mask = np.array([str(g) in test_set for g in groups])
        trainval_mask = ~test_mask
        test_idx = idx_all[test_mask]
        tv_idx = idx_all[trainval_mask]
        tv_groups = groups[trainval_mask]
        tv_y = y[trainval_mask]

        if len(np.unique(tv_groups)) < 2:
            raise ValueError("Need >= 2 train/val groups after fixing test groups")

        # Relative val fraction among train+val groups
        gss = GroupShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed)
        tr_rel, va_rel = next(gss.split(tv_idx, tv_y, groups=tv_groups))
        train_idx = tv_idx[tr_rel]
        val_idx = tv_idx[va_rel]
    else:
        # Stage 1: hold out test groups
        gss_test = GroupShuffleSplit(
            n_splits=1, test_size=test_ratio, random_state=seed
        )
        tv_rel, te_rel = next(gss_test.split(idx_all, y, groups=groups))
        test_idx = idx_all[te_rel]
        tv_idx = idx_all[tv_rel]
        tv_groups = groups[tv_rel]
        tv_y = y[tv_rel]

        # Stage 2: val from remaining
        val_of_tv = val_ratio / max(1e-8, (1.0 - test_ratio))
        val_of_tv = min(max(val_of_tv, 0.05), 0.5)
        gss_val = GroupShuffleSplit(
            n_splits=1, test_size=val_of_tv, random_state=seed + 1
        )
        tr_rel, va_rel = next(gss_val.split(tv_idx, tv_y, groups=tv_groups))
        train_idx = tv_idx[tr_rel]
        val_idx = tv_idx[va_rel]

    result = SplitResult(
        train_idx=np.sort(train_idx),
        val_idx=np.sort(val_idx),
        test_idx=np.sort(test_idx),
        train_groups=groups[train_idx],
        val_groups=groups[val_idx],
        test_groups=groups[test_idx],
    )
    result.assert_no_leakage()

    if ensure_test_classes:
        present_all = set(np.unique(y).tolist())
        present_test = set(np.unique(y[result.test_idx]).tolist())
        missing = present_all - present_test
        if missing:
            # Documented fallback: keep split, caller should log the gap.
            # We do not move patients automatically (would break fixed DS2).
            result._missing_test_classes = sorted(missing)  # type: ignore[attr-defined]
        else:
            result._missing_test_classes = []  # type: ignore[attr-defined]

    return result


def masks_from_split(n: int, split: SplitResult) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_mask = np.zeros(n, dtype=bool)
    val_mask = np.zeros(n, dtype=bool)
    test_mask = np.zeros(n, dtype=bool)
    train_mask[split.train_idx] = True
    val_mask[split.val_idx] = True
    test_mask[split.test_idx] = True
    return train_mask, val_mask, test_mask


def log_split_summary(name: str, split: SplitResult, groups: np.ndarray) -> None:
    s = split.summary()
    print(f"=== Split summary ({name}) ===", flush=True)
    print(
        f"  train: {s['n_train']} samples / {s['n_train_groups']} groups  "
        f"val: {s['n_val']} / {s['n_val_groups']}  "
        f"test: {s['n_test']} / {s['n_test_groups']}",
        flush=True,
    )
    missing = getattr(split, "_missing_test_classes", None)
    if missing:
        print(
            f"  WARNING: test set missing class indices {missing}. "
            "Patient-wise constraint prevents automatic repair; report this in thesis.",
            flush=True,
        )
