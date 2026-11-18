"""Selection rule for the locked DS1 comparison. DS2 is not an input."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def _load():
    path = Path(__file__).resolve().parents[1] / "scripts" / "select_final_model.py"
    spec = importlib.util.spec_from_file_location("select_final_model", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_tie_prefers_lower_variance():
    mod = _load()
    rows = [
        {"name": "a", "f1_mean": 0.50, "f1_std": 0.04},
        {"name": "b", "f1_mean": 0.505, "f1_std": 0.02},
        {"name": "c", "f1_mean": 0.40, "f1_std": 0.00},
    ]
    assert mod.choose_winner(rows)["name"] == "b"


def test_clear_leader_ignores_variance():
    mod = _load()
    rows = [
        {"name": "stable", "f1_mean": 0.40, "f1_std": 0.001},
        {"name": "better", "f1_mean": 0.50, "f1_std": 0.03},
    ]
    assert mod.choose_winner(rows)["name"] == "better"
