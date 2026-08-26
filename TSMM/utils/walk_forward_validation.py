"""Chronological expanding-window split helpers for model validation."""

from __future__ import annotations

from typing import Dict, List


def expanding_window_splits(
    total_rows: int, n_folds: int = 3, test_rows: int = 60,
    gap_rows: int = 1, minimum_train_rows: int = 200,
) -> List[Dict[str, int]]:
    """Return ordered train/gap/test slices ending at the latest observation."""
    total_rows = int(total_rows)
    n_folds = max(int(n_folds), 1)
    test_rows = max(int(test_rows), 2)
    gap_rows = max(int(gap_rows), 0)
    required = minimum_train_rows + n_folds * test_rows + gap_rows
    if total_rows < required:
        raise ValueError(f"Need at least {required} rows for walk-forward validation; got {total_rows}")
    first_test_start = total_rows - n_folds * test_rows
    splits = []
    for fold in range(n_folds):
        test_start = first_test_start + fold * test_rows
        train_end = test_start - gap_rows
        splits.append({
            "fold": fold + 1,
            "train_start": 0,
            "train_end": train_end,
            "gap_start": train_end,
            "gap_end": test_start,
            "test_start": test_start,
            "test_end": test_start + test_rows,
        })
    return splits
