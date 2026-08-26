import unittest

from utils.walk_forward_validation import expanding_window_splits


class WalkForwardSplitTests(unittest.TestCase):
    def test_splits_are_chronological_gapped_and_non_overlapping(self):
        splits = expanding_window_splits(1000, n_folds=3, test_rows=100, gap_rows=6, minimum_train_rows=200)
        self.assertEqual(len(splits), 3)
        for split in splits:
            self.assertLessEqual(split["train_end"], split["gap_start"])
            self.assertEqual(split["gap_end"], split["test_start"])
            self.assertEqual(split["test_end"] - split["test_start"], 100)
            self.assertEqual(split["test_start"] - split["train_end"], 6)
        self.assertGreater(splits[1]["train_end"], splits[0]["train_end"])

    def test_insufficient_history_fails_closed(self):
        with self.assertRaises(ValueError):
            expanding_window_splits(300, n_folds=3, test_rows=100, gap_rows=6, minimum_train_rows=200)


if __name__ == "__main__":
    unittest.main()
