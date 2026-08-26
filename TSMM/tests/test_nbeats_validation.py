import unittest

import numpy as np
import pandas as pd

from models.multivariate_models import TORCH_IMPORT_ERROR, train_nbeats_model


@unittest.skipIf(TORCH_IMPORT_ERROR is not None, "PyTorch is unavailable")
class NBeatsValidationTests(unittest.TestCase):
    def test_scalers_fit_training_period_not_future_validation_or_holdout(self):
        rows = 180
        frame = pd.DataFrame({
            "HIGH": np.arange(rows, dtype=float),
            "y_diff": np.concatenate([np.zeros(145), np.full(rows - 145, 1000.0)]),
        })
        result = train_nbeats_model(
            frame, ["HIGH"], ["y_diff"], n_steps=5, m_steps=1,
            split_ratio=0.8, model_type="blackbox", hidden_size=8, epochs=1,
            batch_size=16, blackbox_config={"num_blocks": 1, "num_layers": 1},
            test_size=20, validation_size=15, random_seed=7,
        )
        self.assertNotIn("error", result)
        self.assertLess(float(result["scalers"]["y"].mean_[0]), 1.0)
        self.assertEqual(result["parameters"]["validation_policy"], "chronological_train_only_scaling")
        self.assertEqual(result["parameters"]["test_size"], 20)


if __name__ == "__main__":
    unittest.main()
