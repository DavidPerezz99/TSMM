import unittest

import numpy as np
import pandas as pd

from models.multivariate_models import TORCH_IMPORT_ERROR, train_nbeats_model


@unittest.skipIf(TORCH_IMPORT_ERROR is not None, "PyTorch is unavailable")
class NBeatsValidationTests(unittest.TestCase):
    def test_scalers_fit_training_period_only(self):
        rows = 180
        frame = pd.DataFrame({
            "HIGH": np.arange(rows, dtype=float),
            "y_diff": np.concatenate([np.zeros(145), np.full(rows - 145, 1000.0)]),
        })
        result = train_nbeats_model(
            frame, ["HIGH"], ["y_diff"], 5, 1, model_type="blackbox",
            hidden_size=8, epochs=1, batch_size=16,
            blackbox_config={"num_blocks": 1, "num_layers": 1},
            test_size=20, validation_size=15, random_seed=7,
        )
        self.assertLess(float(result["scalers"]["y"].mean_[0]), 1.0)
        self.assertEqual(result["parameters"]["validation_policy"], "chronological_train_only_scaling")


if __name__ == "__main__":
    unittest.main()
