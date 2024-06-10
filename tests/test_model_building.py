import os
import sys
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join('src')))  # noqa

import model_building


class TestModelBuilding(unittest.TestCase):
    def setUp(self) -> None:
        self.dataframe = pd.read_csv("data/train.csv")
        self.y = self.dataframe['strategy']

    def test_ks_score(self):
        test, pred = np.array([0, 1, 1, 0]), np.array([0, 1, 0, 1])

        ks = model_building.ks_score(
            test,
            pred
        )
        self.assertEqual(ks, 0.0)

    def test_get_best_model(self):
        results_metrics = [0.1, 0.2, 0.3]
        models_result = [1, 2, 3]
        names = ['a', 'b', 'c']

        best_model = model_building.get_best_model(
            results_metrics, models_result, names
        )
        self.assertEqual(len(best_model), 3)
        self.assertEqual(best_model[0], 'c')

    @patch("src.model_building.pd.read_csv")
    @patch("src.model_building.pd.read_parquet")
    def load_data(self, mock_read_csv, mock_read_parquet):
        mock_read_csv.return_value = self.dataframe
        mock_read_parquet.return_value = self.dataframe

        data = model_building.load_data("data/train.csv")
        self.assertIsInstance(data, pd.DataFrame)

    def test_prepare_test_data(self):
        test_df = self.dataframe.copy()
        test_df = model_building.prepare_test_data(test_df, test_df.columns)

        self.assertEqual(len(test_df.columns), len(self.dataframe.columns))


if __name__ == '__main__':
    unittest.main()
