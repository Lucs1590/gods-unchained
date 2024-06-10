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
        cols = ['mana', 'attack', 'health']
        test_df = model_building.prepare_test_data(
            test_df,
            cols
        )

        self.assertEqual(len(test_df.columns), len(cols))

    @patch("model_building.accuracy_score")
    @patch("model_building.ks_score")
    @patch("model_building.f1_score")
    @patch("model_building.precision_score")
    def test_train_and_evaluate_models(self, mock_precision_score, mock_f1_score, mock_ks_score, mock_accuracy_score):
        mock_accuracy_score.return_value = 1.0
        mock_ks_score.return_value = 0.0
        mock_f1_score.return_value = 1.0
        mock_precision_score.return_value = 1.0

        X_train, X_test, y_train, y_test = self.dataframe, self.dataframe, self.y, self.y

        _, best_model_ks, best_model = model_building.train_and_evaluate_models(
            X_train[['mana', 'health', 'attack']],
            X_test[['mana', 'health', 'attack']],
            y_train,
            y_test.apply(lambda x: 1 if x == 'late' else 0)
        )

        self.assertEqual(best_model_ks, 0.0)
        self.assertIsNotNone(best_model)

    @patch("model_building.joblib.dump")
    @patch("model_building.pd.read_csv")
    @patch("model_building.pd.DataFrame.to_parquet")
    @patch("model_building.pd.DataFrame.to_csv")
    def test_save_results(self, mock_joblib_dump, mock_read_csv, mock_read_parquet, mock_to_csv):
        mock_joblib_dump.return_value = None
        mock_read_csv.return_value = self.dataframe
        mock_read_csv.return_value = self.dataframe
        mock_read_parquet.return_value = self.dataframe
        predictions = np.random.randint(0, 1, self.dataframe.shape[0])
        predictions = np.where(predictions > 0.5, True, False)

        model_building.save_results(
            1,
            predictions,
            "data"
        )

        self.assertTrue(mock_joblib_dump.called)
        self.assertTrue(mock_read_csv.called)
        self.assertTrue(mock_read_parquet.called)
        self.assertTrue(mock_to_csv.called)


if __name__ == '__main__':
    unittest.main()
