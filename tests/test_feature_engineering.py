import unittest
from unittest import mock
from unittest.mock import patch

import random
import numpy as np
import pandas as pd
import src.feature_engineering as fe


class TestFeatureEngineering(unittest.TestCase):
    def setUp(self) -> None:
        self.dataframe = pd.read_csv("data/train.csv")
        self.dataframe = self.dataframe.drop(columns=['id'])
        self.y = self.dataframe.pop('strategy')

    @patch("src.feature_engineering.pd.read_parquet")
    def test_load_preprocessed_data(self, mock_read_parquet):
        mock_read_parquet.return_value = self.dataframe

        dataframe = fe.load_preprocessed_data()
        self.assertIsInstance(dataframe, pd.DataFrame)

    def test_calculate_log_transformations(self):
        random_index = random.randint(0, len(self.dataframe))
        first_attack = self.dataframe.copy().iloc[random_index]['attack']
        dataframe = fe.calculate_log_transformations(self.dataframe)
        first_attack_transformed = dataframe.iloc[random_index]['attack']

        if first_attack > 0:
            self.assertNotEqual(first_attack, first_attack_transformed)
        else:
            self.assertEqual(first_attack, first_attack_transformed)

    def test_calculate_combined_features(self):
        dataframe = fe.calculate_combined_features(self.dataframe)
        self.assertIn('attack_mana', dataframe.columns)
        self.assertIn('attack_health', dataframe.columns)
        self.assertIn('mana_health', dataframe.columns)
        self.assertIn('attack_mana_health', dataframe.columns)

    def test_calculate_comparison_features(self):
        dataframe = fe.calculate_comparison_features(self.dataframe)
        self.assertIn('att_greater_5', dataframe.columns)
        self.assertIn('mana_greater_7', dataframe.columns)
        self.assertIn('health_greater_6', dataframe.columns)
        self.assertIn('att_greater_mana', dataframe.columns)
        self.assertIn('att_greater_health', dataframe.columns)
        self.assertIn('mana_greater_health', dataframe.columns)

    def test_encode_categorical_features(self):
        dataframe = fe.encode_categorical_features(self.dataframe)
        self.assertGreater(len(dataframe.columns), len(self.dataframe.columns))

    def test_impute_missing_values(self):
        random_index = random.randint(0, len(self.dataframe))
        dataframe = self.dataframe.copy()

        dataframe = fe.calculate_combined_features(dataframe)
        dataframe.iloc[random_index]['attack'] = np.nan
        dataframe = fe.impute_missing_values(dataframe)
        self.assertFalse(dataframe['attack'].isnull().values.any())

    def test_select_features(self):
        dataframe = self.dataframe.copy()
        dataframe = fe.calculate_combined_features(dataframe)
        dataframe = fe.calculate_comparison_features(dataframe)
        dataframe = fe.encode_categorical_features(dataframe)
        dataframe = fe.impute_missing_values(dataframe)
        dataframe.drop(columns=['name'], inplace=True)

        X, y = dataframe, self.y
        X = fe.select_features(X, y)
        self.assertIsInstance(X[1], pd.Series)
        self.assertIsInstance(y, pd.Series)

    @patch("src.feature_engineering.os.makedirs")
    @patch("src.feature_engineering.pd.DataFrame.to_parquet")
    @patch("builtins.open", create=True)
    @patch("matplotlib.pyplot.figure")
    @patch("matplotlib.pyplot.bar")
    def test_save_artifacts(self, mock_bar, mock_figure, mock_open, mock_to_parquet, mock_makedirs):
        dataframe = pd.DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6]})
        columns = ['col1', 'col2']
        mutual_info = pd.Series([0.5, 0.3], index=columns)

        fe.save_artifacts(dataframe, columns, mutual_info)

        mock_makedirs.assert_called_once_with("artifacts", exist_ok=True)
        mock_to_parquet.assert_called_once_with(
            "artifacts/train_dataframe_engineered.parquet",
            index=False
        )

        expected_calls = [mock.call("artifacts/selected_columns.txt", "w")]
        expected_calls += [mock.call().__enter__().write(col + "\n")
                           for col in columns]
        expected_calls += [mock.call().__exit__(None, None, None)]
        mock_open.assert_has_calls(expected_calls, any_order=True)

        mock_bar.assert_called_once_with(mutual_info.index, mutual_info)
        mock_figure().savefig.assert_called_once_with(
            "artifacts/feature_importance_engineered.png"
        )

    @patch("src.feature_engineering.logger")
    @patch("src.feature_engineering.load_preprocessed_data")
    @patch("src.feature_engineering.calculate_log_transformations")
    @patch("src.feature_engineering.calculate_combined_features")
    @patch("src.feature_engineering.calculate_comparison_features")
    @patch("src.feature_engineering.encode_categorical_features")
    @patch("src.feature_engineering.impute_missing_values")
    @patch("src.feature_engineering.pd.DataFrame.drop")
    @patch("src.feature_engineering.select_features")
    @patch("src.feature_engineering.save_artifacts")
    @patch("src.feature_engineering.sns")
    @patch("src.feature_engineering.plt")
    def test_run(
        self, mock_plt, mock_sns,
        mock_save_artifacts, mock_select_features, mock_drop, mock_impute_missing_values,
        mock_encode_categorical_features, mock_calculate_comparison_features,
        mock_calculate_combined_features, mock_calculate_log_transformations,
        mock_load_preprocessed_data, mock_logger
    ):
        dataframe = self.dataframe.copy()
        dataframe['strategy'] = self.y
        dataframe.drop(columns=['name, type, god'], inplace=True)

        mock_plt.figure = mock.MagicMock()
        mock_plt.title = mock.MagicMock()
        mock_plt.savefig = mock.MagicMock()
        mock_sns.heatmap = mock.MagicMock()
        mock_load_preprocessed_data.return_value = dataframe
        mock_calculate_log_transformations.return_value = dataframe
        mock_calculate_combined_features.return_value = dataframe
        mock_calculate_comparison_features.return_value = dataframe
        mock_encode_categorical_features.return_value = dataframe
        mock_drop.return_value = dataframe
        mock_impute_missing_values.return_value = dataframe
        mock_select_features.return_value = (
            ['mana', 'attack', 'health'],
            pd.Series([0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05],
                      index=dataframe.columns)
        )
        fe.run()

        mock_load_preprocessed_data.assert_called_once()
        mock_calculate_log_transformations.assert_called_once_with(dataframe)
        mock_calculate_combined_features.assert_called_once_with(dataframe)
        mock_calculate_comparison_features.assert_called_once_with(dataframe)
        mock_encode_categorical_features.assert_called_once_with(dataframe)
        mock_impute_missing_values.assert_called_once_with(dataframe)
        mock_logger.info.assert_called()


if __name__ == '__main__':
    unittest.main()
