import os
import sys
import unittest
from unittest.mock import patch

import pandas as pd

sys.path.append(os.path.abspath(os.path.join('src')))  # noqa

import preprocessing


class TestPreprocessing(unittest.TestCase):
    def setUp(self) -> None:
        self.dataframe = pd.read_csv("data/train.csv")
        self.y = self.dataframe['strategy']

    @patch("src.preprocessing.pd.read_csv")
    def test_load_data(self, mock_read_csv):
        mock_read_csv.return_value = self.dataframe

        dataframe = preprocessing.load_data("data/train.csv")
        self.assertIsInstance(dataframe, pd.DataFrame)

    def test_to_bool(self):
        dataframe = preprocessing.to_bool(self.dataframe, 'strategy', 'late')

        self.assertTrue(dataframe['strategy'].dtype == 'bool')

    def test_identify_column_types(self):
        boolean_cols, non_cat_cols = preprocessing.identify_column_types(
            self.dataframe
        )
        self.assertTrue(len(boolean_cols) == 0)
        self.assertTrue(len(non_cat_cols) == 4)

    def test_analyze_distributions(self):
        corr_matrix = preprocessing.analyze_distributions(self.dataframe)
        self.assertIsInstance(corr_matrix, pd.DataFrame)

    @patch("src.preprocessing.pd.DataFrame.to_csv")
    @patch("src.preprocessing.pd.DataFrame.to_parquet")
    @patch("src.preprocessing.os.makedirs")
    @patch("src.preprocessing.sns.heatmap")
    @patch("src.preprocessing.plt")
    def test_save_artifacts(self, mock_plt, mock_heatmap, mock_makedirs,
                            mock_to_parquet, mock_to_csv):
        mock_plt.show = lambda: None
        mock_heatmap = lambda *args, **kwargs: None

        dataframe = self.dataframe[['mana', 'attack', 'health']]
        preprocessing.save_artifacts(
            dataframe,
            dataframe.corr()
        )

        mock_makedirs.assert_called_once_with("artifacts", exist_ok=True)


if __name__ == '__main__':
    unittest.main()
