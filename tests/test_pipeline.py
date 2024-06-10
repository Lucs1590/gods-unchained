import os
import sys
import unittest
import argparse

from unittest.mock import patch

sys.path.append(os.path.abspath(os.path.join('src')))  # noqa

from pipeline import main


class TestPipeline(unittest.TestCase):
    @patch('src.pipeline.argparse.ArgumentParser.parse_args')
    @patch('src.pipeline.importlib.import_module')
    def test_main_preprocessing(self, mock_import_module, mock_parse_args):
        mock_parse_args.return_value = argparse.Namespace(step='preprocessing')
        mock_step_module = mock_import_module.return_value
        mock_step_module.run.return_value = None
        main()
        mock_parse_args.assert_called_once()
        mock_import_module.assert_called_once_with('preprocessing')
        mock_step_module.run.assert_called_once()

    @patch('src.pipeline.argparse.ArgumentParser.parse_args')
    @patch('src.pipeline.importlib.import_module')
    def test_main_feature_engineering(self, mock_import_module, mock_parse_args):
        mock_parse_args.return_value = argparse.Namespace(
            step='feature_engineering')
        mock_step_module = mock_import_module.return_value
        mock_step_module.run.return_value = None
        main()
        mock_parse_args.assert_called_once()
        mock_import_module.assert_called_once_with('feature_engineering')
        mock_step_module.run.assert_called_once()

    @patch('src.pipeline.argparse.ArgumentParser.parse_args')
    @patch('src.pipeline.importlib.import_module')
    def test_main_model_building(self, mock_import_module, mock_parse_args):
        mock_parse_args.return_value = argparse.Namespace(
            step='model_building'
        )
        mock_step_module = mock_import_module.return_value
        mock_step_module.run.return_value = None
        main()
        mock_parse_args.assert_called_once()
        mock_import_module.assert_called_once_with('model_building')
        mock_step_module.run.assert_called_once()


if __name__ == '__main__':
    unittest.main()
