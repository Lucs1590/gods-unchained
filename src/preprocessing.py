import os
import logging

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.stats import normaltest

logger = logging.getLogger('gods_unchained')


def load_data(data_path: str) -> pd.DataFrame:
    logger.info('Loading training data...')
    return pd.read_csv(data_path)


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    logger.info('Preprocessing data...')
    return to_bool(df, 'strategy', 'late')


def to_bool(dataframe: pd.DataFrame, col: str, conditional: str) -> pd.DataFrame:
    dataframe[col] = dataframe[col] == conditional
    return dataframe


def identify_column_types(dataframe: pd.DataFrame) -> tuple:
    boolean_cols = [
        col for col in dataframe.columns if dataframe[col].dtypes == 'bool'
    ]
    non_cat_cols = [
        col for col in dataframe.columns if dataframe[col].dtypes !=
        'O' and dataframe[col].dtypes != 'bool'
    ]
    return boolean_cols, non_cat_cols


def analyze_distributions(dataframe: pd.DataFrame) -> pd.DataFrame:
    logger.info('Analyzing distributions...')
    boolean_cols, non_cat_cols = identify_column_types(dataframe)
    corr_matrix = dataframe[non_cat_cols + boolean_cols].corr()

    for col in ['attack', 'mana', 'health']:
        is_normal = normaltest(dataframe[col]).pvalue >= 0.05
        logger.info(f'Distribution of "{col}" is normal? {is_normal}')

        for other_col in ['attack', 'mana', 'health']:
            if col != other_col:
                combined_col = dataframe[col] + dataframe[other_col]
                is_normal_combined = normaltest(combined_col).pvalue >= 0.05
                logger.info(
                    f'Combined distribution of "{col}" and "{other_col}" is normal? {is_normal_combined}'
                )

    return corr_matrix


def save_artifacts(dataframe: pd.DataFrame, corr_matrix: pd.DataFrame):
    logger.info('Saving artifacts...')
    os.makedirs("artifacts", exist_ok=True)
    dataframe.to_parquet("artifacts/train_dataframe.parquet", index=False)
    corr_matrix.to_parquet("artifacts/corr_matrix.parquet", index=False)

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='Blues', fmt=".2f")
    plt.savefig("artifacts/corr_matrix.png")
    plt.close()


def run():
    _path = os.path.abspath(os.getcwd())
    data_path = _path + '/data/train.csv'

    df = load_data(data_path)
    df = preprocess_data(df)

    corr_matrix = analyze_distributions(df)
    save_artifacts(df, corr_matrix)

    logger.info('Preprocessing completed successfully!')
