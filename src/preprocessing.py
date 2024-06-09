import os
import sys
import logging

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.stats import normaltest

logger = logging.getLogger()
handler = logging.StreamHandler(sys.stdout)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


def to_bool(dataframe: pd.DataFrame, col: str, conditional: str) -> pd.DataFrame:
    dataframe[col] = dataframe[col] == conditional
    return dataframe


def run():
    _path = os.path.abspath(os.path.dirname(os.path.abspath(os.getcwd())))
    train_dataframe = pd.read_csv(_path + '/data/train.csv')

    train_dataframe = to_bool(train_dataframe, 'strategy', 'late')

    boolean_cols = [
        col for col in train_dataframe.columns if train_dataframe[col].dtypes == 'bool'
    ]
    not_cat_cols = [
        col for col in train_dataframe.columns if train_dataframe[col].dtypes !=
        'O' and train_dataframe[col].dtypes != 'bool'
    ]

    corr_matrix = train_dataframe[not_cat_cols + boolean_cols].corr()

    logger.info(
        f'Attack distribution is normal? {normaltest(train_dataframe["attack"]).pvalue < 0.05}'
    )
    logger.info(
        f'Mana distribution is normal? {normaltest(train_dataframe["mana"]).pvalue < 0.05}'
    )
    logger.info(
        f'Health distribution is normal? {normaltest(train_dataframe["health"]).pvalue < 0.05}'
    )
    logger.info(
        f'Attack and Mana distribution is normal? {normaltest(train_dataframe["attack"] + train_dataframe["mana"]).pvalue < 0.05}'
    )
    logger.info(
        f'Attack and Health distribution is normal? {normaltest(train_dataframe["attack"] + train_dataframe["health"]).pvalue < 0.05}'
    )
    logger.info(
        f'Mana and Health distribution is normal? {normaltest(train_dataframe["mana"] + train_dataframe["health"]).pvalue < 0.05}'
    )
    logger.info(
        f'Attack, Mana and Health distribution is normal? {normaltest(train_dataframe["attack"] + train_dataframe["mana"] + train_dataframe["health"]).pvalue < 0.05}'
    )

    # Save artifacts
    os.makedirs("artifa cts", exist_ok=True)
    train_dataframe.to_parquet(
        "artifacts/train_dataframe.parquet",
        index=False
    )
    corr_matrix.to_parquet(
        "artifacts/corr_matrix.parquet",
        index=False
    )
    sns.heatmap(corr_matrix, annot=True, cmap='Blues', fmt=".2f")
    plt.savefig("artifacts/corr_matrix.png")
