import os
import sys
import logging

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.feature_selection import mutual_info_classif

logger = logging.getLogger()
handler = logging.StreamHandler(sys.stdout)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


def run():
    train_dataframe = pd.read_parquet("artifacts/train_dataframe.parquet")
    _path = os.path.abspath(os.path.dirname(os.path.abspath(os.getcwd())))

    train_dataframe['attack'] = train_dataframe['attack'].apply(
        lambda x: np.log(x) if x > 0 else 0
    )
    train_dataframe['mana'] = train_dataframe['mana'].apply(
        lambda x: np.log(x) if x > 0 else 0
    )
    train_dataframe['health'] = train_dataframe['health'].apply(
        lambda x: np.log(x) if x > 0 else 0
    )
    train_dataframe['attack_mana'] = train_dataframe['attack'] + \
        train_dataframe['mana']
    train_dataframe['attack_health'] = train_dataframe['attack'] + \
        train_dataframe['health']
    train_dataframe['mana_health'] = train_dataframe['mana'] + \
        train_dataframe['health']
    train_dataframe['attack_mana_health'] = train_dataframe['attack'] + \
        train_dataframe['mana'] + train_dataframe['health']
    train_dataframe['att_greater_5'] = train_dataframe['attack'] > 5
    train_dataframe['mana_greater_7'] = train_dataframe['mana'] > 7
    train_dataframe['health_greater_6'] = train_dataframe['health'] > 5
    train_dataframe['att_greater_mana'] = train_dataframe['attack'] > train_dataframe['mana']
    train_dataframe['att_greater_health'] = train_dataframe['attack'] > train_dataframe['health']
    train_dataframe['mana_greater_health'] = train_dataframe['mana'] > train_dataframe['health']

    train_dataframe['att_greater_5'] = train_dataframe['att_greater_5'].astype(
        int
    )
    train_dataframe['mana_greater_7'] = train_dataframe['mana_greater_7'].astype(
        int
    )
    train_dataframe['health_greater_6'] = train_dataframe['health_greater_6'].astype(
        int
    )
    train_dataframe['att_greater_mana'] = train_dataframe['att_greater_mana'].astype(
        int
    )
    train_dataframe['att_greater_health'] = train_dataframe['att_greater_health'].astype(
        int
    )
    train_dataframe['mana_greater_health'] = train_dataframe['mana_greater_health'].astype(
        int
    )

    train_dataframe = pd.get_dummies(
        train_dataframe,
        columns=['god', 'type'],
        drop_first=True,
        prefix=['god', 'type'],
        prefix_sep='_'
    )

    train_dataframe.drop('name', axis=1, inplace=True)
    corr_matrix = train_dataframe.corr()
    sns.heatmap(corr_matrix, cmap='Blues', fmt=".2f")
    plt.savefig("artifacts/corr_matrix_engineered.png")

    train_dataframe.replace([np.inf, -np.inf], np.nan, inplace=True)
    train_dataframe['attack'] = train_dataframe['attack'].fillna(
        train_dataframe['attack'].mean())
    train_dataframe['mana'] = train_dataframe['mana'].fillna(
        train_dataframe['mana'].mean())
    train_dataframe['health'] = train_dataframe['health'].fillna(
        train_dataframe['health'].mean())
    train_dataframe['attack_mana'] = train_dataframe['attack_mana'].fillna(
        train_dataframe['attack_mana'].mean())
    train_dataframe['attack_health'] = train_dataframe['attack_health'].fillna(
        train_dataframe['attack_health'].mean())
    train_dataframe['mana_health'] = train_dataframe['mana_health'].fillna(
        train_dataframe['mana_health'].mean())
    train_dataframe['attack_mana_health'] = train_dataframe['attack_mana_health'].fillna(
        train_dataframe['attack_mana_health'].mean())

    X = train_dataframe.drop(['strategy', 'id'], axis=1)
    y = train_dataframe['strategy']

    mutual_info = mutual_info_classif(X, y)
    mutual_info = pd.Series(mutual_info, index=X.columns)
    mutual_info = mutual_info[mutual_info > 0]
    mutual_info = mutual_info.sort_values(ascending=False)

    plt.figure(figsize=(30, 6))
    plt.bar(mutual_info.index, mutual_info)
    plt.xlabel('Feature')
    plt.ylabel('Score')
    plt.title('Feature Importance')
    plt.savefig("artifacts/feature_importance_engineered.png")

    selected_columns = list(mutual_info.index[:7]) + ['strategy', 'id']
    train_dataframe = train_dataframe[selected_columns]

    train_dataframe.to_parquet(
        "artifacts/train_dataframe_engineered.parquet",
        index=False
    )

    with open("artifacts/selected_columns.txt", "w") as f:
        for col in selected_columns:
            f.write(col + "\n")
