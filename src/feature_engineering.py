import os
import logging

import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_classif

logger = logging.getLogger('gods_unchained')


def load_preprocessed_data() -> pd.DataFrame:
    logger.info("Loading preprocessed data...")
    return pd.read_parquet("artifacts/train_dataframe.parquet")


def calculate_log_transformations(dataframe: pd.DataFrame) -> pd.DataFrame:
    logger.info("Calculating log transformations...")
    for col in ['attack', 'mana', 'health']:
        dataframe[col] = dataframe[col].apply(
            lambda x: np.log(x) if x > 0 else 0)
    return dataframe


def calculate_combined_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    logger.info("Calculating combined features...")
    dataframe['attack_mana'] = dataframe['attack'] + dataframe['mana']
    dataframe['attack_health'] = dataframe['attack'] + dataframe['health']
    dataframe['mana_health'] = dataframe['mana'] + dataframe['health']
    dataframe['attack_mana_health'] = dataframe['attack'] + \
        dataframe['mana'] + dataframe['health']
    return dataframe


def calculate_comparison_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    logger.info("Calculating comparison features...")
    dataframe['att_greater_5'] = (dataframe['attack'] > 5).astype(int)
    dataframe['mana_greater_7'] = (dataframe['mana'] > 7).astype(int)
    dataframe['health_greater_6'] = (dataframe['health'] > 5).astype(int)
    dataframe['att_greater_mana'] = (
        dataframe['attack'] > dataframe['mana']).astype(int)
    dataframe['att_greater_health'] = (
        dataframe['attack'] > dataframe['health']).astype(int)
    dataframe['mana_greater_health'] = (
        dataframe['mana'] > dataframe['health']).astype(int)
    return dataframe


def encode_categorical_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    logger.info("Encoding categorical features...")
    return pd.get_dummies(
        dataframe,
        columns=['god', 'type'],
        drop_first=True,
        prefix=['god', 'type'],
        prefix_sep='_'
    )


def impute_missing_values(dataframe: pd.DataFrame) -> pd.DataFrame:
    logger.info("Imputing missing values...")
    for col in ['attack', 'mana', 'health', 'attack_mana', 'attack_health', 'mana_health', 'attack_mana_health']:
        dataframe[col] = dataframe[col].fillna(dataframe[col].mean())
    return dataframe


def select_features(X: pd.DataFrame, y: pd.Series) -> tuple:
    logger.info("Selecting features...")
    mutual_info = mutual_info_classif(X, y)
    mutual_info = pd.Series(mutual_info, index=X.columns)
    mutual_info = mutual_info[mutual_info > 0].sort_values(ascending=False)
    selected_columns = list(mutual_info.index[:7]) + ['strategy', 'id']
    return selected_columns, mutual_info


def save_artifacts(dataframe: pd.DataFrame, columns: list, mutual_info: pd.Series):
    logger.info("Saving artifacts...")
    os.makedirs("artifacts", exist_ok=True)
    dataframe.to_parquet(
        "artifacts/train_dataframe_engineered.parquet", index=False)

    with open("artifacts/selected_columns.txt", "w") as f:
        for col in columns:
            f.write(col + "\n")

    plt.figure(figsize=(30, 6))
    plt.bar(mutual_info.index, mutual_info)
    plt.xlabel('Feature')
    plt.ylabel('Score')
    plt.title('Feature Importance')
    plt.savefig("artifacts/feature_importance_engineered.png")
    plt.close()


def run():
    logger.info("Starting feature engineering step")

    train_dataframe = load_preprocessed_data()
    train_dataframe.sample(
        frac=1,
        random_state=237
    ).reset_index(drop=True)

    train_dataframe = calculate_log_transformations(train_dataframe)
    train_dataframe = calculate_combined_features(train_dataframe)
    train_dataframe = calculate_comparison_features(train_dataframe)
    train_dataframe = encode_categorical_features(train_dataframe)
    train_dataframe = impute_missing_values(train_dataframe)

    train_dataframe.drop('name', axis=1, inplace=True)

    X = train_dataframe.drop(['strategy', 'id'], axis=1)
    y = train_dataframe['strategy']
    selected_columns, mutual_info = select_features(X, y)
    train_dataframe = train_dataframe[selected_columns]

    corr_matrix_engineered = train_dataframe.corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_matrix_engineered, cmap='Blues', fmt=".2f")
    plt.title("Correlation Matrix")
    plt.savefig("artifacts/corr_matrix_engineered.png")
    save_artifacts(train_dataframe, selected_columns, mutual_info)

    logger.info("Feature engineering completed successfully!")
