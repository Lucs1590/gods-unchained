import os
import logging

import joblib
import numpy as np
import pandas as pd

from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, roc_curve

logger = logging.getLogger('gods_unchained')


def ks_score(y_true, y_score):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return max(tpr - fpr)


def get_best_model(ks: list, models: list, names: list) -> list:
    name = names[np.argmax(ks)]
    ks = max(ks)
    model = models[np.argmax(ks)]
    logging.info('The best model is %s with KS of %s', name, ks)
    return [name, ks, model]


def load_data(data_path: str) -> pd.DataFrame:
    logger.info('Loading train and test data...')
    train_df = pd.read_parquet("artifacts/train_dataframe_engineered.parquet")
    test_df = pd.read_csv(data_path + '/test.csv')
    return train_df, test_df


def prepare_test_data(test_df: pd.DataFrame, train_columns: list) -> pd.DataFrame:
    logger.info('Preparing test data...')

    for col in ['attack', 'mana', 'health']:
        test_df[col] = test_df[col].apply(lambda x: np.log(x) if x > 0 else 0)

    test_df['attack_mana'] = test_df['attack'] + test_df['mana']
    test_df['attack_health'] = test_df['attack'] + test_df['health']
    test_df['mana_health'] = test_df['mana'] + test_df['health']
    test_df['attack_mana_health'] = test_df['attack'] + \
        test_df['mana'] + test_df['health']

    for col1, col2 in [('attack', 'mana'), ('attack', 'health'), ('mana', 'health')]:
        test_df[f'{col1}_greater_{col2}'] = (
            test_df[col1] > test_df[col2]
        ).astype(int)

    test_df = pd.get_dummies(
        test_df,
        columns=['god', 'type'],
        drop_first=True,
        prefix=['god', 'type'],
        prefix_sep='_'
    )

    for col in ['attack', 'mana', 'health', 'attack_mana', 'attack_health', 'mana_health', 'attack_mana_health']:
        test_df[col] = test_df[col].fillna(test_df[col].mean())

    test_df = test_df[train_columns]

    return test_df


def train_and_evaluate_models(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series):
    logger.info('Training and evaluating models...')

    models = [
        ('Logistic Regression', LogisticRegression()),
        ('Random Forest', RandomForestClassifier(
            bootstrap=False,
            criterion="entropy",
         max_features=0.55,
         min_samples_leaf=8,
         min_samples_split=12,
         n_estimators=100
         )),
        ('Gradient Boosting', GradientBoostingClassifier(
            learning_rate=0.1,
            n_estimators=100,
            subsample=0.9
        )),
        ('SVC', SVC()),
        ('Linear SVC', LinearSVC())
    ]

    results_metrics = []
    models_result = []
    names = []

    for name, model in models:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        ks = ks_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)

        results_metrics.append(ks)
        names.append(name)
        models_result.append(model)

        logger.info(f'{name} KS: {ks:.2f}')
        logger.info(f'{name} Accuracy: {accuracy:.2f}')
        logger.info(f'{name} F1: {f1:.2f}')
        logger.info(f'{name} Precision: {precision:.2f}')
        logger.info("\n")

    return get_best_model(results_metrics, models_result, names)


def save_results(model, predictions: np.ndarray, data_path: str):
    logger.info('Saving results...')
    joblib.dump(model, "artifacts/best_model.pkl")

    raw_test = pd.read_csv(data_path + '/test.csv')
    raw_train = pd.read_csv(data_path + '/train.csv')

    raw_test['strategy'] = predictions
    raw_test['strategy'] = raw_test['strategy'].apply(
        lambda x: 'late' if x == 1 else 'early'
    )

    joined = pd.concat([raw_train, raw_test], axis=0)
    joined = joined.sort_values(by='id')
    joined.to_parquet(data_path + '/cards.parquet', index=False)
    joined.to_csv(data_path + '/cards.csv', index=False)


def run():
    _path = os.path.abspath(os.getcwd())
    train_df, test_df = load_data(_path + "/data")

    X = train_df.drop(['strategy', 'id'], axis=1)
    y = train_df['strategy']
    test_df = prepare_test_data(test_df.copy(), X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=237
    )
    best_model_name, best_model_ks, best_model = train_and_evaluate_models(
        X_train,
        X_test,
        y_train,
        y_test
    )

    logger.info(f'Best model: {best_model_name}, KS: {best_model_ks:.2f}')

    y_pred = best_model.predict(test_df)

    save_results(best_model, y_pred, _path + '/data')

    logger.info('Model building completed successfully!')
