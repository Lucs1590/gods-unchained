import os
import sys
import logging

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import accuracy_score, f1_score, precision_score, roc_curve

logger = logging.getLogger()
handler = logging.StreamHandler(sys.stdout)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


def ks_score(y_true, y_score):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return max(tpr - fpr)


def get_best_model(ks: list, models: list, names: list) -> list:
    name = names[np.argmax(ks)]
    ks = max(ks)
    model = models[np.argmax(ks)]
    logging.info('The best model is %s with KS of %s', name, ks)
    return [name, ks, model]


def run():
    _path = os.path.abspath(os.path.dirname(os.path.abspath(os.getcwd())))
    train_dataframe = pd.read_parquet(
        "artifacts/train_dataframe_engineered.parquet"
    )
    test_dataframe = pd.read_csv(_path + '/data/test.csv')

    X = train_dataframe.drop(['strategy', 'id'], axis=1)
    y = train_dataframe['strategy']

    test_dataframe = test_dataframe.sample(
        frac=1,
        random_state=237
    ).reset_index(drop=True)

    test_dataframe['attack'] = test_dataframe['attack'].apply(
        lambda x: np.log(x)
    )
    test_dataframe['mana'] = test_dataframe['mana'].apply(
        lambda x: np.log(x)
    )
    test_dataframe['health'] = test_dataframe['health'].apply(
        lambda x: np.log(x)
    )

    test_dataframe['attack_mana'] = test_dataframe['attack'] + \
        test_dataframe['mana']
    test_dataframe['attack_health'] = test_dataframe['attack'] + \
        test_dataframe['health']
    test_dataframe['mana_health'] = test_dataframe['mana'] + \
        test_dataframe['health']
    test_dataframe['attack_mana_health'] = test_dataframe['attack'] + \
        test_dataframe['mana'] + test_dataframe['health']

    test_dataframe['att_greater_5'] = test_dataframe['attack'] > 5
    test_dataframe['mana_greater_7'] = test_dataframe['mana'] > 7
    test_dataframe['health_greater_6'] = test_dataframe['health'] > 5
    test_dataframe['att_greater_mana'] = test_dataframe['attack'] > test_dataframe['mana']
    test_dataframe['att_greater_health'] = test_dataframe['attack'] > test_dataframe['health']
    test_dataframe['mana_greater_health'] = test_dataframe['mana'] > test_dataframe['health']

    test_dataframe['att_greater_5'] = test_dataframe['att_greater_5'].astype(
        int
    )
    test_dataframe['mana_greater_7'] = test_dataframe['mana_greater_7'].astype(
        int
    )
    test_dataframe['health_greater_6'] = test_dataframe['health_greater_6'].astype(
        int
    )
    test_dataframe['att_greater_mana'] = test_dataframe['att_greater_mana'].astype(
        int
    )
    test_dataframe['att_greater_health'] = test_dataframe['att_greater_health'].astype(
        int
    )
    test_dataframe['mana_greater_health'] = test_dataframe['mana_greater_health'].astype(
        int
    )

    test_dataframe = pd.get_dummies(
        test_dataframe,
        columns=['god', 'type'],
        drop_first=True,
        prefix=['god', 'type'],
        prefix_sep='_'
    )

    test_dataframe.replace([np.inf, -np.inf], np.nan, inplace=True)
    test_dataframe['attack'] = test_dataframe['attack'].fillna(
        test_dataframe['attack'].mean()
    )
    test_dataframe['mana'] = test_dataframe['mana'].fillna(
        test_dataframe['mana'].mean()
    )
    test_dataframe['health'] = test_dataframe['health'].fillna(
        test_dataframe['health'].mean()
    )
    test_dataframe['attack_mana'] = test_dataframe['attack_mana'].fillna(
        test_dataframe['attack_mana'].mean()
    )
    test_dataframe['attack_health'] = test_dataframe['attack_health'].fillna(
        test_dataframe['attack_health'].mean()
    )
    test_dataframe['mana_health'] = test_dataframe['mana_health'].fillna(
        test_dataframe['mana_health'].mean()
    )
    test_dataframe['attack_mana_health'] = test_dataframe['attack_mana_health'].fillna(
        test_dataframe['attack_mana_health'].mean()
    )

    test_dataframe = test_dataframe[X.columns]

    models = [
        ('LR', LogisticRegression()),
        ('RF', RandomForestClassifier(bootstrap=False, criterion="entropy",
                                      max_features=0.55, min_samples_leaf=8, min_samples_split=12, n_estimators=100)),
        ('GB', GradientBoostingClassifier(
            learning_rate=0.1, n_estimators=100, subsample=0.9)),
        ('SVC', SVC()),
        ('LSVC', LinearSVC())
    ]

    results_metrics = []
    models_result = []
    names = []

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=237
    )

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

        logging.info(f'{name} ks: {ks:.2f}')
        logging.info(f'{name} accuracy: {accuracy:.2f}')
        logging.info(f'{name} f1: {f1:.2f}')
        logging.info(f'{name} precision: {precision:.2f}')
        logging.info("\n")

    best_model = get_best_model(results_metrics, models_result, names)
    best_model[2].fit(X, y)
    y_pred = best_model[2].predict(test_dataframe)

    raw_test = pd.read_csv(_path + '/data/test.csv')
    raw_train = pd.read_csv(_path + '/data/train.csv')

    raw_test['strategy'] = y_pred
    raw_test['strategy'] = raw_test['strategy'].apply(
        lambda x: 'late' if x == 1 else 'early'
    )

    joined = pd.concat([raw_train, raw_test], axis=0)
    joined = joined.sort_values(by='id')

    joblib.dump(best_model[2], "artifacts/best_model.pkl")

    joined.to_parquet(_path + '/data/cards.parquet', index=False)
    joined.to_csv(_path + '/data/cards.csv', index=False)
