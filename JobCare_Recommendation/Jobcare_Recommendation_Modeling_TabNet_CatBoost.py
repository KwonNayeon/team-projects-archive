# -*- coding: utf-8 -*-
"""Jobcare_Recommendation_Modeling_TabNet_CatBoost.py

Script for modeling job recommendations using TabNet and CatBoost.
"""

# Install necessary packages
!pip install pytorch_tabnet catboost bayesian-optimization

import os
import numpy as np
import pandas as pd
from typing import Dict
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder
import torch
from pytorch_tabnet.tab_model import TabNetClassifier
from pytorch_tabnet.metrics import Metric
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split, KFold
from bayes_opt import BayesianOptimization
from functools import partial
import warnings
import random

# Suppress warnings
warnings.filterwarnings('ignore')

# Seed initialization for reproducibility
random.seed(2020)
np.random.seed(2020)
torch.manual_seed(2020)

# Define paths (customize according to your environment)
DATA_PATH = "path/to/data"  # Replace with the path to your data
SUBMIT_PATH = "path/to/save/results"  # Replace with the path to save results

# Load data
df_train = pd.read_csv(DATA_PATH + '/train.csv')
df_test = pd.read_csv(DATA_PATH + '/test.csv')

# Define and add code mappings (customize according to your dataset)
def add_code(df: pd.DataFrame) -> pd.DataFrame:
    # Add feature engineering code here
    return df

df_train = add_code(df_train)
df_test = add_code(df_test)

# Feature engineering
df_train['Hour'] = pd.to_datetime(df_train['contents_open_dt']).dt.hour
df_train['day'] = pd.to_datetime(df_train['contents_open_dt']).dt.dayofweek
df_test['Hour'] = pd.to_datetime(df_test['contents_open_dt']).dt.hour
df_test['day'] = pd.to_datetime(df_test['contents_open_dt']).dt.dayofweek

df_train['Hour'] = df_train['Hour'].apply(lambda x: 0 if x < 9 else 1 if x < 19 else 2)
df_test['Hour'] = df_test['Hour'].apply(lambda x: 0 if x < 9 else 1 if x < 19 else 2)
df_train['day'] = df_train['day'].apply(lambda x: 1 if x < 5 else 0)
df_test['day'] = df_test['day'].apply(lambda x: 1 if x < 5 else 0)

# Drop unused columns
df_train = df_train.drop(['id', 'person_prefer_f', 'person_prefer_g', 'contents_open_dt'], axis=1)
df_test = df_test.drop(['id', 'person_prefer_f', 'person_prefer_g', 'contents_open_dt'], axis=1)

# Ensure 'Set' column exists; otherwise, initialize with "train"
if "Set" not in df_train.columns:
    df_train["Set"] = np.random.choice(["train"], p=[1], size=(df_train.shape[0],))

# Split data for modeling
train_indices = df_train[df_train.Set == "train"].index

# Define features and categorical columns
ordinal = ['person_attribute_a_1', 'person_attribute_b', 'person_prefer_e', 'contents_attribute_e']
categorical_columns = []
categorical_dims = {}

# Encode categorical features
for col in df_train.columns:
    if col not in ordinal + ['Set', 'target']:
        l_enc = LabelEncoder()
        df_train[col] = l_enc.fit_transform(df_train[col].values)
        categorical_columns.append(col)
        categorical_dims[col] = len(l_enc.classes_)
        df_test[col] = l_enc.transform(df_test[col].values)

unused_feat = ['Set']
features = [col for col in df_train.columns if col not in unused_feat + ['target']]
cat_idxs = [i for i, f in enumerate(features) if f in categorical_columns]
cat_dims = [categorical_dims[f] for i, f in enumerate(features) if f in categorical_columns]

# Define custom metric for TabNet
class F1_Score(Metric):
    def __init__(self):
        self._name = "f1"
        self._maximize = True

    def __call__(self, y_true, y_score):
        return f1_score(y_true, y_score[:, 1] > 0.5)

# Define function for CatBoost modeling
def catboost_modeling(X_train, y_train, X_test, grow_policy, depth, learning_rate, l2_leaf_reg, random_seed, n):
    test_pred = pd.Series([0] * len(X_test), index=X_test.index)
    for i in range(n):
        kf = KFold(n_splits=10, random_state=random_seed + i, shuffle=True)
        for train_index, valid_index in kf.split(X_train):
            train_X, train_y = X_train.iloc[train_index], y_train.iloc[train_index]
            valid_X, valid_y = X_train.iloc[valid_index], y_train.iloc[valid_index]

            model = CatBoostClassifier(
                eval_metric='F1',
                iterations=5000,
                early_stopping_rounds=1000,
                task_type='CPU',
                grow_policy=grow_policy,
                cat_features=features,
                depth=depth,
                learning_rate=learning_rate,
                l2_leaf_reg=l2_leaf_reg,
                random_seed=random_seed + i
            )
            model.fit(train_X, train_y, eval_set=(valid_X, valid_y), verbose=100)
            test_pred += model.predict_proba(X_test)[:, 1] / n

    return test_pred

# Define function for TabNet modeling
def tabnet_modeling(X_train, y_train, X_test, n):
    test_pred = pd.Series([0] * len(X_test), index=X_test.index)
    for i in range(n):
        kf = KFold(n_splits=10, random_state=2020 + i, shuffle=True)
        for train_index, valid_index in kf.split(X_train):
            train_X, train_y = X_train.iloc[train_index], y_train.iloc[train_index]
            valid_X, valid_y = X_train.iloc[valid_index], y_train.iloc[valid_index]

            model = TabNetClassifier(
                cat_idxs=cat_idxs,
                cat_dims=cat_dims,
                seed=2020 + i,
                n_d=8,
                n_a=8,
                n_steps=3,
                gamma=1.3,
                n_independent=2,
                n_shared=2,
                epsilon=1e-15,
                optimizer_fn=torch.optim.Adam,
                optimizer_params=dict(lr=2e-2),
                scheduler_params={"step_size": 50, "gamma": 0.9},
                scheduler_fn=torch.optim.lr_scheduler.StepLR,
                mask_type='sparse',
                device_name='cpu'
            )
            model.fit(X_train.values, y_train.values, eval_set=[(valid_X.values, valid_y.values)], verbose=100)
            test_pred += model.predict_proba(X_test.values)[:, 1] / n

    return test_pred

# Define the objective function for Bayesian optimization
def objective_func(params, X_train, y_train, X_test, y_test):
    grow_policy = params['grow_policy']
    depth = int(params['depth'])
    learning_rate = params['learning_rate']
    l2_leaf_reg = params['l2_leaf_reg']
    random_seed = int(params['random_seed'])

    test_pred = catboost_modeling(X_train, y_train, X_test, grow_policy, depth, learning_rate, l2_leaf_reg, random_seed, n=1)
    score = roc_auc_score(y_test, test_pred)
    return score

# Tune CatBoost model hyperparameters using Bayesian optimization
def tune_catboost(X_train, y_train, X_test, y_test):
    pbounds = {
        'grow_policy': ('Depthwise', 'Lossguide'),
        'depth': (5, 12),
        'learning_rate': (0.01, 0.1),
        'l2_leaf_reg': (1, 10),
        'random_seed': (1, 100)
    }
    optimizer = BayesianOptimization(
        f=partial(objective_func, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test),
        pbounds=pbounds,
        verbose=2,
        random_state=2020
    )
    optimizer.maximize(init_points=5, n_iter=20)
    return optimizer.max['params']

# Train and predict with both CatBoost and TabNet models
def train_and_predict(X_train, y_train, X_test, y_test):
    best_params = tune_catboost(X_train, y_train, X_test, y_test)
    print("Best params:", best_params)

    grow_policy = best_params['grow_policy']
    depth = int(best_params['depth'])
    learning_rate = best_params['learning_rate']
    l2_leaf_reg = best_params['l2_leaf_reg']
    random_seed = int(best_params['random_seed'])

    test_pred_catboost = catboost_modeling(X_train, y_train, X_test, grow_policy, depth, learning_rate, l2_leaf_reg, random_seed, n=1)
    test_pred_tabnet = tabnet_modeling(X_train, y_train, X_test, n=1)

    test_pred = (test_pred_catboost + test_pred_tabnet) / 2
    return test_pred

# Create training and test data
X_train, X_test, y_train, y_test = train_test_split(df_train[features], df_train['target'], test_size=0.2, random_state=2020)

# Train and predict
test_pred = train_and_predict(X_train, y_train, df_test[features], y_test)

# Save results
submission = pd.read_csv(DATA_PATH + '/sample_submission.csv')
submission['target'] = test_pred
submission.to_csv(SUBMIT_PATH + '/submission.csv', index=False)