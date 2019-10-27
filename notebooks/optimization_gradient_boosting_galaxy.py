#%% Importing libraries and setting styles

import copy
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from bayes_opt import BayesianOptimization
from dotenv import find_dotenv, load_dotenv
from IPython.core.interactiveshell import InteractiveShell
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.feature_selection import RFECV, VarianceThreshold
from sklearn.model_selection import (
    GridSearchCV,
    ShuffleSplit,
    cross_val_score,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    MinMaxScaler,
    RobustScaler,
    StandardScaler,
)

from src.visualization.visualize import (
    crossvalidate_pipeline_scores,
    plot_scores,
    train_and_plot_prediction_metrics,
)

# Setting styles
InteractiveShell.ast_node_interactivity = "all"
sns.set(style="whitegrid", color_codes=True, rc={"figure.figsize": (12.7, 9.27)})

random_state = 123


#%% Loading data

data = pd.read_csv(os.path.join("data", "processed", "galaxy.csv"))

#%% Splitting to train and test

target = "galaxysentiment"

X_train, X_test, y_train, y_test = train_test_split(
    data.drop(columns=target), data[target], test_size=0.2, random_state=random_state
)

#%% [markdown]

# # Comparison of Scalers
#
# * Gradient boosting does not necessarily benefit from scaling, but lets try
# * Creating a pipeline for three different scaler options and no scaling
# * Does not make a difference so lets not scale

#%% Creating pipelines

scaler_pipelines = {
    "robust_scaler": make_pipeline(
        VarianceThreshold(), RobustScaler(), GradientBoostingRegressor()
    ),
    "standard_scaler": make_pipeline(
        VarianceThreshold(), StandardScaler(), GradientBoostingRegressor()
    ),
    "min_max_scaler": make_pipeline(
        VarianceThreshold(), MinMaxScaler(), GradientBoostingRegressor()
    ),
    "no_scaling": make_pipeline(VarianceThreshold(), GradientBoostingRegressor()),
}

#%% Scoring and plotting Iphone data

scores = crossvalidate_pipeline_scores(
    X=X_train,
    y=y_train,
    pipelines=scaler_pipelines,
    n_splits=10,
    random_state=random_state,
)

plot_scores(scores=scores)

#%% [markdown]

# # Optimizing the Hyperparameters for Model Without Feature Selection
#
# * Using scoring metric of mean squared error and bayesian optimization

#%% Optimization


def gradient_boosting_cv(max_depth, min_samples_split, min_samples_leaf, max_features):
    pipeline = make_pipeline(
        VarianceThreshold(),
        GradientBoostingRegressor(
            learning_rate=0.001,
            n_estimators=100,
            max_depth=int(max_depth),
            min_samples_split=int(min_samples_split),
            min_samples_leaf=int(min_samples_leaf),
            max_features=int(max_features),
        ),
    )
    cv_scores = cross_val_score(
        pipeline, X_train, y_train, cv=3, scoring="neg_mean_squared_error", n_jobs=3
    )
    return cv_scores.mean()


parameter_bounds = {
    "max_depth": (4, 7),
    "min_samples_split": (2, 100),
    "min_samples_leaf": (4, 10),
    "max_features": (3, X_train.shape[1]),
}

optimizer = BayesianOptimization(
    f=gradient_boosting_cv,
    pbounds=parameter_bounds,
    verbose=2,
    random_state=random_state,
)

optimizer.maximize(init_points=50, n_iter=50)
print(optimizer.max)


#%% [markdown]

# # Feature Selection
#
# * Doing feature selection with the optimized model
# * Random Forest seems better and keeps way more variables


#%%

selector = RFECV(
    GradientBoostingRegressor(
        learning_rate=0.001,
        n_estimators=100,
        max_depth=7,
        min_samples_split=28,
        min_samples_leaf=10,
        # this causes errors
        # max_features=27,
    ),
    step=1,
    cv=10,
    scoring="neg_mean_squared_error",
    n_jobs=3,
)
selector = selector.fit(X_train, y_train)
columns_to_keep = X_train.columns[selector.support_]
print("Columns to keep:", columns_to_keep)
print(
    "Proportion of features kept:",
    round(len(columns_to_keep) * 100 / len(X_train.columns), 1),
    "%",
)


#%% [markdown]

# # Optimizing the Hyperparameters for Model with Feature Selection
#
# * Using scoring metric of mean squared error

#%% Optimization


def gradient_boosting_cv(max_depth, min_samples_split, min_samples_leaf, max_features):
    pipeline = make_pipeline(
        VarianceThreshold(),
        GradientBoostingRegressor(
            learning_rate=0.001,
            n_estimators=100,
            max_depth=int(max_depth),
            min_samples_split=int(min_samples_split),
            min_samples_leaf=int(min_samples_leaf),
            max_features=int(max_features),
        ),
    )
    cv_scores = cross_val_score(
        pipeline,
        X_train[columns_to_keep],
        y_train,
        cv=3,
        scoring="neg_mean_squared_error",
        n_jobs=3,
    )
    return cv_scores.mean()


parameter_bounds = {
    "max_depth": (4, 7),
    "min_samples_split": (2, 100),
    "min_samples_leaf": (4, 10),
    "max_features": (3, X_train[columns_to_keep].shape[1]),
}

optimizer = BayesianOptimization(
    f=gradient_boosting_cv,
    pbounds=parameter_bounds,
    verbose=2,
    random_state=random_state,
)

optimizer.maximize(init_points=50, n_iter=50)
print(optimizer.max)


#%% [markdown]

# # Performance of the optimized models


#%%


def drop_preselected_columns(X):
    return X[:, selector.support_]


final_pipelines = {
    "default_model": make_pipeline(VarianceThreshold(), GradientBoostingRegressor()),
    "default_model_feature_selection": make_pipeline(
        VarianceThreshold(),
        FunctionTransformer(drop_preselected_columns, validate=False),
        GradientBoostingRegressor(),
    ),
    "optimized_model_for_no_feature_selection": make_pipeline(
        VarianceThreshold(),
        GradientBoostingRegressor(
            learning_rate=0.001,
            n_estimators=100,
            max_depth=7,
            min_samples_split=28,
            min_samples_leaf=10,
            # max_features=15,
        ),
    ),
    "optimized_model_for_no_feature_feature_selection": make_pipeline(
        VarianceThreshold(),
        FunctionTransformer(drop_preselected_columns, validate=False),
        GradientBoostingRegressor(
            learning_rate=0.001,
            n_estimators=100,
            max_depth=7,
            min_samples_split=28,
            min_samples_leaf=10,
            # max_features=15,
        ),
    ),
    "optimized_model": make_pipeline(
        VarianceThreshold(),
        GradientBoostingRegressor(
            learning_rate=0.001,
            n_estimators=100,
            max_depth=7,
            min_samples_split=2,
            min_samples_leaf=10,
            # max_features=15,
        ),
    ),
    "optimized_model_feature_selection": make_pipeline(
        VarianceThreshold(),
        FunctionTransformer(drop_preselected_columns, validate=False),
        GradientBoostingRegressor(
            learning_rate=0.001,
            n_estimators=100,
            max_depth=7,
            min_samples_split=2,
            min_samples_leaf=10,
            # max_features=15,
        ),
    ),
}

scores_optimized = crossvalidate_pipeline_scores(
    X=X_train,
    y=y_train,
    pipelines=final_pipelines,
    n_splits=30,
    random_state=random_state,
)

plot_scores(scores=scores_optimized)

train_and_plot_prediction_metrics(
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    pipelines=final_pipelines,
)


#%%
