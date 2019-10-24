#%% Importing libraries and setting styles

import copy
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from dotenv import find_dotenv, load_dotenv
from IPython.core.interactiveshell import InteractiveShell
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFECV, VarianceThreshold
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import (
    GridSearchCV,
    ShuffleSplit,
    cross_val_predict,
    cross_validate,
    train_test_split,
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import (
    MinMaxScaler,
    RobustScaler,
    StandardScaler,
    FunctionTransformer,
)
from sklearn.tree import ExtraTreeRegressor

from src.visualization.visualize import (
    crossvalidate_pipeline_scores,
    plot_scores,
    train_and_plot_prediction_metrics,
)

# Setting styles
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
# * Creating a pipeline for three different scaler options
# * Robset Scaler has lower variance of metrics values than the two others
# and also better R2 and squared error values

#%% Creating pipelines

scaler_pipelines = {
    "robust_scaler": make_pipeline(
        VarianceThreshold(), RobustScaler(), KNeighborsRegressor()
    ),
    "standard_scaler": make_pipeline(
        VarianceThreshold(), StandardScaler(), KNeighborsRegressor()
    ),
    "min_max_scaler": make_pipeline(
        VarianceThreshold(), MinMaxScaler(), KNeighborsRegressor()
    ),
}

#%% Scoring and plotting Galaxy data

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
# * Using scoring metric of mean squared error

#%% Optimization

grid_search_pipeline = make_pipeline(
    VarianceThreshold(), RobustScaler(), KNeighborsRegressor()
)

parameters = {
    "kneighborsregressor__n_neighbors": list(range(4, 18)),
    "kneighborsregressor__p": list(range(1, 3)),
}
clf = GridSearchCV(
    grid_search_pipeline, parameters, cv=10, n_jobs=3, scoring="neg_mean_squared_error"
)
clf.fit(X_train, y_train)
print("\nBest Parameters:", clf.best_params_)


#%% [markdown]

# # Feature Selection
#
# * The chosen algorithm for recursive feature elimination
# and cross-validated selection makes a lot of difference
# * Extra Trees and Elastic Net hold a very small amount of columns. A bit too small
# * Random Forest seems better and keeps way more variables


#%%

selector = RFECV(
    ExtraTreeRegressor(random_state=random_state),
    step=1,
    cv=10,
    scoring="neg_mean_squared_error",
    n_jobs=3,
)
selector_extra_trees = selector.fit(X_train, y_train)
columns_to_keep_extra_trees = X_train.columns[selector_extra_trees.support_]
print("Columns to keep with Extra Trees:", columns_to_keep_extra_trees)
print(
    "Proportion of features kept with Extra Trees:",
    round(len(columns_to_keep_extra_trees) * 100 / len(X_train.columns), 1),
    "%",
)

selector = RFECV(
    ElasticNet(random_state=random_state),
    step=1,
    cv=10,
    scoring="neg_mean_squared_error",
    n_jobs=3,
)

selector_glmnet = selector.fit(X_train, y_train)
columns_to_keep_glmnet = X_train.columns[selector_glmnet.support_]
print("Columns to keep with Elastic Net:", columns_to_keep_glmnet)
print(
    "Proportion of features kept with Elastic Net:",
    round(len(columns_to_keep_glmnet) * 100 / len(X_train.columns), 1),
    "%",
)

selector = RFECV(
    RandomForestRegressor(n_estimators=100, random_state=random_state),
    step=1,
    cv=10,
    scoring="neg_mean_squared_error",
    n_jobs=3,
)
selector_rf = selector.fit(X_train, y_train)
columns_to_keep_rf = X_train.columns[selector_rf.support_]
print("Columns to keep with Random Forest:", columns_to_keep_rf)
print(
    "Proportion of features kept with Random Forest:",
    round(len(columns_to_keep_rf) * 100 / len(X_train.columns), 1),
    "%",
)


#%% [markdown]

# # Optimizing the Hyperparameters for Model with Feature Selection
#
# * Using scoring metric of mean squared error
# * N neighbors the same as for the data without feature selection, but uses Euclidean instead of
# Manhattan distance metric

#%% Optimization

grid_search_pipeline = make_pipeline(
    VarianceThreshold(), RobustScaler(), KNeighborsRegressor()
)

parameters = {
    "kneighborsregressor__n_neighbors": list(range(4, 18)),
    "kneighborsregressor__p": list(range(1, 3)),
}
clf = GridSearchCV(
    grid_search_pipeline, parameters, cv=10, n_jobs=3, scoring="neg_mean_squared_error"
)
clf.fit(X_train[columns_to_keep_rf], y_train)
print("\nBest Parameters:", clf.best_params_)

#%% [markdown]

# # Performance of the optimized model compared to unoptimized model
#
# * Feature selection does not make much difference, so we will prefer that
# as it makes for a simpler model

#%%


def drop_preselected_columns(X):
    return X[:, selector_rf.support_]


final_pipelines = {
    "default_model": make_pipeline(
        VarianceThreshold(), RobustScaler(), KNeighborsRegressor()
    ),
    "default_model_feature_selection": make_pipeline(
        VarianceThreshold(),
        RobustScaler(),
        FunctionTransformer(drop_preselected_columns, validate=False),
        KNeighborsRegressor(),
    ),
    "optimized_model_for_no_feature_selection": make_pipeline(
        VarianceThreshold(), RobustScaler(), KNeighborsRegressor(n_neighbors=16, p=1)
    ),
    "optimized_model_for_no_feature_feature_selection": make_pipeline(
        VarianceThreshold(),
        RobustScaler(),
        FunctionTransformer(drop_preselected_columns, validate=False),
        KNeighborsRegressor(n_neighbors=16, p=1),
    ),
    "optimized_model": make_pipeline(
        VarianceThreshold(), RobustScaler(), KNeighborsRegressor(n_neighbors=16, p=2)
    ),
    "optimized_model_feature_selection": make_pipeline(
        VarianceThreshold(),
        RobustScaler(),
        FunctionTransformer(drop_preselected_columns, validate=False),
        KNeighborsRegressor(n_neighbors=16, p=2),
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
