import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from dotenv import find_dotenv, load_dotenv
from IPython.core.interactiveshell import InteractiveShell
from joblib import dump
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, RobustScaler

from src.visualization.visualize import crossvalidate_pipeline_scores, plot_scores

random_state = 123

# Setting styles
InteractiveShell.ast_node_interactivity = "all"
sns.set(style="whitegrid", color_codes=True, rc={"figure.figsize": (12.7, 9.27)})


data = pd.read_csv(os.path.join("data", "processed", "iphone.csv"))
target = "iphonesentiment"

X = data.drop(columns=target)
y = data[target]

knn_features = [
    "iphone",
    "samsunggalaxy",
    "ios",
    "googleandroid",
    "iphonecampos",
    "samsungcampos",
    "iphonecamneg",
    "iphonecamunc",
    "iphonedispos",
    "samsungdispos",
    "iphonedisneg",
    "samsungdisneg",
    "iphonedisunc",
    "samsungdisunc",
    "iphoneperpos",
    "samsungperpos",
    "iphoneperneg",
    "samsungperneg",
    "iphoneperunc",
]

gradient_boosting_features = [
    "iphone",
    "samsunggalaxy",
    "googleandroid",
    "iphonecampos",
    "samsungcampos",
    "iphonecamneg",
    "iphonecamunc",
    "iphonedispos",
    "iphonedisneg",
    "samsungdisneg",
    "iphonedisunc",
    "samsungdisunc",
    "iphoneperpos",
    "samsungperpos",
    "samsungperneg",
]

random_forest_features = [
    "iphone",
    "samsunggalaxy",
    "ios",
    "googleandroid",
    "iphonecampos",
    "samsungcampos",
    "iphonecamneg",
    "samsungcamneg",
    "iphonecamunc",
    "samsungcamunc",
    "iphonedispos",
    "samsungdispos",
    "iphonedisneg",
    "samsungdisneg",
    "iphonedisunc",
    "samsungdisunc",
    "iphoneperpos",
    "samsungperpos",
    "iphoneperneg",
    "samsungperneg",
    "iphoneperunc",
    "samsungperunc",
    "googleperpos",
    "iosperunc",
    "googleperunc",
]

knn_feature_selector = X.columns.isin(knn_features)
gradient_boosting_feature_selector = X.columns.isin(gradient_boosting_features)
random_forest_feature_selector = X.columns.isin(random_forest_features)


def keep_knn_features(X):
    return X[:, knn_feature_selector]


def keep_gradient_boosting_features(X):
    return X[:, gradient_boosting_feature_selector]


def keep_random_forest_features(X):
    return X[:, random_forest_feature_selector]


pipelines = {
    "knn": make_pipeline(
        VarianceThreshold(),
        RobustScaler(),
        FunctionTransformer(keep_knn_features, validate=False),
        KNeighborsRegressor(n_neighbors=15, p=1),
    ),
    "gradient_boosting": make_pipeline(
        VarianceThreshold(),
        FunctionTransformer(keep_gradient_boosting_features, validate=False),
        GradientBoostingRegressor(),
    ),
    "random_forest": make_pipeline(
        VarianceThreshold(),
        FunctionTransformer(keep_random_forest_features, validate=False),
        RandomForestRegressor(
            n_estimators=682, max_depth=12, min_samples_split=10, n_jobs=3
        ),
    ),
}

# Training with full data and saving the trained model to models folder
for modelname, pipeline in pipelines.items():
    print("Training", modelname, "for iphone data")
    pipeline.fit(X, y)
    print("Saving", modelname, "for iphone predictions")
    dump(pipeline, os.path.join("models", modelname + "_iphone.joblib"))
