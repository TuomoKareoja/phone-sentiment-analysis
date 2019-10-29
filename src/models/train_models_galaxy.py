import os

import numpy as np
import pandas as pd
from dotenv import find_dotenv, load_dotenv
from sklearn.base import BaseEstimator, TransformerMixin
from IPython.core.interactiveshell import InteractiveShell
from dill import dump
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler

from src.visualization.visualize import crossvalidate_pipeline_scores, plot_scores

random_state = 123

data = pd.read_csv(os.path.join("data", "processed", "galaxy.csv"))
target = "galaxysentiment"

X = data.drop(columns=target)
y = data[target]


class FeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, feature_names):
        self._feature_names = feature_names

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X[self._feature_names]


pipelines = {
    "knn": make_pipeline(
        FeatureSelector(
            [
                "iphone",
                "samsunggalaxy",
                "googleandroid",
                "iphonecampos",
                "samsungcampos",
                "iphonecamneg",
                "iphonecamunc",
                "iphonedispos",
                "samsungdispos",
                "iphonedisneg",
                "iphonedisunc",
                "iphoneperpos",
                "samsungperpos",
                "iphoneperneg",
                "samsungperneg",
                "iphoneperunc",
            ]
        ),
        RobustScaler(),
        KNeighborsRegressor(n_neighbors=16, p=2),
    ),
    "gradient_boosting": make_pipeline(
        FeatureSelector(
            [
                "iphone",
                "samsunggalaxy",
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
                "iphoneperunc",
                "samsungperunc",
            ]
        ),
        GradientBoostingRegressor(),
    ),
    "random_forest": make_pipeline(
        FeatureSelector(
            [
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
                "iosperunc",
            ]
        ),
        RandomForestRegressor(
            n_estimators=667, max_depth=12, min_samples_split=4, n_jobs=3
        ),
    ),
}

# Training with full data and saving the trained model to models folder
for modelname, pipeline in pipelines.items():
    print("Training", modelname, "for galaxy data")
    pipeline.fit(X, y)
    print("Saving", modelname, "for galaxy predictions")
    dump(pipeline, open(os.path.join("models", modelname + "_galaxy.dill"), "wb"))
