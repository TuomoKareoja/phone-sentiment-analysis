import os

import pandas as pd
from dill import dump
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler


def main():

    data = pd.read_csv(os.path.join("data", "processed", "iphone.csv"))
    target = "iphonesentiment"

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
            ),
            RobustScaler(),
            KNeighborsRegressor(n_neighbors=15, p=1),
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
                    "googleperpos",
                    "iosperunc",
                    "googleperunc",
                ]
            ),
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
        dump(pipeline, open(os.path.join("models", modelname + "_iphone.dill"), "wb"))


if __name__ == "__main__":
    main()
