import os
from glob import glob

import pandas as pd
import numpy as np
from dill import load
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, RobustScaler

from src.data.load_data import read_and_combine_crawled_data


def main():

    X = read_and_combine_crawled_data()

    # read all galaxy models from the folder
    model_paths = glob(os.path.join("models", "*.dill"))

    for model_path in model_paths:
        with open(model_path, "rb") as model_file:
            print("Loading", model_path)
            model = load(model_file)
            print("Predicting with", model_path)
            predictions = model.predict(X)
            # sentiment should only have values between 0 and 5
            predictions = np.clip(predictions, 0, 5)
            model_name = model_path[model_path.find("/") + 1 : model_path.find(".")]
            X[model_name] = predictions

    X.to_csv(os.path.join("data", "predictions", "predictions.csv"))


if __name__ == "__main__":
    main()
