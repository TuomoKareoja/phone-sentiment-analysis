# -*- coding: utf-8 -*-

import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import ShuffleSplit, cross_validate


def crossvalidate_pipeline_scores(X, y, pipelines, n_splits, random_state):
    """crossvalidates all pipelines in the provided dictionary and returns scores (R2, neg-MAE, neg-MRE)
    
    :param X: Dataframe with the predictors
    :type X: dict
    :param y: Pandas series with the target values
    :type y: series
    :param pipelines: dictionary with the name of the model as key and pipeline as value
    :type pipelines: dict
    :param n_splits: how many splits to do in crossvalidation
    :type n_splits: int
    :param random_state: random state for splitting
    :type random_state: int
    :return: Dataframe with scores calculated for each fold and model
    :rtype: dataframe
    """

    cv = ShuffleSplit(n_splits=n_splits, random_state=random_state)

    scores = {}
    for modelname, pipeline in pipelines.items():
        print("Crossvalidating", modelname)
        score = cross_validate(
            pipeline,
            X,
            y,
            cv=cv,
            scoring=("r2", "neg_mean_absolute_error", "neg_mean_squared_error"),
        )
        scores.update({modelname: score})

    # opening the nested dictionary to a dataframe
    scores = pd.concat({k: pd.DataFrame(v).T for k, v in scores.items()}, axis=0)
    scores.index.names = "model", "metric"
    scores.reset_index(inplace=True)
    scores = pd.melt(scores, id_vars=["model", "metric"], var_name="fold")
    scores = scores.assign(fold=scores.fold + 1)

    return scores


def plot_scores(scores, show_costs=False, save=False, plotname=None):
    """Generates BoxPlots for all metrics
    
    :param scores: Dataframe with columns model, metric, fold and value (output from crossvalidate_pipelines)
    :type scores: dataframe
    :param show_cost: Plot the computation cost metrics
    :type show_cost: boolean
    :param save: Save created plots to reports/figures/
    :type show_cost: boolean
    """

    for metric in scores.metric.drop_duplicates():
        if not show_costs:
            if metric not in [
                "test_r2",
                "test_neg_mean_absolute_error",
                "test_neg_mean_squared_error",
            ]:
                continue
        sns.boxplot(x="model", y="value", data=scores[scores.metric == metric])
        plt.title(metric)
        plt.tight_layout()
        if save:
            plt.savefig(
                os.path.join("reports", "figures", plotname + "_" + metric + ".png")
            )
        plt.show()


def train_and_plot_prediction_metrics(X_train, y_train, X_test, y_test, pipelines):
    """Trains the pipelines with train data, predict test data with trained
    models and the plots MAE, MSE and R2 metrics
 
    :param X_train: Training data features
    :type X_train: dataframe
    :param y_train: Training data target
    :type y_train: array
    :param y_test: Test data target
    :type y_test: array
    :param pipelines: dictionary with the name of the model as key and pipeline as value
    :type pipelines: dict
    """

    scores = pd.DataFrame(columns=["Model", "MAE", "MSE", "R2"])

    for modelname, pipeline in pipelines.items():
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        scores = scores.append(
            {"Model": modelname, "MAE": mae, "MSE": mse, "R2": r2}, ignore_index=True
        )

    for metric in ["MAE", "MSE", "R2"]:
        ax = sns.barplot(x="Model", y=metric, data=scores)
        ax.set_ylim(bottom=0)
        plt.title("Test data: " + metric)
        plt.show()
