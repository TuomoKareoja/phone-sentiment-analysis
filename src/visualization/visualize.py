# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
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


def plot_scores(scores):
    """Generates BoxPlots for all metrics
    
    :param scores: Dataframe with columns model, metric, fold and value (output from crossvalidate_pipelines)
    :type scores: dataframe
    """

    for metric in scores.metric.drop_duplicates():
        print(metric)
        sns.boxplot(x="model", y="value", data=scores[scores.metric == metric])
        plt.title(metric)
        plt.show()
