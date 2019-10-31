# -*- coding: utf-8 -*-
"""Functions for loading the dataset. Just pure convenience
"""

import os

import pandas as pd


def read_and_combine_crawled_data():
    """Read crawled data from disc and return it as a dataframe
    
    :return: Dataframe with crawled data (including urls, that are in a separate file)
    :rtype: dataframe
    """
    factors = pd.read_csv(
        os.path.join("data", "raw", "concatenated_factors.csv"), index_col="id"
    )
    websites = pd.read_csv(
        os.path.join("data", "raw", "concatenated_websites.csv"), index_col="id"
    )
    # don't have to give column to do the join as joining by index is the default
    df = factors.join(websites, sort=False)
    return df
