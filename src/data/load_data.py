import os

import numpy as np
import pandas as pd


def read_and_combine_crawled_data():
    factors = pd.read_csv(
        os.path.join("data", "raw", "concatenated_factors.csv"), index_col="id"
    )
    websites = pd.read_csv(
        os.path.join("data", "raw", "concatenated_websites.csv"), index_col="id"
    )
    # don't have to give column to do the join as joining by index is the default
    df = factors.join(websites, sort=False)
    return df
