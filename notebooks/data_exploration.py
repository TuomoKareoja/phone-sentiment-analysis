#%%

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.core.interactiveshell import InteractiveShell

# Setting styles
InteractiveShell.ast_node_interactivity = "all"
sns.set(style="whitegrid", color_codes=True)

#%%

data_iphone = pd.read_csv(
    os.path.join("data", "external", "iphone_smallmatrix_labeled_8d.csv")
)
data_galaxy = pd.read_csv(
    os.path.join("data", "external", "galaxy_smallmatrix_labeled_8d.csv")
)

#%% [markdown]

# * Columns
# * The number of missing values

#%%

print("iphone set size:", len(data_iphone))
print("galaxy set size:", len(data_galaxy))

data_iphone.head()
data_galaxy.head()

data_iphone.dtypes
print("missing values in iphone set:", sum(data_iphone.isnull().sum()))
print("missing values in galaxy set:", sum(data_galaxy.isnull().sum()))

#%% [markdown]

# * The dataset do only seem to differ with the last column.
# IF WE DROP IT THE DATASETS ARE IDENTICAL!

#%%

print(
    "Are datasets identical except for the last column?",
    data_iphone.drop(columns="iphonesentiment").equals(
        data_galaxy.drop(columns="galaxysentiment")
    ),
)

#%%
