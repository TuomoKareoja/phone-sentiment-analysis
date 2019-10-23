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

#%%[markdown]

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

#%% [markdown]

# * The distribution of the target values is very polarized. Might the be a reason to split
# the data into two or three categories (neutral, positive, negative)

#%%

sns.countplot(x="iphonesentiment", data=data_iphone)
plt.title("IPhone Sentiment Distribution")
plt.show()

sns.countplot(x="galaxysentiment", data=data_galaxy)
plt.title("Galaxy Sentiment Distribution")
plt.show()

#%% [markdown]

# * All feature columns are numeric and as the data is the same we only need to look at
# one dataset

# * many columns have no variation at all. We should drop these columns even before
# starting to build the models

feature_columns = data_galaxy.columns[:-1]

for column in feature_columns:
    sns.countplot(x=column, data=data_galaxy)
    plt.title(column)
    plt.show()

