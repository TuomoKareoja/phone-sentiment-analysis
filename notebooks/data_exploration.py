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

# # Basic Information about the data

#%%

print("iphone set size:", len(data_iphone))
print("galaxy set size:", len(data_galaxy))

data_iphone.head()
data_galaxy.head()

data_iphone.dtypes
print("missing values in iphone set:", sum(data_iphone.isnull().sum()))
print("missing values in galaxy set:", sum(data_galaxy.isnull().sum()))

#%% [markdown]

# # Checking if Datasets are the Same
#
# * The dataset only differ from the target variable (sentiment towards IPhone or Samsung Galaxy)

#%%

print(
    "Are datasets identical except for the last column?",
    data_iphone.drop(columns="iphonesentiment").equals(
        data_galaxy.drop(columns="galaxysentiment")
    ),
)

#%% [markdown]

# # Value Distributions for the Targets
#
# * Target variables have very uneven distributions with both the lowest and the
# and highest values having a disportionate share
#   * It might be a good to try to split the values into neutral, positive and negative

#%%

sns.countplot(x="iphonesentiment", data=data_iphone)
plt.title("IPhone Sentiment Distribution")
plt.show()

sns.countplot(x="galaxysentiment", data=data_galaxy)
plt.title("Galaxy Sentiment Distribution")
plt.show()

#%% [markdown]

# # Value distributions for Sites with no Mention of the Phone

# * Same as before, but keeping only the websites where there are mentions of the phone
# * Sites that don't mention Galaxy have more sentiments than sites that do! Opposite is true
# for the IPhone
# * There might be some error in the coding or then I'm misunderstanding the columns

#%%

sns.countplot(x="iphonesentiment", data=data_iphone[data_iphone.iphone > 0])
plt.title("IPhone Sentiment Distribution (Iphone mentioned)")
plt.show()

sns.countplot(x="galaxysentiment", data=data_galaxy[data_galaxy.samsunggalaxy > 0])
plt.title("Galaxy Sentiment Distribution (Galaxy mentioned)")
plt.show()

sns.countplot(x="iphonesentiment", data=data_iphone[data_iphone.iphone == 0])
plt.title("IPhone Sentiment Distribution (Iphone not mentioned)")
plt.show()

sns.countplot(x="galaxysentiment", data=data_galaxy[data_galaxy.samsunggalaxy == 0])
plt.title("Galaxy Sentiment Distribution (Galaxy not mentioned)")
plt.show()

#%% [markdown]

# # Are the Sentiments Identical?

# * Sentiment for the both phones is almost the same if you look at all the websites
# * Sites that mention both phones have usually different sentiments, as do sites
# that mention neither of the phones
# * This looks legit unlike the previous findings about the distribution of the values

#%%

print(
    "Percent of sites with identical sentiment to both phones:",
    round(
        sum(data_galaxy.galaxysentiment == data_iphone.iphonesentiment)
        * 100
        / len(data_galaxy),
        1,
    ),
    "%",
)


both_phones_mask = (data_galaxy.samsunggalaxy > 0) & (data_iphone.iphone > 0)

print(
    "Percent of sites with identical sentiment to both phones when both phones mentioned:",
    round(
        sum(
            data_galaxy[both_phones_mask].galaxysentiment
            == data_iphone[both_phones_mask].iphonesentiment
        )
        * 100
        / len(data_galaxy),
        1,
    ),
    "%",
)

neither_phone_mask = (data_galaxy.samsunggalaxy == 0) & (data_iphone.iphone == 0)

print(
    "Percent of sites with identical sentiment to both phones when neither of the phones mentioned:",
    round(
        sum(
            data_galaxy[neither_phone_mask].galaxysentiment
            == data_iphone[neither_phone_mask].iphonesentiment
        )
        * 100
        / len(data_galaxy),
        1,
    ),
    "%",
)

#%% [markdown]

# # Distribution of the Features
#
# * Feature columns don't follow a Gaussian distribution and have outliers
#   * Use RobustScaler instead of StandardScaler
# * Some feature columns have no variance. These should be removed before going further

#%%

feature_columns = data_galaxy.columns[:-1]

for column in feature_columns:
    sns.countplot(x=column, data=data_galaxy)
    plt.title(column)
    plt.show()

#%% [markdown]

# # Correlation Between the Variables
#
# * Correlations are normally low and positive
# * Distinctive click that only correlate with each other
# (htcphone + htcdispos, iosperpos, iosperneg)
# * Correlations to the target values are all small and mostly negative

#%%

# Dropping columns with very low variance as we have so many variables
data_galaxy = data_galaxy.loc[:, data_galaxy.std() > 0.5]
data_iphone = data_iphone.loc[:, data_iphone.std() > 0.5]

corr_iphone = data_iphone.corr()
corr_galaxy = data_galaxy.corr()

sns.set(rc={"figure.figsize": (12.7, 9.27)})

sns.heatmap(corr_iphone, cmap="RdBu_r", center=0)
plt.title("Correlations for IPhone Dataset")
plt.show()

sns.heatmap(corr_galaxy, cmap="RdBu_r", center=0)
plt.title("Correlations for Galaxy Dataset")
plt.show()


#%%
