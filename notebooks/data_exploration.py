#%%

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.core.interactiveshell import InteractiveShell

# Setting styles
InteractiveShell.ast_node_interactivity = "all"
sns.set(style="whitegrid", color_codes=True, rc={"figure.figsize": (12.7, 9.27)})

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
# * The dataset only differ with the target variable (sentiment towards IPhone or Samsung Galaxy)

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

# # How Many Times Websites Actually Mention the Phones
#
# * For Iphone, most sites mention the phone at least once
# * For Galaxy there almost no sites that mention the site more than once
# and even sites that mention the phone 1 time are less than 1000

#%%

sns.countplot(x="iphone", data=data_iphone)
plt.title("Number of Times IPhone Mentioned")
plt.xlabel("Number of Mentions")
plt.show()

sns.countplot(x="samsunggalaxy", data=data_galaxy)
plt.title("Number of Times Galaxy Mentioned")
plt.xlabel("Number of Mentions")
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
# * Distinctive clicks that only correlate with each other
# (htcphone + htcdispos, iosperpos, iosperneg)
# * Correlations to the target values are all small and mostly negative

#%%

# arranging columns by name
corr_iphone = data_iphone.reindex(sorted(data_iphone.columns), axis=1).corr()
corr_galaxy = data_galaxy.reindex(sorted(data_galaxy.columns), axis=1).corr()

sns.heatmap(corr_iphone, cmap="RdBu_r", center=0)
plt.title("Correlations for IPhone Dataset")
plt.show()

sns.heatmap(corr_galaxy, cmap="RdBu_r", center=0)
plt.title("Correlations for Galaxy Dataset")
plt.show()

#%% [markdown]

# # Correlation Between the Variables When Only Keeping the "Interesting" Columns
#
# * Deducting sentiment from mentions of other phones seems very unreliable expect for
# direct comparisons between Apple and Samsung as these are the market leaders and the
# phones we are actually doing the comparison for
# * Dropping columns that don't refer to iphone, galaxy, ios or android
# * As we would expect android and samsung are mentions are correlated
# * Iphone and IOS sentiment don't surprisingly have much correlation


#%%

columns_to_drop = [
    "sonyxperia",
    "nokialumina",
    "htcphone",
    "sonycampos",
    "nokiacampos",
    "htccampos",
    "sonycamneg",
    "nokiacamneg",
    "htccamneg",
    "sonycamunc",
    "nokiacamunc",
    "htccamunc",
    "sonydispos",
    "nokiadispos",
    "htcdispos",
    "sonydisneg",
    "nokiadisneg",
    "htcdisneg",
    "sonydisunc",
    "nokiadisunc",
    "htcdisunc",
    "sonyperpos",
    "nokiaperpos",
    "htcperpos",
    "sonyperneg",
    "nokiaperneg",
    "htcperneg",
    "sonyperunc",
    "nokiaperunc",
    "htcperunc",
]

data_iphone_small = data_iphone.drop(columns=columns_to_drop)
data_galaxy_small = data_galaxy.drop(columns=columns_to_drop)

# arranging columns by name
data_iphone_small = data_iphone_small.reindex(sorted(data_iphone_small.columns), axis=1)
data_galaxy_small = data_galaxy_small.reindex(sorted(data_galaxy_small.columns), axis=1)

corr_iphone = data_iphone_small.corr()
corr_galaxy = data_galaxy_small.corr()


sns.heatmap(corr_iphone, cmap="RdBu_r", center=0)
plt.title("Correlations for IPhone Dataset")
plt.show()

sns.heatmap(corr_galaxy, cmap="RdBu_r", center=0)
plt.title("Correlations for Galaxy Dataset")
plt.show()

# %%
