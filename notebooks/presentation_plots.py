#%%

import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from dotenv import find_dotenv, load_dotenv
from IPython.core.interactiveshell import InteractiveShell
from wordcloud import STOPWORDS, ImageColorGenerator, WordCloud

# Running notebook without this with VSCode will create black borders around
# the images

matplotlib.rcParams.update(_VSCode_defaultMatplotlib_Params)

# Setting styles
InteractiveShell.ast_node_interactivity = "all"
plt.tight_layout()
sns.set(style="whitegrid", color_codes=True, rc={"figure.figsize": (12.7, 9.27)})
sns.set_context("notebook", font_scale=1.25)

savepath = os.path.join("reports", "figures")

#%%

predictions = pd.read_csv(os.path.join("data", "predictions", "predictions.csv"))
train_galaxy = pd.read_csv(
    os.path.join("data", "external", "galaxy_smallmatrix_labeled_8d.csv")
)
train_iphone = pd.read_csv(
    os.path.join("data", "external", "iphone_smallmatrix_labeled_8d.csv")
)

#%%

sns.scatterplot(
    x="random_forest_iphone",
    y="iphone",
    alpha=0.2,
    s=100,
    data=predictions[predictions.iphone < 100],
    label="iPhone",
)
sns.scatterplot(
    x="random_forest_galaxy",
    y="samsunggalaxy",
    alpha=0.2,
    s=100,
    data=predictions,
    label="Samsung Galaxy",
)
plt.xlabel("Positivity of Sentiment")
plt.ylabel("Number of Mentions of Phone")
plt.title(
    "Websites by Predicted Positivity of Sentiment and Number of Mentions of Phones"
)
plt.legend()
plt.savefig(os.path.join(savepath, "sentiment_comparison_predictions.png"))
plt.show()

#%%

noise_galaxy = np.random.normal(0, 0.1, len(train_galaxy))
noise_iphone = np.random.normal(0, 0.1, len(train_iphone))

fig, (ax1, ax2) = plt.subplots(ncols=2)
sns.scatterplot(
    x=train_iphone["iphonesentiment"] + noise_iphone,
    y=train_galaxy["galaxysentiment"] + noise_galaxy,
    alpha=0.005,
    s=100,
    ax=ax1,
    color="green",
)
ax1.set_title("Samsung Galaxy vs iPhone Sentiment")
ax1.set_xlabel("iPhone Sentiment Positivity")
ax1.set_ylabel("Samsung Galaxy Sentiment Positivity")

sns.countplot(
    x=train_galaxy["galaxysentiment"],
    hue=np.where(train_galaxy["samsunggalaxy"] > 0, "1+ mentions", "No mentions"),
    ax=ax2,
)
ax2.set_title("Websites Distribution by Sentiment for Samsung Galaxy")
ax2.set_xlabel("Samsung Galaxy Sentiment Positivity")
ax2.set_ylabel("Number of Websites")
plt.savefig(os.path.join(savepath, "training_data_problems.png"))
plt.show()

# %%

stopwords = set(STOPWORDS)
stopwords.update(
    [
        "html",
        "www",
        "https",
        "http",
        "wordpress",
        "org",
        "www",
        "amp",
        "tag",
        "com",
        "net",
    ]
)


def getWordsFromURL(url):
    return re.compile(r"[\:/?=\-&]+", re.UNICODE).split(url)


def create_text(data):
    return " ".join(" ".join(url) for url in data.url.apply(getWordsFromURL))


wordcloud_iphone = WordCloud(
    stopwords=stopwords, background_color="white", width=700, height=1000
).generate(create_text(data=predictions[predictions.iphone > 10]))

wordcloud_galaxy = WordCloud(
    stopwords=stopwords, background_color="white", width=700, height=1000
).generate(create_text(data=predictions[predictions.samsunggalaxy > 10]))

fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12.7, 9.27))
ax1.imshow(wordcloud_iphone, interpolation="bilinear")
ax1.set_title("Over 10 Mentions of iPhone")
ax1.axis("off")

ax2.imshow(wordcloud_galaxy, interpolation="bilinear")
ax2.set_title("Over 10 Mentions of Samsung Galaxy")
ax2.axis("off")

plt.savefig(os.path.join(savepath, "wordclouds_for_sites_with_lots_of_mentions.png"))
plt.show()

# %%
