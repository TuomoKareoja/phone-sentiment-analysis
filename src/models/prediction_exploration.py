#%%


import re
import os
import pprint

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from dotenv import find_dotenv, load_dotenv
from IPython.core.interactiveshell import InteractiveShell
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

# Setting styles
pp = pprint.PrettyPrinter(indent=4)
InteractiveShell.ast_node_interactivity = "all"
sns.set(style="whitegrid", color_codes=True, rc={"figure.figsize": (12.7, 9.27)})

#%%

data = pd.read_csv(os.path.join("data", "predictions", "predictions.csv"))

#%%

sns.countplot(x="iphone", data=data)
plt.title("Number of Times IPhone Mentioned")
plt.xlabel("Number of Mentions")
plt.show()

sns.countplot(x="samsunggalaxy", data=data)
plt.title("Number of Times Galaxy Mentioned")
plt.xlabel("Number of Mentions")
plt.show()

# %%

sns.scatterplot(
    x="random_forest_iphone",
    y="iphone",
    alpha=0.2,
    data=data[data.iphone < 100],
    s=100,
    label="iPhone",
)
sns.scatterplot(
    x="random_forest_galaxy",
    y="samsunggalaxy",
    alpha=0.2,
    data=data,
    s=100,
    label="Samsung Galaxy",
)
plt.xlabel("Positivity of Sentiment")
plt.ylabel("Number of Mentions")
plt.legend()
plt.show()


# %%

sns.scatterplot(
    x="random_forest_iphone",
    y="iphone",
    alpha=0.2,
    data=data[(data.iphone > 0) & (data.iphone < 100)],
    s=100,
    label="iPhone",
)
sns.scatterplot(
    x="random_forest_galaxy",
    y="samsunggalaxy",
    alpha=0.2,
    data=data[data.samsunggalaxy > 0],
    s=100,
    label="Samsung Galaxy",
)
plt.xlabel("Positivity of Sentiment")
plt.ylabel("Number of Mentions")
plt.legend()
plt.show()

#%%

sns.scatterplot(
    x="random_forest_iphone",
    y="iphone",
    alpha=0.2,
    data=data[
        (data.url.str.contains("iphone")) & (data.iphone > 1) & (data.iphone < 100)
    ],
    s=100,
    label="iPhone",
)
sns.scatterplot(
    x="random_forest_galaxy",
    y="samsunggalaxy",
    alpha=0.2,
    data=data[(data.url.str.contains("galaxy")) & (data.samsunggalaxy > 1)],
    s=100,
    label="Samsung Galaxy",
)
plt.xlabel("Positivity of Sentiment")
plt.ylabel("Number of Mentions")
plt.legend()
plt.show()

pp.pprint(data[(data.url.str.contains("iphone"))].url.head(20))
pp.pprint(data[(data.url.str.contains("galaxy"))].url.head(20))

# %%

stopwords = set(STOPWORDS)
stopwords.update(
    ["html", "www", "https", "http", "wordpress", "www", "amp", "tag", "com", "net"]
)


def getWordsFromURL(url):
    return re.compile(r"[\:/?=\-&]+", re.UNICODE).split(url)


text = " ".join(
    " ".join(url)
    for url in data[(data.iphone == 0) & (data.random_forest_iphone == 0)].url.apply(
        getWordsFromURL
    )
)

wordcloud = WordCloud(stopwords=stopwords).generate(text)
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

text = " ".join(
    " ".join(url) for url in data[(data.iphone == 1)].url.apply(getWordsFromURL)
)

wordcloud = WordCloud(stopwords=stopwords).generate(text)
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

text = " ".join(
    " ".join(url) for url in data[(data.iphone >= 10)].url.apply(getWordsFromURL)
)

wordcloud = WordCloud(stopwords=stopwords).generate(text)
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

text = " ".join(
    " ".join(url)
    for url in data[
        (data.samsunggalaxy == 0) & (data.random_forest_galaxy >= 4)
    ].url.apply(getWordsFromURL)
)

wordcloud = WordCloud(stopwords=stopwords).generate(text)
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()


text = " ".join(
    " ".join(url) for url in data[(data.samsunggalaxy == 1)].url.apply(getWordsFromURL)
)

wordcloud = WordCloud(stopwords=stopwords).generate(text)
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()


text = " ".join(
    " ".join(url) for url in data[(data.samsunggalaxy >= 10)].url.apply(getWordsFromURL)
)

wordcloud = WordCloud(stopwords=stopwords).generate(text)
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

# %%

sns.pairplot(
    data[data.iphone < 100][
        ["iphone", "samsunggalaxy", "random_forest_iphone", "random_forest_galaxy"]
    ].sample(n=5000)
)
# plt.xlabel("Positivity of Sentiment")
# plt.ylabel("Number of Mentions")
# plt.legend()
plt.show()


# %%

sns.distplot(
    data.random_forest_galaxy,
    bins=50,
    kde=False,
    hist=True,
    norm_hist=True,
    label="Samsung Galaxy",
)
sns.distplot(
    data.random_forest_iphone,
    bins=50,
    kde=False,
    hist=True,
    norm_hist=True,
    label="iPhone",
)
plt.title("Phone Sentiment Distribution")
plt.xlabel("Positivity of Sentiment")
plt.ylabel("")
plt.legend()
plt.show()

# %%

sns.distplot(
    data[data.samsunggalaxy > 0].random_forest_galaxy,
    bins=50,
    kde=False,
    hist=True,
    norm_hist=True,
    label="Samsung Galaxy",
)
sns.distplot(
    data[data.iphone > 0].random_forest_iphone,
    bins=50,
    kde=False,
    hist=True,
    norm_hist=True,
    label="iPhone",
)
plt.title("Phone Sentiment Distribution (Phones Mentioned at Least Once)")
plt.xlabel("Positivity of Sentiment")
plt.ylabel("")
plt.legend()
plt.show()

# %%

sns.distplot(
    data[data.samsunggalaxy == 0].random_forest_galaxy,
    bins=50,
    kde=False,
    hist=True,
    norm_hist=True,
    label="Samsung Galaxy",
)
sns.distplot(
    data[data.iphone == 0].random_forest_iphone,
    bins=50,
    kde=False,
    hist=True,
    norm_hist=True,
    label="iPhone",
)
plt.title("Phone Sentiment Distribution (Phones Mentioned at Least Once)")
plt.xlabel("Positivity of Sentiment")
plt.ylabel("")
plt.legend()
plt.show()

#%%

data[data.iphone > 100].url.head()

# %%

plt.show()


# %%
