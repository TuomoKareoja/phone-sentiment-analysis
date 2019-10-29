phone-sentiment-analysis
==============================

Sentiment analysis of different smartphones and operating systems performed on AWS-cluster using Common Crawl data

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project
    ├── data
    │   ├── external       <- Manually labeled training datasets
    │   ├── prediction     <- Data with predictions added
    │   ├── processed      <- The final, canonical data sets for modeling
    │   └── raw            <- Raw input from the EMR job
    │
    ├── docs               <- html docs for GitHub Pages
    │
    ├── emr                <- scripts and files for running the EMR job in AWS
    │
    ├── sphinx             <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models
    │
    ├── notebooks          <- Jupyter notebooks and notebook-like objects (e.g. .py files meant to
    │                         run with vscode IPython)
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to clean the data and get it to proper form
    │   │   └── make_dataset.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_models_galaxy.py
    │   │   ├── predict_models_iphone.py
    │   │   └── train_models_galaxy.py
    │   │   └── train_models_iphone.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.testrun.org


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
