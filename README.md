k-means-clustering-implementation
==============================

Implementation of a K-Means clustering algorithm to cluster crowd-acquired user stories about smart home applications.

## Setting up locally

Make sure you have [Python](https://www.python.org/) and [pip](https://pip.pypa.io/en/stable/installing/) installed.

```bash
git clone https://github.com/diodax/k-means-clustering-implementation.git
cd k-means-clustering-implementation
pip install -r requirements.txt
```

## Step 1: Text Pre-processing

```bash
python src/data/make_dataset.py data/raw/smarthome-userstories-1k.csv /data/processed/smarthome-userstories.csv
```

The results of the text pre-processing will be saved on the `/data/processed` folder.

## Step 2: TF-IDF Computation and Vector Space Representation

To generate the table with the TF-IDF scores of each row (user story) in the dataset, execute the following command:

```bash
python src/features/build_features.py data/processed/smarthome-userstories.csv models/tf-idf-scores.csv
```

The resulting file with the TF-IDF scores with be saved on the `/models` folder.

## Step 3: K-Means Clustering

To generate the clusters and print the algorithm results, execute:

```bash
python src/models/train_model.py models/tf-idf-scores.csv reports/ --k=2
```

The report will be created on the `/reports` folder.

## Project Organization

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.testrun.org


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
