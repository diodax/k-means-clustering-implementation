K Means Clustering Implementation
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

In this step, we take the raw `smarthome-userstories-1k.csv` dataset as input, apply a list of 
pre-processing steps to it, and save the results on the `smarthome-userstories.csv` file.
To do so, we run the `make_dataset.py` script from the `/src/data` folder. The first 
argument corresponds to the filepath of the raw dataset, and the second one corresponds 
to the expected path for the results file.

```bash
python src/data/make_dataset.py data/raw/smarthome-userstories-1k.csv /data/processed/smarthome-userstories.csv
```

The results of the text pre-processing will be saved on the `/data/processed` folder.

## Step 2: TF-IDF Computation and Vector Space Representation

This step uses the pre-processed file from the previous step to generate a file with the TF-IDF scores 
of each row (user story) in the dataset. To do so, we run the `build_features.py` script from 
the `/src/features` folder.

In this command, the first argument corresponds to the pre-processed data `data/processed/smarthome-userstories.csv` 
and the second argument specifies the output path for the table with the TF-IDF scores.

```bash
python src/features/build_features.py data/processed/smarthome-userstories.csv models/tf-idf-scores.csv
```

Using the previous command, the resulting file will be saved on the `/models` folder as `tf-idf-scores.csv`.

## Step 3: K-Means Clustering

With the user stories represented on a vector space by the previous step, we run the `train_model.py` script on
`/src/models` to implement the K-Means clustering algorithm. This script accepts the following arguments:
- _input_filepath_: Specifies the location of the `tf-idf-scores.csv` file from the previous step.
- _output_folder_: Specifies the folder in which the output files will be generated. This included a file with the
clusters for the given value of K and a _.csv_ table with the computed SSE and MSC scores.
- _--k_: Option used to specify the number of centroids to be used by the algorithm. Defaults to 3. 

To generate the clusters and print the algorithm results, execute:

```bash
python src/models/train_model.py models/tf-idf-scores.csv reports/ --k=2
```

The following files will be created on the output folder (in this case, `/reports`):
- _k-means-plot-results.csv_: Table that will be updated with the SSE and MSC scores for value of K chosen.
- _k-means-results-X.txt_: Output file with the user story IDs in each cluster,in which _X_ represents 
the value of K chosen.

## Step 4: Reporting Results

Based on the results of `k-means-plot-results.csv`, running the `visualize.py` script on the `/src/visualization` 
folder will generate a figure for the SSE and the MSC scores measured in the previous step. The first 
argument of the command corresponds to the path of the input `k-means-plot-results.csv` file, while the
second argument specifies the output path for the graphs. 

```bash
python src/visualization/visualize.py reports/k-means-plot-results.csv reports/figures/
```

## Project Organization

    ├── LICENSE
    ├── README.md          <- The top-level README for this project.
    ├── data
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── reports            <- Generated output files produced for each value of K
    │   └── figures        <- Generated graphics for the SSE and MSC scores
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    └── src                <- Source code for use in this project.
        ├── __init__.py    <- Makes src a Python module
        │
        ├── data           <- Scripts to download or generate data (Step 1)
        │   └── make_dataset.py
        │
        ├── features       <- Scripts to turn raw data into features for modeling (Step 2)
        │   └── build_features.py
        │
        ├── models         <- Scripts to train models and then use trained models to make
        │   │                 predictions (Step 3)
        │   └── train_model.py
        │
        └── visualization  <- Scripts to show graphs and results oriented visualizations (Step 4)
            └── visualize.py
