# -*- coding: utf-8 -*-
import click
import logging
import pandas as pd
from k_means import KMeans
import os.path

# String constants
PLOT_TABLE_FILENAME = 'k-means-plot-results.csv'
MODEL_REPORT_FILENAME = 'k-means-results.txt'


def generate_report(clusters, sse_score, msc_score, filepath):
    """
    Generates the txt file at filepath with the clustering results.
    """
    with open(filepath, 'w') as file:
        for key, value in clusters.items():
            # Print the header for the cluster
            file.write("Cluster " + str(key) + ": ")
            # Print the IDs for the user stories inside that cluster
            user_stories_list = []
            for i in value:
                value_text = str(int(i))
                user_stories_list.append(value_text)
            user_stories_string = ", ".join(user_stories_list)
            file.write(user_stories_string + "\n")
        file.write("SSE: " + str(sse_score) + ", MSC: " + str(msc_score) + "\n")


def create_plot_results_table():
    data = []
    k_range = range(2,11)
    for i in k_range:
        data.append([i, 0, 0])
    df = pd.DataFrame(data, columns=['K Size', 'SSE Score', 'MSC Score'], dtype=float)
    df.set_index('K Size', inplace=True)
    return df


def update_plot_results_table(df, tuple_k_scores):
    k_size = tuple_k_scores[0]
    df.loc[[k_size], ['SSE Score']] = tuple_k_scores[1]
    df.loc[[k_size], ['MSC Score']] = tuple_k_scores[2]
    return df


def generate_vector_dict(dataset):
    """
    Generates a dictionary that uses the user story vectors (tuples) as keys, with the user story IDs as values.
    """
    vector_dict = {}
    dataset_list = dataset.values.tolist()

    for row in dataset_list:
        id = row[0]
        vector = row[1:]
        vector_dict[tuple(vector)] = id
    return vector_dict


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_folder', type=click.Path())
@click.option('--k', default=3, help='Number of centroids.')
def main(input_filepath, output_folder, k):
    """
    Receives the location of the tf-idf scores as a
    command-line Path argument.
    """
    logger = logging.getLogger(__name__)
    logger.info('Training the K-Means clustering algorithm based on the TF-IDF scores')

    # Get the models/tf-idf-scores.csv file
    dataset = pd.read_csv(input_filepath)
    logger.info('Loaded data file ' + input_filepath + ' with ' + str(len(dataset)) + ' rows')

    # Removes the first column and formats it like a list
    x = dataset.drop(dataset.columns[0], axis=1).values
    vector_dict = generate_vector_dict(dataset)

    # Number of clusters and max. number of iterations
    km = KMeans(k=k, max_iterations=500)
    km.fit(x)
    clusters = km.get_clusters(vector_dict)

    # Based on the value of K used, change the destination filename
    filepath_list = (output_folder + MODEL_REPORT_FILENAME).rsplit('.', 1)
    output_filepath = filepath_list[0] + '-' + str(k) + '.' + filepath_list[1]

    # Calculate SSE and MSC
    sse_score = km.get_sse_score()
    logger.info('SSE Score: ' + str(sse_score))
    msc_score = km.get_msc_avg()
    logger.info('MSC Score: ' + str(msc_score))

    # Generate the results report
    generate_report(clusters, sse_score, msc_score, output_filepath)
    logger.info('Created report file on ' + output_filepath)

    # Generate / Update the results table for future plots
    if os.path.isfile(output_folder + PLOT_TABLE_FILENAME):
        # Update the existing file
        dataset = pd.read_csv(output_folder + PLOT_TABLE_FILENAME)
        dataset.set_index('K Size', inplace=True)
        k_means_results = update_plot_results_table(dataset, (k, sse_score, msc_score))
    else:
        # Create and update the file
        dataset = create_plot_results_table()
        k_means_results = update_plot_results_table(dataset, (k, sse_score, msc_score))
    k_means_results.to_csv(output_folder + PLOT_TABLE_FILENAME, encoding='utf-8')
    logger.info('Updated report table on ' + output_folder + PLOT_TABLE_FILENAME)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
