# -*- coding: utf-8 -*-
import click
import logging
from dotenv import find_dotenv, load_dotenv
import math
import pandas as pd
import copy


def get_vocabulary_set(dataset):
    """ Create the keyword associated to the position of the elements within the document vectors """
    vocabulary_string = ""
    for column in dataset:
        if column != 'id':
            column_list = dataset[column].tolist()
            vocabulary_string = vocabulary_string + " ".join(map(str, column_list))

    tokens = vocabulary_string.split(' ')
    vocabulary_set = set(tokens)
    return vocabulary_set


def get_vector_bow(row):
    """ Given a dataframe row with an user story, return the bag of words (BoW) across the
        'Role', 'Feature' and 'Benefit' columns of that user story
    """
    bow_role = str(row['role']).split(' ')
    bow_feature = str(row['feature']).split(' ')
    bow_benefit = str(row['benefit']).split(' ')

    bow_total = set(bow_role).union(set(bow_feature)).union(set(bow_benefit))
    return bow_total


def populate_vector_dimensions(vector_dict, dataset):
    """
    Returns a vector dictionary with the dimension values populated based on the dataset info
    """
    vector_dict_copy = copy.deepcopy(vector_dict)
    for index, row in dataset.iterrows():
        bow = get_vector_bow(row)
        for word in bow:
            vector_id = row['id']
            vector_dict_copy[vector_id][word] = vector_dict_copy[vector_id][word] + 1
    return vector_dict_copy


def compute_tf(vector, bow):
    """
    Computes the Term Frequency of a vector, where:
        tf(w) = (Number of times the word appears in a user story) / (Total number of words in the user story)
    """
    tf_dict = {}
    bow_count = len(bow)
    for word, count in vector.items():
        tf_dict[word] = count / float(bow_count)
    return tf_dict


def compute_idf(dataset):
    """
    Computes the Inverse Document Frequency of all words, where:
        idf(w) = log(Number of user stories / Number of user stories that contain word w )
    """
    total_rows = len(dataset)

    # counts the number of documents that contain a word w
    idf_dict = dict.fromkeys(get_vocabulary_set(dataset), 0)
    for index, row in dataset.iterrows():
        # the fuck is this
        bow_row = get_vector_bow(row)
        for word in bow_row:
            idf_dict[word] += 1

    # divide total_rows by denominator above, take the log of that
    for word, val in idf_dict.items():
        if float(val) > 0:
            idf_dict[word] = math.log(total_rows / float(val))

    return idf_dict


def compute_tfidf(tf_bow, idfs):
    """
    Computes the TF-IDF of a user story, based on the TF and IDF results
    """
    tfidf = {}
    for word, val in tf_bow.items():
        tfidf[word] = val * idfs[word]
    return tfidf


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """
    Starting point of the project. Receives the location of the dataset as a
    command-line Path argument.
    """
    logger = logging.getLogger(__name__)
    logger.info('Making final data set from raw data')

    # Get the data/processed/smarthome-userstories.csv file
    # Also excludes the last column from each row due to extra commas on the csv
    dataset = pd.read_csv(input_filepath, usecols=['id', 'role', 'feature', 'benefit'])
    logger.info('Loaded data file ' + input_filepath + ' with ' + str(len(dataset)) + ' rows')

    # Generates the set with all the words used across all user stories
    vocabulary_set = get_vocabulary_set(dataset)

    # List of vector dictionaries, in which each vector is a user story
    # Each vector has another dictionary with vector dimensions of each word from the vocabulary set
    vector_dict = dict.fromkeys(dataset['id'].tolist())

    for key, value in vector_dict.items():
        vector_dict[key] = dict.fromkeys(vocabulary_set, 0)

    logger.info('Populating the user story vectors based on the dataset...')
    vector_dict_populated = populate_vector_dimensions(vector_dict, dataset)
    idfs = compute_idf(dataset)

    # Generate the tf-idf scores dictionary
    logger.info('Generating the TF-IDF scores for each vector...')
    tfidf_scores_dict = {}
    for index, row in dataset.iterrows():
        row_id = row['id']
        tf_bow = compute_tf(vector_dict_populated[row_id], get_vector_bow(row))
        tfidf_bow = compute_tfidf(tf_bow, idfs)
        tfidf_scores_dict[row_id] = tfidf_bow

    # Generate new DataFrame with the TF-IDF scores
    logger.info('Saving TF-IDF scores in a new .csv file...')
    tfidf_scores = []
    tfidf_ids = []
    for data in tfidf_scores_dict.items():
        tfidf_ids.append(data[0])
        tfidf_scores.append(data[1])
    dataframe_scores = pd.DataFrame(tfidf_scores, index=tfidf_ids, columns=get_vocabulary_set(dataset))

    # Save the TF-IDF scores on data models/tf-idf-scores.csv
    logger.info('Saved processed scores on ' + output_filepath + ' with ' + str(len(dataset)) + ' rows')
    dataframe_scores.to_csv(output_filepath, encoding='utf-8')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    # project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()