# -*- coding: utf-8 -*-
import click
import logging
from dotenv import find_dotenv, load_dotenv
import csv
import pandas as pd
import nltk


def get_vector_keyword_index(dataset):
    """ create the keyword associated to the position of the elements within the document vectors """
    vocabulary_string = ""
    for column in dataset:
        if column != 'id':
            column_list = dataset[column].tolist()
            vocabulary_string = vocabulary_string + " ".join(map(str, column_list))

    tokens = nltk.word_tokenize(vocabulary_string)
    vocabulary_set = set(tokens)
    vector_index = {}
    offset = 0
    # Associate a position with the keywords which maps to the dimension on the vector used to represent this word
    for word in vocabulary_set:
        vector_index[word] = offset
        offset += 1
    return vector_index  # (keyword:position)


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
def main(input_filepath):
    """
    Starting point of the project. Receives the location of the dataset as a
    command-line Path argument.
    """
    logger = logging.getLogger(__name__)
    logger.info('Making final data set from raw data')

    # Get the data/raw/smarthome-userstories-1k.csv file
    # Also excludes the last column from each row due to extra commas on the csv
    dataset = pd.read_csv(input_filepath, usecols=['id', 'role', 'feature', 'benefit'])
    logger.info('Loaded data file ' + input_filepath + ' with ' + str(len(dataset)) + ' rows')

    vector_index = get_vector_keyword_index(dataset)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    # project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()