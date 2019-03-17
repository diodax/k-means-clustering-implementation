# -*- coding: utf-8 -*-
import json
import os
import nltk
import click
import logging
from pathlib import Path
from nltk.stem import WordNetLemmatizer
from dotenv import find_dotenv, load_dotenv
import pandas as pd
from nltk.corpus import wordnet


def get_wordnet_pos(tag):
    """Map POS tag to first character lemmatize() accepts"""
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag.upper(), wordnet.NOUN)


def get_stopwords(filename):
    """
    Given a .JSON file location with a list of stopwords, extract it and return them as an array
    """
    # Open our JSON file and load it into python
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)) as json_data:
        data = json.load(json_data)
    return data


def preprocess(text):
    """
    Applies the following pre-processing steps to a given string:
    Step 1: Convert to lower case, replace slashes (/) with spaces
    Step 1: Tokenize the strings
    Step 3: Retain only nouns, verbs, adjectives and adverbs (using a POS-tagger)
    Step 4: Remove default English stopwords
    Step 5: Remove custom domain stopwords
    Step 6: Lemmatize each word
    """
    tokens = nltk.word_tokenize(text.lower().replace('/', ' '))
    tagged_tokens = nltk.pos_tag(tokens)
    filtered_tokens = [t for t in tagged_tokens if t[1] in ["NN", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ", "JJ", "RB"]]

    # Remove stop words
    standard_stop_words = get_stopwords("stopwords.json")
    custom_stop_words = get_stopwords("stopwords_custom.json")
    tokens_no_standr_stopwords = [word for word in filtered_tokens if word[0] not in standard_stop_words]
    tokens_no_custom_stopwords = [word for word in tokens_no_standr_stopwords if word[0] not in custom_stop_words]

    # Lemmatize
    wordnet_lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [wordnet_lemmatizer.lemmatize(w[0], get_wordnet_pos(w[1])) for w in tokens_no_custom_stopwords]

    preprocessed_text = ' '.join(lemmatized_tokens)
    return preprocessed_text


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('Making final data set from raw data')

    # Get the data/raw/smarthome-userstories-1k.csv file
    # Also excludes the last column from each row due to extra commas on the csv
    dataset = pd.read_csv(input_filepath, usecols=['id', 'role', 'feature', 'benefit'])
    logger.info('Loaded data file ' + input_filepath + ' with ' + str(len(dataset)) + ' rows')

    # Download NLTK resources
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('wordnet')

    logger.info('Applying pre-processing steps on the "Role" column...')
    dataset['role'] = dataset['role'].apply(preprocess)
    logger.info('Applying pre-processing steps on the "Feature" column...')
    dataset['feature'] = dataset['feature'].apply(preprocess)
    logger.info('Applying pre-processing steps on the "Benefit" column...')
    dataset['benefit'] = dataset['benefit'].apply(preprocess)

    # Save the processed subset on data data/processed/smarthome-userstories.csv
    logger.info('Saved processed results on ' + output_filepath + ' with ' + str(len(dataset)) + ' rows')
    dataset.to_csv(output_filepath, encoding='utf-8', index=False)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
