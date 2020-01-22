#!/usr/bin/env python
import argparse
from util import load_function_words, parse_federalist_papers, labels_to_key, labels_to_y, split_data
import numpy as np
from collections import Counter
from sklearn.naive_bayes import MultinomialNB, BernoulliNB

# TODO create a function that loads all the essays into a matrix
def load_features(list_of_essays, list_of_features):

    X = np.zeros((0, 0), dtype=np.int)

    return X


def main(data_file, vocab_path):
    """Build and evaluate Naive Bayes classifiers for the federalist papers"""

    function_words = load_function_words(vocab_path)

    authors, essays, essay_ids = parse_federalist_papers(data_file)

    function_words = load_function_words(vocab_path)

    # load the attributed essays into a feature matrix
    X = load_features(essays, function_words)
    # TODO: load the author names into a vector y, mapped to 0 and 1, using functions from util.py
    labels_map = labels_to_key(authors)
    y = np.asarray(labels_to_y(authors, labels_map))

    print(f"Numpy array has shape {X.shape} and dtype {X.dtype}")

    # TODO shuffle, then split the data



    # TODO: train a multinomial NB model, evaluate on validation split


    # TODO: train a Bernoulli NB model, evaluate on validation split


    # TODO: fit the zero rule




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='feature vector homework')
    parser.add_argument('--path', type=str, default="federalist_dev.json",
                        help='path to author dataset')
    parser.add_argument('--function_words_path', type=str, default="ewl_function_words.txt",
                        help='path to the list of words to use as features')
    args = parser.parse_args()

    main(args.path, args.function_words_path)
