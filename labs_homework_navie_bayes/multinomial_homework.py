#!/usr/bin/env python
import argparse
from util import load_function_words, parse_federalist_papers, \
    labels_to_key, labels_to_y, split_data, find_zero_rule_class, apply_zero_rule, calculate_accuracy
import numpy as np
from nltk import word_tokenize
from collections import Counter
from sklearn.naive_bayes import MultinomialNB, BernoulliNB


# TODO create a function that loads all the essays into a matrix
def load_features(list_of_essays, list_of_features):
    X = np.zeros((len(list_of_essays), len(list_of_features)), dtype=np.int)
    for i, essay in enumerate(list_of_essays):
        word_in_essay = word_tokenize(essay.lower())
        for j, f in enumerate(list_of_features):
            feature = [w for w in word_in_essay if w == f]
            X[i, j] = len(feature)
    # IndexError: index 0 is out of bounds for axis 0 with size 0  length not enough
    return X


def main(data_file, vocab_path):
    """Build and evaluate Naive Bayes classifiers for the federalist papers"""

    authors, essays, essay_ids = parse_federalist_papers(data_file)

    function_words = load_function_words(vocab_path)
    # load the attributed essays into a feature matrix
    # label mapping is for me to track
    # make them into two classifiers, zero and one.
    # the distribution of  the zero (ham) was higher？
    # the distribution of one (man) was higher?
    # output: two classes zero and one

    X = load_features(essays, function_words)
    # TODO: load the author names into a vector y, mapped to 0 and 1, using functions from util.py

    labels_map = labels_to_key(authors)
    # y output, a list of zeros and ones, 相对应，第几篇文章里面是什么
    # y is the golden standard, it is used for both training, and evaluation
    y = np.asarray(labels_to_y(authors, labels_map))
    # numerical
    print(f"Numpy array has shape {X.shape} and dtype {X.dtype}")

    # TODO shuffle, then split the data
    # if split has already had a shuffle function embedded in it, no need for importing
    train, test = split_data(X, y, 0.25)

    # TODO: train a multinomial NB model, evaluate on validation split
    nbm = MultinomialNB()
    # to see what is the definition of nbm, what it requires as in the parameter
    # train is array, two tuples with [] in it, the first one is a array, teh second one is target
    # rows of X and the len of y are not identical.
    # y 的长度要大于X， 不能直接用y, 需要用剪裁过在train 里面的
    nbm.fit(train[0], train[1])  # change
    preds_nbm = nbm.predict(test[0])
    test_y = test[1]
    accuracy = calculate_accuracy(preds_nbm, test_y)

    print(f" the accuracy for multinomial NB model is {accuracy}")

    # TODO: train a Bernoulli NB model, evaluate on validation split

    nbb = BernoulliNB()
    nbb.fit(train[0], train[1])
    preds_nbb = nbb.predict(test[0])
    accuracy = calculate_accuracy(preds_nbb, test_y)

    print(f" the accuracy for Bernoulli NB model is {accuracy}")

    # TODO: fit the zero rule
    train_y = train[1]
    most_frequent_class = find_zero_rule_class(train_y)
    test_predictions = apply_zero_rule(test[0], most_frequent_class)
    test_accuracy= calculate_accuracy(test_predictions,test_y)
    print(f" the accuracy for the baseline is {accuracy}")




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='feature vector homework')
    parser.add_argument('--path', type=str, default="federalist_dev.json",
                        help='path to author dataset')
    parser.add_argument('--function_words_path', type=str, default="ewl_function_words.txt",
                        help='path to the list of words to use as features')
    args = parser.parse_args()

    main(args.path, args.function_words_path)
