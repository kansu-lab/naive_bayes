#!/usr/bin/env python
import argparse
from util import parse_federalist_papers


def word_probabilities(list_of_reviews, feature_list):
    """calculates probabilities of each feature given this dataset using Laplace smoothing
    returns a dict {feature_1: probability_1, ... feature_n: probability_n}"""
    author_probs = {}
    return author_probs


def score(review, author_prob, feature_probs):
    """Calculates a naive bayes score for a string, given class estimate and feature estimates"""
    tokenized_review = review.strip().lower().split()
    score = 0.0
    return score

def main(data_file, features):
    """extract function word features from a text file"""

    # TODO: create a dictionary from author -> list of essays for two authors we will model
    authors, essays, essay_ids = parse_federalist_papers(data_file)
    essay_dict = {"HAMILTON":[], "MADISON":[]}

    # hold out one review per author to test the model
    training_essays = {author: essays[:-1] for author, essays in essay_dict.items()}
    heldout_essays = {author: essays[-1] for author, essays in essay_dict.items()}

    # TODO estimate author probabilities. Creates a dict {author_1: probability_1, ...}
    author_probs = {}
    print(f"Author prior: {author_probs}")

    # TODO estimate word probabilities per author. Define the function word_probabilities
    author_word_probs = {}
    print(author_word_probs)


    for author in heldout_essays:
        essay = heldout_essays[author]
        essay_snippet = " ".join(essay[500:700].split()) # print a snippet
        print(f"\nChecking heldout essay by {author}")
        print(f"{essay_snippet}")

        for testauthor in heldout_essays:
            this_author_probability = author_probs[testauthor]
            word_probs = author_word_probs[testauthor]
            #TODO define the function score
            prob = score(essay, this_author_probability, word_probs)
            print(f"model probability essay is from {testauthor}: {prob:0.02}")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='artisinal naive bayes lab')
    parser.add_argument('--path', type=str, default="federalist_dev.json",
                        help='path to author data')

    args = parser.parse_args()
    features = ["in", "while", "until", "which", "how"]

    main(args.path, features)
