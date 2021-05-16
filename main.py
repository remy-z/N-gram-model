# !/usr/bin/python3
from argparse import ArgumentParser

from bigram import LanguageModel as BigramModel
from trigram import LanguageModel as TrigramModel
from unigram import LanguageModel as UnigramModel
import re
import math


def main():
    '''
    Interact with a LanguageModel() object via the command line.

    When this script is called from the command line, this function will
    instantiate a LanguageModel() object and store it as the variable `lm`.

    It will then train the language model on the provided training corpus
    by calling:

        `lm.train()`

    If a dev/test corpus is also provided, it will then evaluate the language
    model on the provided corpus by calling:

        `lm.score()

    To read more about `argparse`, which handles interfacing with the command
    line, check out the documentation:

        https://docs.python.org/3/library/argparse.html

    '''
    # process the command line arguments
    parser = ArgumentParser(description='Interact with a language model!')

    parser.add_argument('train_corpus',
                        help='filepath to training corpus')

    parser.add_argument('-t', '--test_corpus', required=False,
                        help='filepath to dev/test corpus')

    parser.add_argument('-n', '--ngram', default=1, type=int,
                        help='the order of n-gram')

    args = parser.parse_args()

    # determine which language model to instantiate
    if args.ngram == 2:
        LanguageModel = BigramModel

    elif args.ngram == 3:
        LanguageModel = TrigramModel

    else:
        LanguageModel = UnigramModel

    # instantiate the language model
    lm = LanguageModel()

    # train the language model
    print()
    lm_counts = lm.train(args.train_corpus)
    print(lm_counts)
    print()
    # evaluates the language model
    # prints out every sentence in test corpus with it's probability
    # and the entire testset's perplexity
    if args.test_corpus:
        print()
        score = lm.score(args.test_corpus, lm_counts)
        for i in range(0, len(score)):
            print("{}  {}".format(score[i][0],score[i][1]))


    print()


if __name__ == '__main__':
    main()
