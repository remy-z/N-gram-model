# !/usr/bin/python3
from argparse import ArgumentParser

from bigram import LanguageModel as BigramModel
from trigram import LanguageModel as TrigramModel
from unigram import LanguageModel as UnigramModel


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
    lm.train(args.train_corpus)

    # evaluates the language model
    if args.test_corpus:
        print()
        lm.score(test_corpus=args.test_corpus)

    print()


if __name__ == '__main__':
    main()
