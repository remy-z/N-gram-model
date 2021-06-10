# !/usr/bin/python3
from argparse import ArgumentParser

from bigram import LanguageModel as BigramModel
from trigram import LanguageModel as TrigramModel
from unigram import LanguageModel as UnigramModel
from trigram_laplace import LanguageModel as LapTrigramModel
from trigram_interpolation import LanguageModel as InterpTrigramModel

def main():
    # process the command line arguments
    parser = ArgumentParser(description='Interact with a language model!')

    parser.add_argument('train_corpus',
                        help='filepath to training corpus')

    parser.add_argument('-t', '--test_corpus', required=False,
                        help='filepath to dev/test corpus')

    parser.add_argument('-n', '--ngram', default=1, type=int,
                        help='the order of n-gram')
    
    parser.add_argument('-s', '--shannon', required = False, type= int,
                        help = 'Number of Shannon Visualization sentences to generate')


    args = parser.parse_args()

    # determine which language model to instantiate
    if args.ngram == 2:
        LanguageModel = BigramModel

    elif args.ngram == 3:
        LanguageModel = TrigramModel
    #elif args.ngram == 4:
    #    LanguageModel = LapTrigramModel
    elif args.ngram == 5:
        LanguageModel = InterpTrigramModel

    else:
        LanguageModel = UnigramModel

    # instantiate the language model
    lm = LanguageModel()

    # train the language model
    lm.train(args.train_corpus)
    print()
    
    # evaluates the language model
    #if args.test_corpus:    
            #lm.score(args.test_corpus)   
    print()
    HYPERPARAMETER = {}
    for i in range(0,100,1):
        lambda1 = i
        for j in range (0, 100-lambda1, 1):
            lambda2 = j
            lambda3 = (100-j-i)
            lambda1, lambda2, lambda3 = lambda1, lambda2/100, lambda3/100
            perplexity = lm.score(args.test_corpus, lambda1/100, lambda2, lambda3)
            HYPERPARAMETER.update({f"tri: {lambda1/100} bi: {lambda2} uni: {lambda3}": perplexity})
            print(f"tri: {lambda1/100} bi: {lambda2} uni: {lambda3} | {perplexity}")

    the_best = sorted(HYPERPARAMETER.items(), key = lambda x: (x[1]))
    print(the_best)

    #Generate sentences with Shannon Visualization
    if args.shannon:
        lm.shannon(args.shannon)


if __name__ == '__main__':
    main()
