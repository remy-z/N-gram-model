# !/usr/bin/python3
import math
import general
import random 

class LanguageModel:

    def __init__(self):
        pass
    
    vocab_size = 0
    unigram_counts = {}
    bigram_counts = {}
    bigram_probs = {}
    unigram_probs = {}
    trigram_counts = {}
    trigram_probs = {}
    total_tokens = 0
    train_corpus = ""

    def train(self, train_corpus):
        LanguageModel.train_corpus = train_corpus
        # Opens train_corpus and save it as a list of lists
        train_sentences = general.Tokenizer(general.Opener(train_corpus))
        # UNK the tokenized sentences
        general.Unker(train_sentences)
        # count unigrams
        #print("Calculating Unigram Probabilites...")
        for i in range(len(train_sentences)):
            train_sentences[i].append('</s>')
        LanguageModel.unigram_counts = general.UniCounter(train_sentences)
        # exclude <s> from vocab size
        LanguageModel.vocab_size = len(LanguageModel.unigram_counts) - 1
        LanguageModel.total_tokens = len([i for x in train_sentences for i in x])  # this is our N for unigrams

        # calculate unigram probabilities
        for key in LanguageModel.unigram_counts:
            probability = math.log(
                ((LanguageModel.unigram_counts[key]) / LanguageModel.total_tokens), 2)
            # probability = round(probability, 3)
            LanguageModel.unigram_probs.update({key: probability})    
        
        #print("Calculating Bigram Probabilites...")
        # Add <s> and </s> tags for bigrams
        for i in range(len(train_sentences)):
            train_sentences[i].insert(0, '<s>')

        # recount unigrams for bigrams since we have sentence tags
        LanguageModel.unigram_counts = general.UniCounter(train_sentences)

        # count bigrams
        LanguageModel.bigram_counts = general.BiCounter(train_sentences)

        # get vocab size without counting <s>
        LanguageModel.vocab_size = len(LanguageModel.unigram_counts) - 1

        # determine probabilities using log calculation
        for k in LanguageModel.bigram_counts:
            for nk in LanguageModel.bigram_counts[k]:
                probability = math.log(((LanguageModel.bigram_counts[k][nk]) / (
                            LanguageModel.unigram_counts[nk])), 2)
                LanguageModel.bigram_probs.update({"{} {}".format(nk, k): probability})
        
        #print("Calculating Trigram Probabilites...")
        # insert <s> for trigram counting
        for i in range(len(train_sentences)):
            train_sentences[i].insert(0, '<s>')

        # recount bigram and count trigram
        LanguageModel.bigram_counts = general.BiCounter(train_sentences)
        LanguageModel.trigram_counts = general.TriCounter(train_sentences)

        # substract <s> tag
        LanguageModel.vocab_size = len(LanguageModel.trigram_counts) - 1

        for k in LanguageModel.trigram_counts:
            for nk in LanguageModel.trigram_counts[k]:
                probability = math.log( ((LanguageModel.trigram_counts[k][nk]) / (LanguageModel.bigram_counts[nk[1]][nk[0]])), 2)
                LanguageModel.trigram_probs.update({f"{nk[0]} {nk[1]} {k}": probability})
        """
        #sort and output the probabilites
        trigram_probs_sorted = dict(sorted(LanguageModel.trigram_probs.items(), key = lambda x: (-x[1], x[0])))
        output_this = ""
        for key in trigram_probs_sorted:
            output_this += f"{key} {round(trigram_probs_sorted[key], 3)} \n"
        print("Trigram probabilites: ")
        print(output_this)
        """

    def score(self, test_corpus, lambda1, lambda2, lambda3):
        #lambda1 is for trigram, 2 for bigram, 3 for unigram
        # add this back in eventually: , lambda1, lambda2, lambda3

        # process test_corpus
        test_sentence_strings = general.Opener(test_corpus)
        test_sentences = general.Tokenizer(test_sentence_strings)
        general.Unker(test_sentences, LanguageModel.unigram_counts)

        # insert our sentence tags
        for i in range(0, len(test_sentences)):
            test_sentences[i].insert(0, '<s>')
            test_sentences[i].insert(0, '<s>')
            test_sentences[i].append('</s>')

        prob_cum_sum = 0
        word_count = 0
        list_of_probs = []

        for i in range(0, len(test_sentences)):
            sen_prob = 0
            for j in range(2, len(test_sentences[i])):
                word_count += 1
                tri_prob = 0
                bi_prob = 0
                uni_prob = 0
                # trigram probability
                if f"{test_sentences[i][j-2]} {test_sentences[i][j-1]} {test_sentences[i][j]}" in LanguageModel.trigram_probs:
                    tri_prob +=  (lambda1 * math.pow(2, LanguageModel.trigram_probs[f"{test_sentences[i][j-2]} {test_sentences[i][j-1]} {test_sentences[i][j]}"]))

                # bigram probability
                if f"{test_sentences[i][j-1]} {test_sentences[i][j]}" in LanguageModel.bigram_probs:
                    bi_prob +=  (lambda2 * math.pow(2, LanguageModel.bigram_probs[f"{test_sentences[i][j-1]} {test_sentences[i][j]}"]))
                    
                #unigram probability
                uni_prob += (lambda3 * math.pow(2, LanguageModel.unigram_probs[test_sentences[i][j]]))
                word_prob = uni_prob + bi_prob + tri_prob
                sen_prob += math.log2(word_prob)

            prob_cum_sum += sen_prob       
            list_of_probs.append(sen_prob)

        sentences_and_probs = list(zip(test_sentence_strings, list_of_probs))

        h = (-1 / word_count) * (prob_cum_sum)
        perplexity = round(math.pow(2, h), 3)
        return perplexity
        """
        print("Test sentence probabilites:")
        output_this = ""
        for tuple in sentences_and_probs:
            output_this += f"{tuple[0]} {round(tuple[1], 3)} \n"
        print(output_this)
        print()
        print("Trigram Perplexity, Linear Interpolation: " + str(perplexity))
        """


                