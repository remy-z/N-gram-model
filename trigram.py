# !/usr/bin/python3
import math
import general
import random 

# TODO: Implement a Laplace-smoothed trigram model :)
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

    def train(self, train_corpus):
        
        # Opens train_corpus and save it as a list of lists
        train_sentences = general.Tokenizer(general.Opener(train_corpus))

        # UNK the tokenized sentences
        #general.Unker(train_sentences)

        # count unigrams
        LanguageModel.unigram_counts = general.UniCounter(train_sentences)

        # exclude <s> from vocab size
        LanguageModel.vocab_size = len(LanguageModel.unigram_counts)

        LanguageModel.total_tokens = len([i for x in train_sentences for i in x])  # this is our N for unigrams

        # calculate unigram probabilities
        for key in LanguageModel.unigram_counts:
            probability = math.log(
                ((LanguageModel.unigram_counts[key]) / LanguageModel.total_tokens), 2)
            # probability = round(probability, 3)
            LanguageModel.unigram_probs.update({key: probability})

        # print stuff to check it's equal to other unigrams
        unigram_probs_sorted = dict(sorted(LanguageModel.unigram_probs.items(), key=lambda x: (-x[1], x[0])))
        #for key in unigram_probs_sorted:
        #    print("{} {}".format(key, round(unigram_probs_sorted[key], 3)))
        print()
        print()


        # Add <s> and </s> tags for bigrams
        for i in range(len(train_sentences)):
            train_sentences[i].insert(0, '<s>')
            train_sentences[i].append('</s>')

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
                # probability = round(probability, 3)
                LanguageModel.bigram_probs.update({"{} {}".format(nk, k): probability})

        # print stuff for general clarity and checking against our other bigram model
        bigram_probs_sorted = dict(sorted(LanguageModel.bigram_probs.items(), key = lambda x: (-x[1], x[0])))

        #for key in bigram_probs_sorted:
        #    print("{} {}".format(key, round(bigram_probs_sorted[key], 3)))

        print()
        print()


        # insert <s> for trigram counting
        for i in range(len(train_sentences)):
            train_sentences[i].insert(0, '<s>')

        # recount unigram and bigram and count trigram
        LanguageModel.unigram_counts = general.UniCounter(train_sentences)
        LanguageModel.bigram_counts = general.BiCounter(train_sentences)
        LanguageModel.trigram_counts = general.TriCounter(train_sentences)

        # substract <s> tag
        LanguageModel.vocab_size = len(LanguageModel.trigram_counts) - 1

        for k in LanguageModel.trigram_counts:
            for nk in LanguageModel.trigram_counts[k]:
                probability = math.log( ((LanguageModel.trigram_counts[k][nk]) / (LanguageModel.bigram_counts[nk[1]][nk[0]])), 2)
                #probability = round(probability, 3)
                LanguageModel.trigram_probs.update({"{} {} {}".format(nk[0], nk[1], k): probability})
        
        #sort the probabilites
        trigram_probs_sorted = dict(sorted(LanguageModel.trigram_probs.items(), key = lambda x: (-x[1], x[0])))

        #output the sorted probabilites
        #for key in trigram_probs_sorted:
        #    print("{} {}".format(key, round(trigram_probs_sorted[key], 3)))
          
        

    def score(self, test_corpus):

        # count type of ngram for sanity
        unigrams = 0
        bigrams = 0
        trigrams = 0
        unseen_end = 0

        # process test_corpus
        test_sentence_strings = general.Opener(test_corpus)
        test_sentences = general.Tokenizer(test_sentence_strings)
        general.Unker(test_sentences, LanguageModel.unigram_counts)

        # insert 2 sentence tags for trigram and </s>
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
                # stupid backoff implementation using unigram, bigram and trigram probabilities
                # If we have n-2, n-1, n trigram use that probability
                if "{} {} {}".format(test_sentences[i][j-2], test_sentences[i][j-1], test_sentences[i][j]) in LanguageModel.trigram_probs:
                    sen_prob += LanguageModel.trigram_probs["{} {} {}".format(test_sentences[i][j-2], test_sentences[i][j-1], test_sentences[i][j])]
                    prob_cum_sum += LanguageModel.trigram_probs["{} {} {}".format(test_sentences[i][j-2], test_sentences[i][j-1], test_sentences[i][j])]
                    # print("In trigram probs: " + "{} {} {}".format(test_sentences[i][j-2], test_sentences[i][j-1], test_sentences[i][j]))
                    trigrams += 1
                # if we don't have that trigram, check for n-1, n bigram
                elif "{} {}".format(test_sentences[i][j-1], test_sentences[i][j]) in LanguageModel.bigram_probs:
                    sen_prob += math.log(0.4, 2) + LanguageModel.bigram_probs["{} {}".format(test_sentences[i][j-1], test_sentences[i][j])]
                    prob_cum_sum += math.log(0.4, 2) + LanguageModel.bigram_probs["{} {}".format(test_sentences[i][j-1], test_sentences[i][j])]
                    # print("In bigram probs: " + "{} {} {}".format(test_sentences[i][j-2], test_sentences[i][j-1], test_sentences[i][j]))
                    # print("    " + "{} {}".format(test_sentences[i][j-1], test_sentences[i][j]))
                    bigrams += 1
                # unseen trigram and bigram so just do unigram calc
                else:
                    # improv solution for calculating unigram prob for end sentence token
                    if test_sentences[i][j] == "</s>":
                        sen_prob += math.log(0.16, 2) + math.log(LanguageModel.unigram_counts["</s>"] / (LanguageModel.unigram_counts["</s>"] + LanguageModel.unigram_counts["<s>"] + LanguageModel.total_tokens), 2)
                        prob_cum_sum += math.log(0.16, 2) + math.log(1 / (len(LanguageModel.unigram_counts) - 1), 2)
                        unseen_end +=1
                    # general case
                    else:
                        sen_prob += math.log(0.16, 2) + LanguageModel.unigram_probs[test_sentences[i][j]]
                        prob_cum_sum += math.log(0.16, 2) + LanguageModel.unigram_probs[test_sentences[i][j]]
                    # print("Sadge: " + "{} {} {}".format(test_sentences[i][j-2], test_sentences[i][j-1], test_sentences[i][j]))
                    # print("    " + test_sentences[i][j])
                    unigrams += 1
            list_of_probs.append(sen_prob)
        sentences_and_probs = list(zip(test_sentence_strings, list_of_probs))

        h = (-1 / word_count) * (prob_cum_sum)
        perplexity = round(math.pow(2, h), 3)

        sentences_and_probs.append(("perplexity: ", perplexity))

        for i in range(len(sentences_and_probs)):
            print("{} {}".format(sentences_and_probs[i][0], sentences_and_probs[i][1]))

        print("Unigrams: " + str(unigrams))
        print("Bigrams: " + str(bigrams))
        print("Trigrams: " + str(trigrams))
        print("unseen </s>: " + str(unseen_end))

    def shannon(self):
        shannon_list = []
        for key in LanguageModel.trigram_probs:
            trigram_prob = (key.split(), math.pow(2,LanguageModel.trigram_probs[key]))
            if "<UNK>" not in trigram_prob[0]:
                shannon_list.append(trigram_prob)

        print("Shannon Visualization using trigram: ")
        for i in range(10):
            end_sentence = False
            viz = ""
            last_bigram = ("<s>","<s>")
            
            while not end_sentence: 
                current_word = []
                current_prob = []
                
                for item in shannon_list:
                    if (item[0][0],item[0][1]) == last_bigram:
                        current_word.append((item[0][2]))
                        current_prob.append(item[1])
            
                choice = random.choices(current_word, current_prob, k=1)
                if choice[0] == "</s>":
                    end_sentence = True
                else:
                    last_bigram = (last_bigram[1],choice[0])
                    viz += f"{choice[0]} "
                
            print(viz)
