# !/usr/bin/python3
import math
import general

class LanguageModel:

    def __init__(self):
        pass
    
    vocab_size = 0
    unigram_counts = {}
    unigram_probs = {}

    def train(self, train_corpus, output = True):
           
        # Opens train_corpus and save it as a list of lists
        train_sentences = general.Tokenizer(general.Opener(train_corpus))
        # UNK the tokenized sentences
        general.Unker(train_sentences)
        # calculate counts with the UNKed set
        LanguageModel.unigram_counts = general.UniCounter(train_sentences)
        LanguageModel.vocab_size = len(LanguageModel.unigram_counts)                       
        total_tokens = len([i for x in train_sentences for i in x]) # this is our N for unigrams
        
        for key in LanguageModel.unigram_counts:
            
            probability = math.log(((LanguageModel.unigram_counts[key] + 1) / (total_tokens + LanguageModel.vocab_size)), 2)
            #probability = round(probability, 3)
            LanguageModel.unigram_probs.update({key: probability})
        
        if output:
            unigram_probs_sorted = dict(sorted(LanguageModel.unigram_probs.items(), key = lambda x: (-x[1], x[0])))
            for key in unigram_probs_sorted:
                print("{} {}".format(key, round(unigram_probs_sorted[key], 3)))



    def score(self, test_corpus, output = True):    
        
        # Open test_corpus and save as a list of sentences 
        test_sentence_strings = general.Opener(test_corpus) 
        # Turn sentences into list of lists
        test_sentences = general.Tokenizer(test_sentence_strings)
        # UNK everything in the test set that doesn't appear in our vocabulary
        general.Unker(test_sentences, LanguageModel.unigram_counts)

        prob_cum_sum = 0   # cumulative probability
        word_count = 0     # N for calculating perplexity
        list_of_probs = [] # probabilites for each sentence 
        
        for i in range(0, len(test_sentences)):
            
            #keeps track of probability for each sentence
            sen_prob = 0 
            
            for j in range(0, len(test_sentences[i])):
                
                word_count += 1
                sen_prob += LanguageModel.unigram_probs[test_sentences[i][j]]
                prob_cum_sum += LanguageModel.unigram_probs[test_sentences[i][j]]
            
            list_of_probs.append(sen_prob)
        
        # make a list of sentences and their corresponding probabilities to return 
        sentences_and_probs = list(zip(test_sentence_strings, list_of_probs))

        #calculate perplexity and add it as the list entry to the list we return
        h = (-1 / word_count) * (prob_cum_sum)
        perplexity = round(math.pow(2,h), 3)
        
        if output:
            for i in range(len(sentences_and_probs)):
                print("{}  {}".format(sentences_and_probs[i][0],round(sentences_and_probs[i][1], 3)))
        print("Unigram Perplexity, Laplace Smoothing: " + str(perplexity))
