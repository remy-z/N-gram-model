# !/usr/bin/python3
import re
import math
import globalfunctions as gf

#get our global functions
gfuncs = gf.GlobalFunctions()

# TODO: Implement a Laplace-smoothed unigram model :)
class LanguageModel:

    def __init__(self):
        pass

    def train(self, train_corpus):
        
        #get our global functions
        #funcs = gf.GlobalFunctions()
       
        # opens train_corpus and returns it as a list (sentences) of lists (tokens)
        list_of_sentences = gfuncs.Opener(train_corpus)

        # Tokenize the list of sentences (split sentences into a list of their tokens)
        tokenized_sentences = gfuncs.Tokenizer(list_of_sentences)

        # UNK the tokenized sentences
        unked_sentences = gfuncs.Unker(tokenized_sentences)
        
        # calculate counts with the UNKed set
        unigram_counts = gfuncs.UniCounter(unked_sentences)
        
        # every key in the unigram_counts dictionary is one unique word, so our vocab size 
        # is equal to len(unigram_counts)-1 because we don't want to inlcude UNK
        vocab_size = len(unigram_counts) - 1            
        

        # make a dictionary with our probabilites (in log form with laplace smoothing)
        
        unigram_probs = {}
        total_tokens = len([i for x in list_of_sentences for i in x]) # this is our N for unigrams
        
        for key in unigram_counts:
            probability = math.log(((unigram_counts[key][0] + 1) / (total_tokens + vocab_size)), 2)
            probability = round(probability, 3)
            unigram_probs.update({key: probability})
        
        # Make a sorted dictionary 
        # Sorted first by probability (descending) then alphabetically for keys with the same value
        unigram_probs_sorted = dict(sorted(unigram_probs.items(), key = lambda x: (-x[1], x[0])))
        
        return unigram_probs_sorted



    def score(self, test_corpus, unigram_probs):
              
        
        # open test_corpus and save as a list of sentences 
        test_sentences = gfuncs.Opener(test_corpus)
        random_list = ["hello I am"]
        #tokenize the sentences
        tokenized_test = gfuncs.Tokenizer(test_sentences)
        
        #UNK everything in the test set that doesn't appear in our vocabulary (isn't a key in unigram_probs)
        unked_test_set = gfuncs.Unker(tokenized_test, unigram_probs)

        """
        Calculate the probability
        all of our probabilities are log probabilities, so we can just 
        add probabilites for each individual sentence,
        and keep a running probability for calculating perplexity
        """

        prob_cum_sum = 0   # this is the sum of probabilities for every sentence
        word_count = 0     # this is our N for calculating perplexity
        list_of_probs = [] #keep track of our probabilites 
        
        for i in range(0, len(unked_test_set)):
            #keeps track of probability for each sentence
            sen_prob = 0 
            
            for j in range(0, len(unked_test_set[i])):
                word_count += 1
                sen_prob += unigram_probs[unked_test_set[i][j]]
                prob_cum_sum += unigram_probs[unked_test_set[i][j]]
            
            list_of_probs.append(round(sen_prob,3))
        
        # make a list of sentences and their corresponding probabilities to return 
        sentences_and_probs = list(zip(test_sentences, list_of_probs))

        #calculate perplexity and add it as the list entry to the list we return
        h = (-1 / word_count) * (prob_cum_sum)
        perplexity = round(math.pow(2,h), 3)
        
        #add perplexity as the last index of the list we return
        sentences_and_probs.append(("perplexity: ", perplexity))
        
        return sentences_and_probs
