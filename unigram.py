# !/usr/bin/python3
import math
import general

# TODO: Implement a Laplace-smoothed unigram model :)
class LanguageModel:

    def __init__(self):
        pass
    
    vocab_size = 0
    unigram_counts = {}
    unigram_probs = {}

    def train(self, train_corpus):
        
       
        # opens train_corpus and returns it as a list (sentences) of lists (tokens)
        list_of_sentences = general.Opener(train_corpus)

        # Tokenize the list of sentences (split sentences into a list of their tokens)
        tokenized_sentences = general.Tokenizer(list_of_sentences)

        # UNK the tokenized sentences
        unked_sentences = general.Unker(tokenized_sentences)
        
        # calculate counts with the UNKed set
        LanguageModel.unigram_counts = general.UniCounter(unked_sentences)
        
        LanguageModel.vocab_size = len(LanguageModel.unigram_counts)                       
        total_tokens = len([i for x in tokenized_sentences for i in x]) # this is our N for unigrams
        
        for key in LanguageModel.unigram_counts:
            
            probability = math.log(((LanguageModel.unigram_counts[key] + 1) / (total_tokens + LanguageModel.vocab_size)), 2)
            probability = round(probability, 3)
            LanguageModel.unigram_probs.update({key: probability})
        
        # Make a sorted dictionary 
        # Sorted first by probability (descending) then alphabetically for keys with the same value
        unigram_probs_sorted = dict(sorted(LanguageModel.unigram_probs.items(), key = lambda x: (-x[1], x[0])))
        
        return unigram_probs_sorted



    def score(self, test_corpus):    
        
        # open test_corpus and save as a list of sentences 
        test_sentences = general.Opener(test_corpus)
        
        
        #tokenize the sentences
        tokenized_test = general.Tokenizer(test_sentences)
        
        #UNK everything in the test set that doesn't appear in our vocabulary
        unked_test_set = general.Unker(tokenized_test, LanguageModel.unigram_counts)

        prob_cum_sum = 0   # cumulative probability
        word_count = 0     # N for calculating perplexity
        list_of_probs = [] # probabilites for each sentence 
        
        for i in range(0, len(unked_test_set)):
            
            #keeps track of probability for each sentence
            sen_prob = 0 
            
            for j in range(0, len(unked_test_set[i])):
                
                word_count += 1
                sen_prob += LanguageModel.unigram_probs[unked_test_set[i][j]]
                prob_cum_sum += LanguageModel.unigram_probs[unked_test_set[i][j]]
            
            list_of_probs.append(round(sen_prob,3))
        
        # make a list of sentences and their corresponding probabilities to return 
        sentences_and_probs = list(zip(test_sentences, list_of_probs))

        #calculate perplexity and add it as the list entry to the list we return
        h = (-1 / word_count) * (prob_cum_sum)
        perplexity = round(math.pow(2,h), 3)
        
        #add perplexity as the last index of the list we return
        sentences_and_probs.append(("perplexity: ", perplexity))
        
        return sentences_and_probs
