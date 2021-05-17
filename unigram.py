# !/usr/bin/python3
import re
import math
import globalfunctions as gf

# TODO: Implement a Laplace-smoothed unigram model :)
class LanguageModel:

    def __init__(self):
        pass

    def train(self, train_corpus):
        
        #get our global functions
        funcs = gf.GlobalFunctions()
       
        # opens train_corpus and returns it as a list (sentences) of lists (tokens)
        # Unks the list
        list_of_sentences = funcs.TrainUnker(funcs.TrainOpener(train_corpus))


        
        # calculate counts with the UNKed set
        unigram_counts = funcs.UniCounter(list_of_sentences)
        
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
        #these two lists will have corresponding indices
        list_of_test_sen = []    #splits every sentence into tokens to to iterate over
        sentences_and_probs = [] #keeps track of each sentence and it's probability
        
        # open the test_corpus and turn sentences into a list of lists
        # also add each sentence to a list of test senteces for printing 
        with open(test_corpus, 'r', encoding = 'utf8') as f:
            for line in f:
                line = re.sub(r'\n',"" , line)
                sentences_and_probs.append([line])
                line = line.split()
                list_of_test_sen.append(line)
        
        #UNK everything in the test set that doesn't appear in our vocabulary (isn't a key in unigram_counts)
        for i in range(0, len(list_of_test_sen)):
            for j in range(0, len(list_of_test_sen[i])):
                if list_of_test_sen[i][j] not in unigram_probs:
                    list_of_test_sen[i][j] = "<UNK>"

       
        #calculate the probability
        #all of our probabilities are log probabilities, so we can just add everything       
        
        # adds probabilites for each individual sentence,
        # and keeps a running probability for calculating perplexity
        
        prob_cum_sum = 0  # this is the sum of probabilities for every sentence
        word_count = 0    # this is our N for calculating perplexity
        
        for i in range(0, len(list_of_test_sen)):
            #keeps track of probability for each sentence
            sen_prob = 0 
            
            for j in range(0, len(list_of_test_sen[i])):
                word_count += 1
                sen_prob += unigram_probs[list_of_test_sen[i][j]]
                prob_cum_sum += unigram_probs[list_of_test_sen[i][j]]
            
            sentences_and_probs[i].append(round(sen_prob,3))
        
        #calculate perplexity and add it as the list entry to the list we return
        h = (-1 / word_count) * (prob_cum_sum)
        perplexity = round(math.pow(2,h), 3)
        
        #add perplexity as the last index of the list we return
        sentences_and_probs.append(["perplexity: ", perplexity])
        
        return sentences_and_probs
