# !/usr/bin/python3
import re
import math
# TODO: Implement a Laplace-smoothed unigram model :)
class LanguageModel:

    def __init__(self):
        pass

    def train(self, train_corpus):
        list_of_sentences = []

        #open text and clean each line
        #create a list for which each index holds one line as a list of tokens in that line
        with open(train_corpus, 'r', encoding = 'utf8') as f:
            for line in f:
                line = re.sub(r'\n',"" , line)
                line = line.split() #this turns the line string into a list of tokens 
                list_of_sentences.append(line) 
        
        pre_unk_counts = {}
        total_tokens = 0
        # go through every word in clean text and make a dictionary 
        # with words as keys, and their counts as values
        for i in range(0, len(list_of_sentences)):
            for j in range(0, len(list_of_sentences[i])):    
                if list_of_sentences[i][j] in pre_unk_counts:
                    pre_unk_counts[list_of_sentences[i][j]] += 1
                else:
                    pre_unk_counts.update({list_of_sentences[i][j] : 1})
        
        #UNK TIME EVERYBODY ITS UNK TIME 
        
        
        #Go through and replace every word that appears only once with the <UNK> token
        for i in range(0,len(list_of_sentences)):
            for j in range(0, len(list_of_sentences[i])):
                if pre_unk_counts[list_of_sentences[i][j]] == 1:
                    list_of_sentences[i][j] = '<UNK>'
                else:
                    pass

        """
        TODO PUT EVERYTHING ABOVE THIS IN A SEPERATE CLASS AS A GLOBAL FUNCTION
        """

        #recalculate counts with the UNKed set
        unigram_counts = {}
        for i in range(0, len(list_of_sentences)):
            for j in range(0, len(list_of_sentences[i])):    
                total_tokens += 1
                if list_of_sentences[i][j] in unigram_counts:
                    unigram_counts[list_of_sentences[i][j]][0] += 1
                else:
                    unigram_counts.update({list_of_sentences[i][j]:[1]})
        
        
        #every key in the unigram_counts dictionary is one unique word, so our vocab size 
        #is equal to len(unigram_counts)-1 because we don't want to inlcude UNK
        vocab_size = len(unigram_counts) - 1            
        
        #add log probabilities to the dictionary with laplace smoothing
        unigram_probs = {}
        for key in unigram_counts:
            probability = math.log(((unigram_counts[key][0] + 1) / (total_tokens + vocab_size)), 2)
            probability = round(probability, 3)
            unigram_probs.update({key: probability})

        return unigram_probs

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
