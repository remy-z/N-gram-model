# !/usr/bin/python3
import math
import general

# TODO: Implement a Laplace-smoothed bigram model :)
class LanguageModel:

    def __init__(self):
        pass

    def train(self, train_corpus):
        
        # Opens train_corpus and returns it as a list (sentences) of lists (tokens)
        list_of_sentences = general.Opener(train_corpus)

        # Tokenize the list of sentences (split sentences into a list of their tokens)
        tokenized_sentences = general.Tokenizer(list_of_sentences)

        # Add <s> and </s> tags 
        for i in range(0, len(tokenized_sentences)):
            
            tokenized_sentences[i].insert(0,'<s>')
            tokenized_sentences[i].append('</s>')  

        # UNK the tokenized sentences
        unked_sentences = general.Unker(tokenized_sentences)
        
        # Get Unigram counts that will be needed for calculating bigram probability
        unigram_counts = general.UniCounter(unked_sentences)

        #get a nested dictionary of bigram counts
        bigram_counts = general.BiCounter(unked_sentences)
        
        print()
        # TODO calculate the probabilites for each bigram in bigram_counts
        # TODO put these probabilites in a dictionary with bigram as key, probability as value


    def score(self, test_corpus):
        print('I am an unimplemented BIGRAM score() method.')  # delete this!
