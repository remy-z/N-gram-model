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
        
        # subtract both UNK and <s> from vocab size
        vocab_size = len(unigram_counts) - 2
        bigram_probs = {}

        
        for k in bigram_counts:

            for nk in bigram_counts[k]:
                
                probability = math.log( ((bigram_counts[k][nk] + 1) / (unigram_counts[nk] + vocab_size)), 2)
                probability = round(probability, 3)
                bigram_probs.update({"{} {}".format(nk,k): probability})
        
        bigram_probs_sorted = dict(sorted(bigram_probs.items(), key = lambda x: (-x[1], x[0])))

        return bigram_probs_sorted  



    def score(self, test_corpus, bigram_probs):
        sentences_and_probs = []
        return sentences_and_probs
