# !/usr/bin/python3
import math
import general

# TODO: Implement a Laplace-smoothed trigram model :)
class LanguageModel:

    def __init__(self):
        pass
    
    vocab_size = 0
    unigram_counts = {}
    bigram_counts = {}
    trigram_counts = {}
    trigram_probs = {}

    def train(self, train_corpus):
        
        # Opens train_corpus and returns it as a list of sentences
        list_of_sentences = general.Opener(train_corpus)

        # Tokenize the list of sentences (split sentences into a list of their tokens)
        tokenized_sentences = general.Tokenizer(list_of_sentences)

        # UNK the tokenized sentences
        unked_sentences = general.Unker(tokenized_sentences)
        
        # Add <s> and </s> tags 
        for i in range(len(unked_sentences)):            
            unked_sentences[i].insert(0,'<s>')
            unked_sentences[i].insert(0,'<s>')
            unked_sentences[i].append('</s>')
        
        # Get Unigram counts that will be needed for calculating bigram probability
        LanguageModel.unigram_counts = general.UniCounter(unked_sentences)

        #get a nested dictionary of bigram counts
        LanguageModel.bigram_counts = general.BiCounter(unked_sentences)
        
        LanguageModel.trigram_counts = general.TriCounter(unked_sentences)
        
        
        # exclude <s> from vocab size
        vocab_size = len(LanguageModel.unigram_counts) - 1

        for k in LanguageModel.trigram_counts:

            for nk in LanguageModel.trigram_counts[k]:
                
                probability = math.log( ((LanguageModel.trigram_counts[k][nk] + 1) / (LanguageModel.bigram_counts[nk[1]][nk[0]] + vocab_size)), 2)
                probability = round(probability, 3)
                LanguageModel.trigram_probs.update({"{} {} {}".format(nk[0], nk[1], k): probability})
        
        trigram_probs_sorted = dict(sorted(LanguageModel.trigram_probs.items(), key = lambda x: (-x[1], x[0])))

        return trigram_probs_sorted  
        

    def score(self, test_corpus):
        print('I am an unimplemented TRIGRAM score() method.')  # delete this!
