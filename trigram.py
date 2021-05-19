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
        
        # Opens train_corpus and save it as a list of lists
        train_sentences = general.Tokenizer(general.Opener(train_corpus))

        # UNK the tokenized sentences
        general.Unker(train_sentences)
        
        # Add <s> and </s> tags 
        for i in range(len(train_sentences)):            
            train_sentences[i].insert(0,'<s>')
            train_sentences[i].insert(0,'<s>')
            train_sentences[i].append('</s>')
        
        LanguageModel.unigram_counts = general.UniCounter(train_sentences)

        LanguageModel.bigram_counts = general.BiCounter(train_sentences)
        
        LanguageModel.trigram_counts = general.TriCounter(train_sentences)
            
        # exclude <s> from vocab size
        LanguageModel.vocab_size = len(LanguageModel.unigram_counts) - 1

        for k in LanguageModel.trigram_counts:

            for nk in LanguageModel.trigram_counts[k]:
                
                probability = math.log( ((LanguageModel.trigram_counts[k][nk] + 1) / (LanguageModel.bigram_counts[nk[1]][nk[0]] + LanguageModel.vocab_size)), 2)
                probability = round(probability, 3)
                LanguageModel.trigram_probs.update({"{} {} {}".format(nk[0], nk[1], k): probability})
        
        #sort the probabilites
        trigram_probs_sorted = dict(sorted(LanguageModel.trigram_probs.items(), key = lambda x: (-x[1], x[0])))

        #output the sorted probabilites
        for key in trigram_probs_sorted:
            print("{} {}".format(key, trigram_probs_sorted[key]))
          
        

    def score(self, test_corpus):
        print('I am an unimplemented TRIGRAM score() method.')  # delete this!
