# !/usr/bin/python3
import math
import general
import random

class LanguageModel:

    def __init__(self):
        pass
    
    vocab_size = 0
    unigram_counts = {}
    unigram_probs = {}

    def train(self, train_corpus):
        print("Calculating Unigram Probabilites...")   
        # Opens train_corpus and save it as a list of lists
        train_sentences = general.Tokenizer(general.Opener(train_corpus))
        
        # UNK the tokenized sentences and add end sentence tag
        general.Unker(train_sentences)
        for i in range(len(train_sentences)):            
            train_sentences[i].append('</s>')  

        # calculate counts with the UNKed set
        LanguageModel.unigram_counts = general.UniCounter(train_sentences)
        LanguageModel.vocab_size = len(LanguageModel.unigram_counts)  
                            
        total_tokens = len([i for x in train_sentences for i in x]) # this is our N for unigram probability calculation
        
        for key in LanguageModel.unigram_counts:   
            probability = math.log(((LanguageModel.unigram_counts[key] + 1) / (total_tokens + LanguageModel.vocab_size)), 2)
            LanguageModel.unigram_probs.update({key: probability})
        
        unigram_probs_sorted = dict(sorted(LanguageModel.unigram_probs.items(), key = lambda x: (-x[1], x[0])))
        
        # output our results
        output_this = ""
        for key in unigram_probs_sorted:
            output_this += f"{key} {round(unigram_probs_sorted[key], 3)} \n"
        print("Unigram probabilites:")
        print(output_this)



    def score(self, test_corpus):    
        # Open test_corpus and save as a list of sentences 
        test_sentence_strings = general.Opener(test_corpus) 
        # Turn sentences into list of lists
        test_sentences = general.Tokenizer(test_sentence_strings)
        for i in range(len(test_sentences)):            
            test_sentences[i].append('</s>')
        # UNK everything in the test set that doesn't appear in our vocabulary
        general.Unker(test_sentences, LanguageModel.unigram_counts)

        prob_cum_sum = 0   # cumulative probability
        word_count = 0     # N for calculating perplexity
        list_of_probs = [] # probabilites for each sentence 
        
        for i in range(0, len(test_sentences)):
            
            #keep track of probability for each sentence
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
        
        
        output_this = ""
        for tuple in sentences_and_probs:
            output_this += f"{tuple[0]} {round(tuple[1], 3)} \n"
        print("Test sentence probabilites:")
        print(output_this)
        print()
        print("Unigram Perplexity, Laplace Smoothing: " + str(perplexity))

    def shannon(self, how_many):
        
        shannon_probs = LanguageModel.unigram_probs
        # With our probabilites, generate random sentences using the Shannon Visualization method
        print("Shannon Visualization using unigram probabilites: ")
        print("")
        for i in range(how_many):
            end_sentence = False
            viz = ""
            while not end_sentence: 
                current_word = []
                current_prob = []
                items = shannon_probs.items()
                for item in items:
                    current_word.append(item[0]), current_prob.append(math.pow(2, item[1]))
            
                choice = random.choices(current_word, current_prob, k=1)

                if choice[0] == "</s>":
                    end_sentence = True
                else:
                    viz += f"{choice[0]} "   
            
            print(f'{i +1}) {viz}')