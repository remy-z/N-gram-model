# !/usr/bin/python3
import re
import math
# TODO: Implement a Laplace-smoothed unigram model :)
class LanguageModel:

    def __init__(self):
        pass

    def train(self, train_corpus):
        list_of_sentences = []
        
        #I'm gonna have to redo this entire thing to do UNKing properly

        #open text and clean each line
        #create a list for which each index holds one line as a list of tokens in that line
        with open(train_corpus, 'r', encoding = 'utf8') as f:
            for line in f:
                line = re.sub(r'\n',"" , line)
                line = line.split()
                list_of_sentences.append(line) 
        
        pre_unk_counts = {}
        total_tokens = 0
        #go through every word in clean text and make a dictionary
        for i in range(0, len(list_of_sentences)):
            for j in range(0, len(list_of_sentences[i])):    
                if list_of_sentences[i][j] in pre_unk_counts:
                    pre_unk_counts[list_of_sentences[i][j]][0] += 1
                else:
                    pre_unk_counts.update({list_of_sentences[i][j]:[1]})
        
        #UNK TIME EVERYBODY ITS UNK TIME 
        for i in range(0,len(list_of_sentences)):
            for j in range(0, len(list_of_sentences[i])):
                if pre_unk_counts[list_of_sentences[i][j]][0] == 1:
                    list_of_sentences[i][j] = '<UNK>'
                else:
                    pass

        unigram_counts = {}
        for i in range(0, len(list_of_sentences)):
            for j in range(0, len(list_of_sentences[i])):    
                total_tokens += 1
                if list_of_sentences[i][j] in unigram_counts:
                    unigram_counts[list_of_sentences[i][j]][0] += 1
                else:
                    unigram_counts.update({list_of_sentences[i][j]:[1]})
        
        vocab_size = len(unigram_counts)            
        
        #add log probabilities to the dictionary with laplace smoothing
        for key in unigram_counts:
            probability = math.log(((unigram_counts[key][0] + 1) / (total_tokens + vocab_size)), 2)
            unigram_counts[key].append(probability)
        
       

        print(unigram_counts) #this is here to check what I'm doing
        print()
        print('I am an in progress UNIGRAM train() method.')  # delete this!

    def score(self, test_corpus):
        print('I am an unimplemented UNIGRAM score() method.')  # delete this!
