# !/usr/bin/python3
import math
import general
import random 

class LanguageModel:

    def __init__(self):
        pass
    
    unigram_counts = {}
    vocab_size = 0
    bigram_counts = {}
    bigram_probs = {}

    def train(self, train_corpus):
        
        # Opens train_corpus and save it as a list of lists
        train_sentences = general.Tokenizer(general.Opener(train_corpus))

        # UNK the tokenized sentences
        general.Unker(train_sentences)

        # Add <s> and </s> tags 
        for i in range(len(train_sentences)):            
            train_sentences[i].insert(0,'<s>')
            train_sentences[i].append('</s>')  
        
        # Get Unigram counts that will be needed for calculating bigram probability
        LanguageModel.unigram_counts = general.UniCounter(train_sentences)

        #get a nested dictionary of bigram counts
        LanguageModel.bigram_counts = general.BiCounter(train_sentences)
        
        # exclude <s> from vocab size
        LanguageModel.vocab_size = len(LanguageModel.unigram_counts) - 1       

        for k in LanguageModel.bigram_counts:

            for nk in LanguageModel.bigram_counts[k]:
                
                probability = math.log( ((LanguageModel.bigram_counts[k][nk] + 1) / (LanguageModel.unigram_counts[nk] + LanguageModel.vocab_size)), 2)
                #probability = round(probability, 3)
                LanguageModel.bigram_probs.update({"{} {}".format(nk,k): probability})
        
        bigram_probs_sorted = dict(sorted(LanguageModel.bigram_probs.items(), key = lambda x: (-x[1], x[0])))

        #for key in bigram_probs_sorted:
        #    print("{} {}".format(key, round(bigram_probs_sorted[key], 3)))
          



    def score(self, test_corpus):
        
        # open test_corpus and save as a list of sentences 
        test_sentence_strings = general.Opener(test_corpus) 
        # turn sentences into list of lists
        test_sentences = general.Tokenizer(test_sentence_strings)
        # UNK everything in the test set that doesn't appear in our vocabulary
        general.Unker(test_sentences, LanguageModel.unigram_counts)
        
        # Add <s> and </s> tags 
        for i in range(0, len(test_sentences)):
            
            test_sentences[i].insert(0,'<s>')
            test_sentences[i].append('</s>')  

        prob_cum_sum = 0   # this is the sum of probabilities for every sentence
        word_count = 0     # this is our N for calculating perplexity
        list_of_probs = [] #keep track of -our probabilites 
        seen = 0 
        unseen = 0 
        for i in range(0, len(test_sentences)):

            sen_prob = 0
            # word_count += 1 #include <s> in our word count(N)?
            for j in range(1, len(test_sentences[i])):
                word_count += 1
                if "{} {}".format(test_sentences[i][j-1],test_sentences[i][j]) in LanguageModel.bigram_probs:  
                    sen_prob += LanguageModel.bigram_probs["{} {}".format(test_sentences[i][j-1],test_sentences[i][j])]
                    prob_cum_sum += LanguageModel.bigram_probs["{} {}".format(test_sentences[i][j-1],test_sentences[i][j])]
                    seen += 1
                else:
                    sen_prob += math.log( ((1) / (LanguageModel.unigram_counts[test_sentences[i][j-1]] + LanguageModel.vocab_size)), 2)
                    prob_cum_sum += math.log( ((1) / (LanguageModel.unigram_counts[test_sentences[i][j-1]] + LanguageModel.vocab_size)), 2)
                    unseen += 1

            list_of_probs.append(sen_prob)    


        # make a list of sentences and their corresponding probabilities to return 
        sentences_and_probs = list(zip(test_sentence_strings, list_of_probs))

        #calculate perplexity and add it as the list entry to the list we return
        h = (-1 / word_count) * (prob_cum_sum)
        perplexity = round(math.pow(2,h), 3)
        
        #add perplexity as the last index of the list we return
        sentences_and_probs.append(("perplexity: ", perplexity))
        
        #for i in range(len(sentences_and_probs)):
        #    print("{}  {}".format(sentences_and_probs[i][0],sentences_and_probs[i][1]))
        #print("unseen: " + str(unseen))
        #print("seen: " + str(seen))
    
    def shannon(self):
        shannon_list = []
        
        for key in LanguageModel.bigram_probs:
            bigram_prob = (key.split(), math.pow(2,LanguageModel.bigram_probs[key]))
            if "<UNK>" not in bigram_prob[0]:
                shannon_list.append(bigram_prob)


        print("Shannon Visualization using bigram: ")
        for i in range(10):
            end_sentence = False
            viz = ""
            last_word = "<s>"
            
            while not end_sentence: 
                current_word = []
                current_prob = []
                
                for item in shannon_list:
                    if item[0][0] == last_word:
                        current_word.append((item[0][1]))
                        current_prob.append(item[1])
            
                choice = random.choices(current_word, current_prob, k=1)
                if choice[0] == "</s>":
                    end_sentence = True
                else:
                    last_word = choice[0]
                    viz += f"{choice[0]} "
                
            print(viz)