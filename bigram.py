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
        for i in range(len(tokenized_sentences)):
            
            tokenized_sentences[i].insert(0,'<s>')
            tokenized_sentences[i].append('</s>')  

        # UNK the tokenized sentences
        unked_sentences = general.Unker(tokenized_sentences)
        
        # Get Unigram counts that will be needed for calculating bigram probability
        unigram_counts = general.UniCounter(unked_sentences)

        #get a nested dictionary of bigram counts
        bigram_counts = general.BiCounter(unked_sentences)
        
        # exclude <s> from vocab size
        vocab_size = len(unigram_counts) - 1
        
        bigram_probs = {}
        for k in bigram_counts:

            for nk in bigram_counts[k]:
                
                probability = math.log( ((bigram_counts[k][nk] + 1) / (unigram_counts[nk] + vocab_size)), 2)
                probability = round(probability, 3)
                bigram_probs.update({"{} {}".format(nk,k): probability})
        
        bigram_probs_sorted = dict(sorted(bigram_probs.items(), key = lambda x: (-x[1], x[0])))

        train_data = [bigram_probs_sorted, unigram_counts]

        return train_data  



    def score(self, test_corpus, train_data):
        
        bigram_probs = train_data[0]
        unigram_counts = train_data[1]
        vocab_size = len(unigram_counts) - 1 #exclude <s>

        # open test_corpus and save as a list of sentences 
        test_sentences = general.Opener(test_corpus) 
        # tokenize the sentences
        tokenized_test = general.Tokenizer(test_sentences)
        # Add <s> and </s> tags 
        for i in range(0, len(tokenized_test)):
            
            tokenized_test[i].insert(0,'<s>')
            tokenized_test[i].append('</s>')  

        # UNK everything in the test set that doesn't appear in our vocabulary
        unked_test_set = general.Unker(tokenized_test, unigram_counts)

        # calculate the probability

        prob_cum_sum = 0   # this is the sum of probabilities for every sentence
        word_count = 0     # this is our N for calculating perplexity
        list_of_probs = [] #keep track of -our probabilites 
        
        for i in range(0, len(unked_test_set)):

            sen_prob = 0

            for j in range(1, len(unked_test_set[i])):
                
                word_count += 1

                if "{} {}".format(unked_test_set[i][j-1],unked_test_set[i][j]) in bigram_probs:
                    
                    sen_prob += bigram_probs["{} {}".format(unked_test_set[i][j-1],unked_test_set[i][j])]
                    prob_cum_sum += bigram_probs["{} {}".format(unked_test_set[i][j-1],unked_test_set[i][j])]

                else:
                    
                    sen_prob += math.log( ((1) / (unigram_counts[unked_test_set[i][j-1]] + vocab_size)), 2)
                    prob_cum_sum += math.log( ((1) / (unigram_counts[unked_test_set[i][j-1]] + vocab_size)), 2)
                
            list_of_probs.append(round(sen_prob,3))    


        # make a list of sentences and their corresponding probabilities to return 
        sentences_and_probs = list(zip(test_sentences, list_of_probs))

        #calculate perplexity and add it as the list entry to the list we return
        h = (-1 / word_count) * (prob_cum_sum)
        perplexity = round(math.pow(2,h), 3)
        
        #add perplexity as the last index of the list we return
        sentences_and_probs.append(("perplexity: ", perplexity))
        
        return sentences_and_probs