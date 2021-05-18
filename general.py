import re
import copy
#open text and clean each line
#create a list for which each index holds one line 
def Opener(train_corpus):
        
    list_of_sentences = []
        
    with open(train_corpus, 'r', encoding = 'utf8') as f:
            
        for line in f:
            
            line = re.sub(r'\n',"" , line) 
            list_of_sentences.append(line) 
        
    return list_of_sentences


def Tokenizer(list_of_sentences):
    
    list_copy = list_of_sentences.copy() 

    #tokenize the list_of_sentences
    for i in range(0, len(list_copy)):
        
        list_copy[i] = list_copy[i].split()
        
    return list_copy   


# unks all tokens that appear only once in a list(sentences) of lists (tokens)
# if passed a dictionary: will UNK all words in list_of_sentences that do not appear as keys in the dictionary
def Unker(list_of_sentences, train_prob_dict = False):
    
    tokenized_sentences = copy.deepcopy(list_of_sentences)       

    #if passed a dictionary 
    if train_prob_dict:
        
        # UNK every word that doesn't appear as a dictionary key (in our vocabulary) 
        for i in range(0, len(tokenized_sentences)):
            
            for j in range(0, len(tokenized_sentences[i])):
                
                if tokenized_sentences[i][j] not in train_prob_dict:
                    tokenized_sentences[i][j] = "<UNK>"


    else:
        pre_unk_counts = {}

        # go through every word in list and make a dictionary 
        # with words as keys, and their counts as values
        for i in range(0, len(tokenized_sentences)):
            
            for j in range(0, len(tokenized_sentences[i])):    
                
                if tokenized_sentences[i][j] in pre_unk_counts:
                    pre_unk_counts[tokenized_sentences[i][j]] += 1
                
                else:
                    pre_unk_counts.update({tokenized_sentences[i][j] : 1})
        
        #Go through and replace every word that appears only once with the <UNK> token
        for i in range(0,len(tokenized_sentences)):
            
            for j in range(0, len(tokenized_sentences[i])):
                
                if pre_unk_counts[tokenized_sentences[i][j]] == 1:
                    tokenized_sentences[i][j] = '<UNK>'
                else:
                    pass

    return tokenized_sentences
    


# given our list of tokenized sentences returns a dictionary with counts of unigrams 
def UniCounter(list_of_sentences):

    unigram_counts = {}

    for i in range(0, len(list_of_sentences)):
        
        for j in range(0, len(list_of_sentences[i])):    
                
            if list_of_sentences[i][j] in unigram_counts:
                
                unigram_counts[list_of_sentences[i][j]] += 1
            
            else:

                unigram_counts.update({list_of_sentences[i][j]: 1})

    return unigram_counts


#TODO write a function that takes a list of sentences and returns the bigram counts of that list in a nested dictionary 

def BiCounter(list_of_sentences):
    
    bigram_counts = {}

    for i in range(0, len(list_of_sentences)):

        for j in range(1, len(list_of_sentences[i])):
            
            if list_of_sentences[i][j] not in bigram_counts:
                
                bigram_counts.update( { list_of_sentences[i][j] : {list_of_sentences[i][j-1]: 1} } )

            else:
                
                if list_of_sentences[i][j-1] not in bigram_counts[list_of_sentences[i][j]]:

                    bigram_counts[list_of_sentences[i][j]].update({list_of_sentences[i][j-1]: 1 })

                    
                
                else:

                    bigram_counts[list_of_sentences[i][j]][list_of_sentences[i][j-1]] += 1

    return bigram_counts
