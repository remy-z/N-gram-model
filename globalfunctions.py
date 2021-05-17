import re

class GlobalFunctions:

    def __init__(self):
        pass

    
    #open text and clean each line
    #create a list for which each index holds one line as a list of tokens in that line
    
    def TrainOpener(self, train_corpus):
        list_of_sentences = []
        
        with open(train_corpus, 'r', encoding = 'utf8') as f:
            for line in f:
                line = re.sub(r'\n',"" , line)
                line = line.split() #this turns the line string into a list of tokens 
                list_of_sentences.append(line) 
        
        return list_of_sentences

    # unks a list(sentences) of lists (tokens)
    def TrainUnker(self, list_of_sentences):
        pre_unk_counts = {}
        # go through every word in list and make a dictionary 
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

        return list_of_sentences
    
    # given our list of sentences returns a dictionary with counts of unigrams 
    def UniCounter(self, list_of_sentences):

        unigram_counts = {}

        for i in range(0, len(list_of_sentences)):
            for j in range(0, len(list_of_sentences[i])):    
                
                if list_of_sentences[i][j] in unigram_counts:
                    unigram_counts[list_of_sentences[i][j]][0] += 1
                else:
                    unigram_counts.update({list_of_sentences[i][j]:[1]})

        return unigram_counts

