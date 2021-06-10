import math
import general
 
class LanguageModel:

    def __init__(self):
        pass
    
    vocab_size = 0
    bigram_counts = {}
    trigram_counts = {}
    trigram_probs = {}


    def train(self, train_corpus):
        LanguageModel.train_corpus = train_corpus
        # Opens train_corpus and save it as a list of lists
        train_sentences = general.Tokenizer(general.Opener(train_corpus))
        # UNK the tokenized sentences
        general.Unker(train_sentences)
        
        print("Calculating Laplace Smoothing Trigram Probabilites...")
        for i in range(len(train_sentences)):
            train_sentences[i].insert(0, '<s>')
            train_sentences[i].insert(0, '<s>')
            train_sentences[i].append('</s>')      

        LanguageModel.bigram_counts = general.BiCounter(train_sentences)
            
        LanguageModel.trigram_counts = general.TriCounter(train_sentences)

        # substract <s> tag
        LanguageModel.vocab_size = len(LanguageModel.trigram_counts) - 1
        
        #calculate laplace smoothed probabilites
        for k in LanguageModel.trigram_counts:
            for nk in LanguageModel.trigram_counts[k]:
                probability = ((LanguageModel.trigram_counts[k][nk] + 1) / (LanguageModel.bigram_counts[nk[1]][nk[0]] + LanguageModel.vocab_size))
                probability = math.log2(probability)
                LanguageModel.trigram_probs.update({"{} {} {}".format(nk[0], nk[1], k): probability})
        
        #sort the probabilites
        trigram_probs_sorted = dict(sorted(LanguageModel.trigram_probs.items(), key = lambda x: (-x[1], x[0])))
        output_this = ""
        for key in trigram_probs_sorted:
            output_this += f"{key} {round(trigram_probs_sorted[key], 3)} \n"
        print("Trigram probabilites: ")
        print(output_this)

    def score(self, test_corpus):
        # process test_corpus
        test_sentence_strings = general.Opener(test_corpus)
        test_sentences = general.Tokenizer(test_sentence_strings)
        general.Unker(test_sentences, LanguageModel.trigram_counts)

        # insert our sentence tags
        for i in range(0, len(test_sentences)):
            test_sentences[i].insert(0, '<s>')
            test_sentences[i].insert(0, '<s>')
            test_sentences[i].append('</s>')

        prob_cum_sum = 0
        word_count = 0
        list_of_probs = []
        tri = 0
        unseentri = 0 
        unseenbi = 0
        for i in range(0, len(test_sentences)):
            sen_prob = 0
            for j in range(2, len(test_sentences[i])):
                word_count += 1
                #if trigram probability use that
                if "{} {} {}".format(test_sentences[i][j-2], test_sentences[i][j-1], test_sentences[i][j]) in LanguageModel.trigram_probs:
                    sen_prob += LanguageModel.trigram_probs["{} {} {}".format(test_sentences[i][j-2], test_sentences[i][j-1], test_sentences[i][j])]
                    prob_cum_sum += LanguageModel.trigram_probs["{} {} {}".format(test_sentences[i][j-2], test_sentences[i][j-1], test_sentences[i][j])]
                    tri +=1 
                # if unseen trigram, calculate a laplace probability with bigram count 
                elif test_sentences[i][j-2] in LanguageModel.bigram_counts[test_sentences[i][j-1]]:
                    sen_prob += math.log2(1 / (LanguageModel.bigram_counts[test_sentences[i][j-1]][test_sentences[i][j-2]] + LanguageModel.vocab_size))
                    prob_cum_sum += math.log2(1 / (LanguageModel.bigram_counts[test_sentences[i][j-1]][test_sentences[i][j-2]] + LanguageModel.vocab_size))
                    unseentri+=1
                # if n-2, n-1 is an unseen bigram 
                else:
                    sen_prob += math.log2(1/ LanguageModel.vocab_size)
                    prob_cum_sum += math.log2(1/ LanguageModel.vocab_size)
                    unseenbi += 1
            list_of_probs.append(sen_prob)

        sentences_and_probs = list(zip(test_sentence_strings, list_of_probs))

        h = (-1 / word_count) * (prob_cum_sum)
        perplexity = round(math.pow(2, h), 3)
        
        print("Test sentence probabilites:")
        output_this = ""
        for tuple in sentences_and_probs:
            output_this += f"{tuple[0]} {round(tuple[1], 3)} \n"
        print(output_this)
        print()
        print("Trigram Perplexity, Laplace Smoothing: " + str(perplexity))
        print(f'tri: {tri}\nunseentri: {unseentri}\nunseenbi: {unseenbi}')