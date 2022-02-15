# N-gram 
N-gram model project 
By Remy Zachwieja & Mark Min

This is an N-gram language model, after being training the model by determining probabilities of "ngrams" 
it predicts the possibility of any given text occurring. An ngram is simply a sequence of n words, a 2-gram is called a bigram,
a 3-gram is a trigram, and an ngram of just one words is a unigram. Ngram models calculate probability on the assumption that we can 
predict the probability of a word by looking at what precedes it. (e.g we're more likely to see the Trigram "he eats bread" than 
"he eats highways") By counting all the ngrams in a corpus we can then calculate the probabilities of a word occurring given it's context.
For instance, a bigram probability would be calculated as P(Wn|Wn−1) = C(Wn−1 Wn) / C(Wn−1), the probability of Word n occurring given context 
Word n-1 (the previous word) is equal to the count of the (Word n-1 Word n) bigram over the total occurrence of Word n-1. Using the 
chain rule of probability we can compute probabilities of entire sequences like P(W1, W2,..., Wn), and in turn the probability of an entire text.

We can evaluate our model's effectiveness using a measure called Perplexity. Perplexity is the inverse probability of our test set, 
normalized by the number of words in the set. Since the goal is having a model that predicts a test set as well as possible, a higher probability 
is better, and in turn a lower Perplexity means a better model. 

To avoid issues that occur when a word we have not seen in our training set appears in the test set we implemented some smoothing methods. 
First, Laplace smoothing accounts for out of vocabulary words by adding one to the count of every ngram. If we encounter an unseen ngram, we can 
just pretend that we've seen it one time, so that we don't assign any ngrams a probability of zero. We also make use of the UNK method. When training 
our ngram model, any word that occurs only one time is replaced by an <UNK> tag, then probabilities are calculated on the UNKed train set. When evaluating the model
we preprocess the test set by replacing all out of vocabulary words (any word that didn't appear in the train set) with the <UNK> tag. 

To avoid issues with underflow, all probabilities are calculated as log probabilities. 

Finally, we've implemented a Shannon Visualization Model that allows us to visualize sentences that are randomly generated using our n gram models. 

The data we've used for this project is a compilation of sentences from the works of 18th century English novelist Jane Austen. 

This project is all in done all in python. 
When run the program will output:
	A list of all ngrams in the train set along side their probabilities 
	A list of every all sentences in the test set with their probabilities 
	The perplexity of the model when evaluated on the given test set
	Randomly generated sentences using Shannon Visualization 

To run the model, pass it the following arguments 

	filepath for train set, -t filepath for test set, -n order of n-gram, -s number of sentences to generate with Shannon Visualization

Try these commands!


	python main.py data/austen-train.txt -n 1 -t data/austen-dev.txt -s 10
	python main.py data/austen-train.txt -n 2 -t data/austen-test.txt -s 10
	python main.py data/austen-train.txt -n 3 -t data/austen-dev.txt -s 10

	


