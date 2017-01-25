---
geometry: margin=1in
fontfamily: utopia
fontsize: 11pt
---

<!-- fontfamily: font package to use for LaTeX documents (with pdflatex): TeXLive has bookman (Bookman), utopia or fourier (Utopia), fouriernc (New Century Schoolbook), times or txfonts (Times), mathpazo or pxfonts or mathpple (Palatino), libertine (Linux Libertine), arev (Arev Sans), and the default lmodern, among others. -->

# Word Prediction using Recurrent Neural Networks

Brian Burns  
Udacity Machine Learning Engineer Nanodegree  
January 1, 2016  

<!-- from https://review.udacity.com/#!/rubrics/108/view -->

## Definition

### Project Overview

<!-- Student provides a high-level overview of the project in layman's terms.
Background information such as the problem domain, the project origin, and
related data sets or input data is given. -->

Word prediction is the task of predicting the most likely words following the
preceding text - in the literature this is known as Language modeling. It has
many applications, such as suggesting the next word as text is entered, as an
aid in resolving ambiguity in speech and handwriting recognition, and machine
translation.

The generation of a likely word given prior words goes back to Claude Shannon's
work on information theory (Shannon 1948) based in part on Markov models
introduced by Andrei Markov (Markov 1913) - counts of encountered word sequences
are used to estimate the conditional probability of seeing a word given the
prior words. These so called n-grams formed the basis of commercial word
prediction software in the 1980's, eventually supplemented with similar syntax
and part of speech predictions (Carlberger 1997).

More recently, distributed representations of words have been used in recurrent
neural networks (RNNs), which can better handle data sparsity and allow more of
the context to affect the prediction (Bengio 2003).

The problem is a supervised learning task, and any text can be used to train and
evaluate the models - we'll be using a million words from books digitized by the
Gutenberg Project (Gutenberg 2016) for evaluation. Others use larger corpora,
e.g. Google's billion word corpus (Chelba 2013). Depending on the problem
domain, different corpora might be more appropriate, e.g. training on a
chat/texting corpus would be more applicable for a phone text entry application.


<!-- "Language modeling---or more specifically, history-based language modeling (as -->
<!-- opposed to full sentence models)---is the task of predicting the next word in a -->
<!-- text given the previous words." -->
<!-- "Language modeling is the art of determining the probability of a sequence of words." -->


"The main advantage of NNLMs over n-grams is that history is no longer seen as
exact sequence of n - 1 words H, but rather as a projection of H into some lower
dimensional space. This reduces number of parameters in the model that have to
be trained, resulting in automatic clustering of similar histories." mikolov 2012 thesis
The hidden layer of RNN represents all previous
history and not just n -1 previous words, thus the model can theoretically represent long
context patterns
however the error gradients quickly vanish as they get backpropagated in time
(in rare cases the errors can explode), so several steps of unfolding are
sufficient (this is sometimes referred to as truncated BPTT).
**While for word
based LMs, it seems to be sufficient to unfold network for about 5 time steps,
it is interesting to notice that this still allows the network to learn to store
information for more than 5 time steps.
Similarly, network that is trained by normal backpropagation can be seen as a
network trained with one unfolding step, and still as we will see later, even
this allows the network to learn longer context patterns, such as 4-gram
information.
A simple solution to the exploding gradient problem is to truncate values of the
gradients. In my experiments, I did limit maximum size of gradients of errors
that get accumulated in the hidden neurons to be in a range < -15; 15 >. This
greatly increases stability of the training, and otherwise it would not be
possible to train RNN LMs successfully on large data sets.
(Mikolov 2012)

### Problem Statement

<!-- The problem which needs to be solved is clearly defined.
A strategy for solving the problem,
including discussion of the expected solution, has been made. -->

Problem: Given a sequence of *n* words, predict the *k* most likely next words
and their probabilities.

For example, for the sequence "The dog", a solution might be
(barked 10%, slept 9%, ran 8%).

We'll use some different neural network architectures to find the most likely
next words - a standard Recurrent Neural Network (RNN), a Long Short-Term Memory
(LSTM) RNN, and a GRU (Gated Recurrent Unit) RNN - and compare these against
some baseline n-gram models. The GRU RNN is expected to offer the best
performance for a given amount of training time [cite!].


### Metrics

<!-- Metrics used to measure performance of a model or result are clearly
defined. Metrics are justified based on the characteristics of the problem. -->

The metrics used to evaluate the performance of the models are relevance,
accuracy, and perplexity.

The primary metric used to evaluate the models is **relevance**, which we'll
define as

> **Relevance** = # correct predictions / # total predictions

where a prediction will be considered *correct* if the actual word is in the
list of *k* most likely words. This is relevant to the task of presenting the
user with a list of most likely next words as they are entering text - we'll use
*k* = 3 for evaluation.

We'll also report the **accuracy**, which measures the number of predictions
where the most likely prediction is the correct one (which is *relevance* where
*k* = 1).

Results in the literature are often reported as **perplexity**, which gives an idea
of how well the model has narrowed down the possible choices for the next word -
e.g. a perplexity of 100 corresponds roughly to a uniform choice from 100 words.
We'll report this as well to see how our models compare with those in the literature. 


## Analysis

### Data Exploration

<!-- If a dataset is present, _features_ and _calculated statistics_ relevant to the
problem have been reported and discussed, along with a _sampling_ of the data.
_Abnormalities_ or characteristics about the data or input that need to
be addressed have been identified. -->

The training and testing data are obtained from ten books from Project
Gutenberg, totalling roughly one million words -

<!-- note: can make this fixed chars by indenting, but needs to be at left margin to make a latex table -->
<!-- this is output from print(util.table(data.analyze())) -->


-> subtract one from chars/word for space!

\small

| Text                                                         |   Words | Chars / Word   | Words / Sentence   | Unique Words   |   Grade Level |
|--------------------------------------------------------------+---------+----------------+--------------------+----------------+---------------|
| 1851 Nathaniel Hawthorne The House of the Seven Gables (G77) |   96217 | 6.2            | 21.2               | 22214          |            12 |
| 1862 Victor Hugo Les Miserables (G135)                       |  516244 | 6.2            | 14.6               | 82177          |            10 |
| 1865 Lewis Carroll Alice in Wonderland (G28885)              |   26758 | 5.6            | 16.4               | 6346           |             9 |
| 1883 Robert Louis Stevenson Treasure Island (G120)           |   62826 | 5.7            | 16.9               | 13894          |             8 |
| 1898 Henry James The Turn of the Screw (G209)                |   38663 | 5.9            | 15.4               | 9417           |             8 |
| 1899 Joseph Conrad Heart of Darkness (G219)                  |   34833 | 6.0            | 14.5               | 9871           |             9 |
| 1905 M R James Ghost Stories of an Antiquary (G8486)         |   42338 | 5.9            | 19.6               | 10882          |             9 |
| 1907 Arthur Machen The Hill of Dreams (G13969)               |   60528 | 6.0            | 25.7               | 14406          |            10 |
| 1908 Kenneth Graham The Wind in the Willows (G289)           |   54160 | 5.9            | 16.8               | 13102          |             9 |
| 1919 P G Woodhouse My Man Jeeves (G8164)                     |   46947 | 5.8            | 10.1               | 10917          |             8 |
| 1920 M R James A Thin Ghost and Others (G20387)              |   29311 | 5.7            | 21.3               | 7767           |             8 |

\normalsize


The grade level is calculated using the Coleman-Liau Index (Coleman 1975), which
is based on the average word and sentence lengths.

The Gutenberg text number is listed in parentheses, and the texts can be found
online - e.g. Alice in Wonderland can be found at
http://www.gutenberg.org/etext/28885.


Some sample text:

> "Speak English!" said the Eaglet. "I don't know the meaning of half those long words, and, what's more, I don't believe you do either!" - *Alice in Wonderland* (Shortest words)

> I went up and passed the time of day. "Well, well, well, what?" I said. "Why, Mr. Wooster! How do you do?" - *My Man Jeeves* (Shortest sentences)

> From the eminence of the lane, skirting the brow of a hill, he looked down into deep valleys and dingles, and beyond, across the trees, to remoter country, wild bare hills and dark wooded lands meeting the grey still sky. - *The Hill of Dreams* (Longest sentences)

> He was one of the martyrs to that terrible delusion, which should teach us, among its other morals, that the influential classes, and those who take upon themselves to be leaders of the people, are fully liable to all the passionate error that has ever characterized the maddest mob. - *The House of the Seven Gables* (Highest grade level)



### Exploratory Visualization

<!-- A visualization has been provided that summarizes or extracts a relevant
characteristic or feature about the dataset or input data with thorough
discussion. Visual cues are clearly defined. -->

<!-- The sentence length distributions for the different texts are as follows - note -->
<!-- that both of M. R. James' works have similar sentence length distributions. -->

<!-- ![Sentence length distributions](images/sentence_lengths_boxplot.png) -->

Neural networks are able to represent words in a vector space, e.g. as an array
of 300 floating point numbers - this allows similar words to be closer together
in vector space, and to trigger similar following words. Pre-trained word
embeddings, such as word2vec (Mikolov 2013) or GloVe (Pennington 2014), can be
used to save on training time.

For this project we'll use 50-dimensional word vectors from GloVe - this plot
shows some sample word embeddings projected to 2 dimensions using PCA - note how
the adjectives, verbs, and nouns/agents are all grouped together:

![Sample word embeddings](images/word_embeddings.png)



<!-- the keyword is *relevant* - what vis would be relevant for this problem? -->
<!-- and *thorough discussion* - needs to be something interesting.  -->


<!-- we're doing word prediction -->
<!-- maybe something more like information content? -->
<!-- ie how compressible the text is? -->
<!-- ie how predictable it is? -->
<!-- cf pure randomness (log2 26 ~ (log 26 2) ~ 4.7bits?) -->
<!-- how calculate? ngrams?  -->

<!-- -> information content of english - shannon paper -->
<!-- use to compare texts? -->
<!-- plot against mean/median sentence lengths? -->

<!-- what if compressed the text and compared percent reduction against something?  -->
<!-- but also depends on length of text.  -->


<!-- say alphabet is 26 characters, which is log_2 of 26 = (log 26 2) = 4.70 bits/character -->
<!-- say each word is 5 characters - if completely random, then this would be (* 5 4.70) 23.5 bits per word -->
<!-- this would be the  -->


### Algorithms and Techniques

<!-- Algorithms and techniques used in the project are thoroughly discussed and
properly justified based on the characteristics of the problem. -->

Until recently, n-grams were state of the art in word prediction [cite ] -
Recurrent Neural Networks were able to beat them in 2003, though at the cost of
greater training time (Bengio 2003).

A Recurrent Neural Network (RNN) is able to remember arbitrary amounts of
context, while n-grams are effectively limited to about 4 words of context (a
5-gram will give 4 words of context) - going beyond 5-grams requires increasing
amounts of resources in terms of training data and storage space, as the
resources required grow exponentially with the amount of context.

An RNN is able to make predictions based on words using arbitrarily long
context, because it can represent words more compactly with an internal
representation (embedding in a vector space), which also allows words to have
arbitrary degrees of similarities to other words. Hence, for instance, a cat can
be predicted to 'sleep', even if the model was only trained on a dog sleeping,
due to the similarity of the words 'dog' and 'cat'.

An RNN 'unfolds' to a neural network of arbitrary depth, depending on how far
back in a sequence it is trained. It keeps track of what it has seen through a
hidden state at each step in a sequence - this is combined with the current
token's representation (by addition) and the sum is then used to make
predictions about the next word.

![RNN (LeCun 2015)](images/rnn_nature.jpg)

-> is this correct about U? read something different. what about matrix E the
   embedding layer? it's separate from U...

The matrix **U** amounts to a table of word embeddings in a vector space of many
dimensions (which could be e.g. 50-300) - each word in the vocabulary
corresponds with a row in the table, and the dot product between any two words
gives their similarity, once the network is trained. Alternatively, pre-trained
word embeddings can be used to save on training time.

The matrix **W** acts as a filter on the internal hidden state, which represents
the prior context of arbitrary length.

The matrix **V** allows each word in the vocabulary to 'vote' on how likely it
thinks it will be next, based on the context (current + previous words). The
softmax layer then converts these scores into probabilities, so the top *k* most
likely words can be found for a given context.


A LSTM (Long Short-Term Memory) RNN (Hochreiter 1997) works similarly, but has a
'memory cell' at the center, which has 'switches' for storing memory, or
forgetting memory, and the state of the switches can be learned along with the
rest of the parameters. This turns out to be easier to train than plain RNN's,
which have problems with vanishing and exploding gradients, which make them slow
and difficult to train.

A GRU (Gated Recurrent Unit) RNN (Chung 2014) is similar to an LSTM, but has
fewer parameters, so is a bit easier to train - so this is what we will use for our
base RNN.

-> show calcs and matrices for abcd example - nvocab=5, nhidden=2, incl loss vs
   accuracy, perplexity

see http://www.wildml.com/2015/10/recurrent-neural-network-tutorial-part-4-implementing-a-grulstm-rnn-with-python-and-theano/
incls diagrams

-> then attention 2015? discuss briefly, cite


-> compare word-level with character-level rnn learning - maybe character level
   easier on the cpu, as nvocab=26 (or 52 with caps). plus invented words.



-> do big O analysis, estimate memory needed, # calcs, time


how to train RNN - use fwd and backprop?
"The difference is recurrence. The RNN cannot be easily trained as if you try to compute gradient - you will soon figure out that in order to get a gradient on n'th step - you need to actually "unroll" your network history for n-1 previous steps. This technique, known as BPTT (backpropagation through time) is exactly this - direct application of backpropagation to RNN. Unfortunately this is both computationaly expensive as well as mathematically challenging (due to vanishing/exploding gradients). People are creating workaround on many levels, by for example introduction of specific types of RNN which can be efficiently trained (LSTM, GRU), or by modification of training procedure (such as gradient clamping). To sum up - theoreticaly you can do "typical" backprop in the mathematical sense, from programming perspective - this requires more work as you need to "unroll" your network through history. This is computationaly expensive, and hard to optimize in the mathematical sense."



### Benchmark

<!-- Student clearly defines a benchmark result or threshold for comparing
performances of solutions obtained. -->

For the benchmark model a simple n-gram model will be used - this is a standard
approach for next word prediction based on Markov models. A multidimensional
array, indexed by vocabulary words, stores counts of occurrences of n-tuples of
words based on the training data. These are then normalized to get a probability
distribution, which can be used to predict the most likely words following a
sequence.

A trigram (3-gram) model will be used as the baseline, as it should work fairly
well with a million words of data - going to 4- or 5- grams would
require more training data.

-> find published results, history (when were these first developed, called n-grams, get citations)

"n-gram models (Jelinek and Mercer, 1980;Katz 1987). See (Manning and Schutze, 1999) for a review."




## Methodology

### Data Preprocessing

<!-- All preprocessing steps have been clearly documented. Abnormalities or
characteristics about the data or input that needed to be addressed have been
corrected. If no data preprocessing is necessary, it has been clearly justified.
-->

The text files were downloaded from the Gutenberg site - they each contain
header and footer information separate from the actual text, including a
Gutenberg license, but these are left in place, and read together into a single
Python UTF-8 string. The string is split into paragraphs, which are then
shuffled randomly - this will allow the RNN to learn more context than if the
text was split on sentences and scrambled.

The paragraphs are combined, and the text is converted to lowercase to increase
the amount of vocabulary that can be learned for a given amount of time and
memory. The text is then tokenized to the top NVOCAB words - the rarer words are
**not recorded** - punctuation marks are treated as separate tokens.

The sequence of tokens is then split into training, validation, and test sets -
the validation and test sets were set at 10,000 tokens, so the training set has
roughly 1,000,000 tokens.


### Implementation

<!-- The process for which metrics, algorithms, and techniques were implemented
with the given datasets or input data has been thoroughly documented.
Complications that occurred during the coding process are discussed. -->


The RNN was implemented with Python using Keras (Chollet 2015) with a TensorFlow
backend (Abadi 2015). Different architectures and parameters were explored, and
compared based on their loss and accuracy scores. The most promising model was
then selected for further analysis.

-> compare Keras code vs TensorFlow for a simple RNN?

Keras is a wrapper around TensorFlow that simplifies its use - e.g.


The basic architecture of the RNN is a word embedding input layer, then a GRU
RNN with one or two layers, a densely connected output layer, and a softmax
layer to convert the outputs to probabilities.

The RNN was initially implemented using one-hot encoded token sequences, but
this proved to be infeasible memory-wise - the training labels needed to store ~
ntraining_elements * nvocab integers, which would be ~ 1e6 * 1e4 = 1e10 integers,
or roughly 1e11 bytes = 100 GB, too large for a 16 GB system. 

Fortunately Keras/TensorFlow allows the use of 'sparse' sequences, so they can
be fed sequences of integers, and output the same.

<!-- x_train will be O(N*nelements) ~ 10 * 1mil * 8bytes = 80mb -->
<!-- y_train one-hot will be O(nelements*NVOCAB) ~ 1mil * 10k * 8bytes = 80gb ! even 1k vocab -> 8gb -->
<!-- so need to use generators -->
<!-- unless use sparse_categorical_crossentropy, then y_train would just be O(nelements) ~ 1mil ~ 8mb -->


Training

For the training step, the training sequence was fed to the network, which was
trained for a certain number of epochs.





Each epoch

Testing

For the testing step, the RNN predictors were fed word sequences from the test
data, and the top *k* predicted words were compared against the actual word, and
a *relevance* score tallied.

<!-- Training sets of increasing sizes were used - 1k, 10k, 100k, 1 million words, -->
<!-- and the results recorded for comparison. Timing and memory information were also -->
<!-- recorded for all processes for analysis. -->


"Input vector x(t) represents word in time t encoded using 1-of-N coding and
previous context layer - size of vector x is equal to size of vocabulary V (this
can be in practice 30000 -200000) plus size of context layer. Size of context
(hidden) layer s is usually 30 - 500 hidden units. Based on our experiments,
size of hidden layer should reflect amount of training data - for large amounts
of data, large hidden layer is needed
In our experiments, networks do not overtrain significantly, even if very large
hidden layers are used - regularization of networks to penalize large weights
did not provide any significant improvements.
The training algorithm described here is also referred to as truncated
backpropagation through time with t = 1. It is not optimal, as weights of
network are updated based on error vector computed only for current time step.
To overcome this simplification, backpropagation through time (BPTT) algorithm
is commonly used (see Boden for details)
in some experiments we have
achieved almost twice perplexity reduction over n-gram models by using a
recurrent network instead of a feedforward network.
it
takes around 6 hours for our basic implementation to train RNN model based on
Brown corpus (800K words, 100 hidden units and vocabulary threshold 5), while
Bengio reports 113 days for basic implementation and 26 hours with importance
sampling , when using similar data and size of neural network. [ie FNN]
we denote modified Kneser-Ney smoothed 5-gram as KN5
RNN 90/2, indicate that the hidden layer size is 90 and threshold for merging
words to rare token is 2.
we use open vocabulary language models (unknown words are assigned small
probability).
However, it does not seem that simple recurrent neural networks can capture
truly long context information, as cache models still provide complementary
information even to dynamic models trained with BPTT.
"
mikolov 2010

-> Experiment class, Ngram class, RnnKeras class
keeps logs of experiments done, makes plots


N-gram baseline

For the training step, the baseline trigram predictor was fed all word
triples, which were accumulated in the nested dictionaries and converted to
probabilities. 


### Refinement

<!-- The process of improving upon the algorithms and techniques used is clearly
documented. Both the initial and final solutions are reported, along with
intermediate solutions, if necessary. -->


parameters

Objective Fns/Loss Fns
https://keras.io/objectives/
will use categorical cross entropy -
illustrate with a simple example from abcd dataset

Initialization
uniform, gaussian, other

Optimizers
sgd - would work but too slow
rmsprop - rare features get a larger gradient
adam - like rmsprop with momentum
we'll use adam

Regularization
eg keep weights from getting too large, because _________ (leads to overfitting?)
L1, L2, other?
early stopping is a form of regularization
see https://keras.io/regularizers/
dropout - "Dropout consists in randomly setting a fraction `p` of input units to 0 at each update during training time, which helps prevent overfitting. [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](http://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf)
(Srivatsava 2014)

Embeddings
https://www.tensorflow.org/tutorials/word2vec/
http://nlp.stanford.edu/projects/glove/


show overfitting curve - too many epochs and loss starts to increase, so need to
do early stopping - eg stop if loss doesn't decrease for n epochs.
because it's fittting to the test data, but loss is against the validation data. 
so want to stop at the lowest loss point.

**add penalty for 'complexity' (ie overfitting, as with decision trees etc) - complexity ~ more nodes, more layers, and larger weights

could also do crossvalidation to get more accurate scores, but would add more training time

"overfitting is a very common problem when the dataset is too small compared with the number of model parameters that need to be learned."
so need more data, or simpler model

incl hyperparameter tuning here - list available parameters, possible ranges

take initial stab at training, get scores
then use grid search with sklearn keras wrapper to find good parameters with alice_ch1.txt
then compare those parameters with original guesses on the whole gutenberg dataset




## Results

### Model Evaluation and Validation

<!-- The final model's qualities - such as parameters - are evaluated in detail.
Some type of analysis is used to validate the robustness of the model's
solution. -->

for rnns show the Loss vs epoch graphs to show increasing accuracy

show overfitting training curves - too much training -> overfitting

robustness - ie how performs in the wild - try on some non-gutenberg text - compare diff models - ngrams, rnns


perplexity

"The lowest perplexity that has been published on the Brown Corpus (1 million
words of American English of varying topics and genres) as of 1992 is indeed
about 247 per word, corresponding to a cross-entropy of log2 247 = 7.95 bits per
word or 1.75 bits per letter using a trigram model. It is often possible to
achieve lower perplexity on more specialized corpora, as they are more
predictable."
uh... what's more uptodate value





### Justification

<!-- The final results are compared to the benchmark result or threshold with
some type of statistical analysis. Justification is made as to whether the final
model and solution is significant enough to have adequately solved the problem.
-->





## Conclusion

### Free-Form Visualization

<!-- A visualization has been provided that emphasizes an important quality
about the project with thorough discussion. Visual cues are clearly defined. -->


-> show t-SNE plots of word embeddings eg dog and cat. alice, rabbit, other
   character names vs verbs, nouns, adjectives, etc

-> add beam search


Examples of text generated by the different models:

n-gram (n=2):

- Not a sudden vibration in the gate , but in books .
- Hucheloup . `` and whatever , and the Italian , there was no additional contact information about the discovery
- She opened the brim and terrible quagmire was trying to my fancy it had n't know what may be

n-gram (n=3):

- Joy and pride was shortly to be feared on the ground at the old woman , who saw him no
- Blcher ordered Blow to attack us . Here was another , to such an authority in reference to what Boulatruelle
- Dry happiness resembles the voice of the choir , as strange as anything that was easy to inspire my pupils

n-gram (n=4):

- Terror had seized on the whole day , with intervals of listening ; and the gipsy never grudged it him .
- The glass must be violet for iron jewellery , and black for gold jewellery . 
- Dry happiness resembles dry bread . 

RNN (n=10): ?

LSTM

GRU



### Reflection

<!-- Student adequately summarizes the end-to-end problem solution and discusses
one or two particular aspects of the project they found interesting or
difficult. -->

it took a while to set up a good test harness and data preprocessing pipeline.
then getting good architecture and parameters took a while, because each epoch took so long ~30mins

**"Have in mind, however, that depending on your application, moving away from traditional n-grams with smoothing techniques may not be the best approach, since the state-of-the-art RNN-based models can be very slow to train."




### Improvement

<!-- Discussion is made as to how one aspect of the implementation could be
improved. Potential solutions resulting from these improvements are considered
and compared/contrasted to the current solution. -->

train longer, more vocabulary, more hidden nodes

online learning (?) - ie learn new vocab words like phone does

better training/testing - distribute text by paragraphs, not sentences


## References

(Abadi 2015) Abadi, Martin, et al. "TensorFlow: Large-scale machine learning on heterogeneous systems." Software available from tensorflow.org, 2015.

(Bengio 2003) Bengio, Yoshua, et al. "A neural probabilistic language model." Journal of Machine Learning Research, Feb 2003.

(Bird 2009) Bird, Steven, Edward Loper and Ewan Klein, "Natural Language Processing with Python (NLTK Book)." O'Reilly Media Inc., 2009. http://nltk.org/book

(Carlberger 1997) Carlberger, Alice, et al. "Profet, a new generation of word prediction: An evaluation study." Proceedings, ACL Workshop on Natural Language Processing for Communication Aids, 1997.

(Chelba 2013) Chelba, Ciprian, et al. "One billion word benchmark for measuring progress in statistical language modeling." arXiv preprint arXiv:1312.3005, 2013.

(Chollet 2015) Chollet, Francois, "Keras." https://github.com/fchollet/keras, 2015

(Chung 2014) Chung, Junyoung, et al. "Empirical evaluation of gated recurrent neural networks on sequence modeling." (2014) _______________link

(Coleman 1975) Coleman, Meri; and Liau, T. L., "A computer readability formula designed for machine scoring." Journal of Applied Psychology, Vol. 60, pp. 283 - 284, 1975.

(Gutenberg 2016) Project Gutenberg. (n.d.). Retrieved December 16, 2016, from www.gutenberg.org. 

(Hochreiter 1997) Hochreiter, Sepp, "Long Short-Term Memory." Neural Computation 9(8), 1997. 

(LeCun 2015) LeCun, Bengio, Hinton, "Deep learning." Nature 521, 436 - 444 (28 May 2015) doi:10.1038/nature14539

(Markov 1913) Markov, Andrei, "An example of statistical investigation of the text Eugene Onegin concerning the connection of samples in chains." Bulletin of the Imperial Academy of Sciences of St. Petersburg, Vol 7 No 3, 1913. English translation by Nitussov, Alexander et al., Science in Context, Vol 19 No 4, 2006

(Mikolov 2012) Mikolov, Tomas, PhD thesis, "Statistical Language Models Based on Neural Networks." http://www.fit.vutbr.cz/~imikolov/rnnlm/thesis.pdf, 2012

(Mikolov 2013) Mikolov, Tomas; et al. "Efficient Estimation of Word Representations in Vector Space." arXiv:1301.3781, 2013.

(Pennington 2014) Pennington, Jeffrey et al.. "GloVe: Global Vectors for Word Representation." http://nlp.stanford.edu/projects/glove/, 2014

>(Rosenblatt 1957) Rosenblatt, F. "The perceptron, a perceiving and recognizing automaton." Project Para. Cornell Aeronautical Laboratory, 1957.

(Shannon 1948) Shannon, Claude, "A Mathematical Theory of Communication." The Bell System Technical Journal, Vol. 27, July 1948.

(Srivatsava 2014) Srivatsava, Nitish, "Dropout: A Simple Way to Prevent Neural Networks from Overfitting." Journal of Machine Learning Research 15, 2014. 




--------------------------------------------------------------------------------

## Quality

### Presentation

<!-- Project report follows a well-organized structure and would be readily
understood by its intended audience. Each section is written in a clear, concise
and specific manner. Few grammatical and spelling mistakes are present.
All resources used to complete the project are cited and referenced. -->

### Functionality

<!-- Code is formatted neatly with comments that effectively explain complex
implementations.
Output produces similar results and solutions as to those
discussed in the project. -->






