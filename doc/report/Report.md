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
preceding text. This has many applications, such as suggesting the next word as
text is entered, as an aid in resolving ambiguity in speech and handwriting
recognition, or machine translation.

The generation of a likely word given prior words goes back to Claude Shannon's
work on information theory (Shannon 1948) based in part on Markov models
introduced by Andrei Markov (Markov 1913) - counts of encountered word tuples are
used to estimate the conditional probability of seeing a word given the prior
words. These so called n-grams formed the basis of commercial word prediction
software in the 1980's, eventually supplemented with similar syntax and part of
speech predictions (Carlberger 1997).

More recently, distributed representations of words have been used in recurrent
neural networks (RNNs), which can better handle data sparsity and allow more of
the context to affect the prediction (Bengio 2003).

The problem is a supervised learning task, and any text can be used to train and
evaluate the models - we'll be using a million words from books digitized by the
Gutenberg Project (Gutenberg 2016) for evaluation. Others use larger corpora,
e.g. Google's billion word corpus (Chelba 2013). Depending on the problem
domain, different corpora might be more appropriate, e.g. training on a
chat/texting corpus would be good for a phone text entry application.


"The main advantage of NNLMs over n-grams is that history is no longer seen as
exact sequence of n - 1 words H, but rather as a projection of H into some lower
dimensional space. This reduces number of parameters in the model that have to
be trained, resulting in automatic clustering of similar histories." mikolov 2012 thesis
The hidden layer of RNN represents all previous
history and not just n -1 previous words, thus the model can theoretically represent long
context patterns
however the error gradients quickly vanish as they get backpropagated in time
(in rare cases the errors can explode), so several steps of unfolding are
sufficient (this is sometimes referred to as truncated BPTT). While for word
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

The primary metric used to evaluate the performance of the models will be
**relevance**, which we'll define as

> **Relevance** = # correct predictions / # total predictions

where a prediction will be considered *correct* if the actual word is in the
list of *k* most likely words. This is relevant to the task of presenting the
user with a list of most likely next words as they are entering text - we'll use
*k* = 3 for evaluation.

We'll also report the **accuracy** in places, which measures the number of
predictions where the most likely prediction is the correct one (which is just
*relevance* where *k* = 1).

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
Gutenberg, totalling 1,008,825 words -

<!-- note: can make this fixed chars by indenting, but needs to be at left margin to make a latex table -->
<!-- this is output from print(util.table(data.analyze())) -->

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

The grade level is calculated using the Coleman-Liau Index (Coleman 1975).

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

![Sentence length distributions](images/sentence_lengths_boxplot.png)


make a histogram of sentence lengths
but why?
keyword is *relevant* - what vis would be relevant for this problem?
and *thorough discussion* - needs to be something interesting. 

we're doing word prediction
maybe something more like information content?
ie how compressible the text is?
ie how predictable it is?
cf pure randomness (log2 26 ~ (log 26 2) ~ 4.7bits?)
how calculate? ngrams? 

-> information content of english - shannon paper
use to compare texts?
plot against mean/median sentence lengths?







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

An RNN is able to make predictions based on words further back in the sequence,
e.g. 10 words, because it can represent words more compactly with an internal
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

<!-- is this correct about U? read something different.  -->
<!-- what about matrix E the embedding layer? it's separate from U, eh? -->
The matrix **U** amounts to a table of word embeddings in a vector space of many
dimensions (which could be e.g. 50-300) - each word in the vocabulary
corresponds with a row in the table, and the dot product between any two words
gives their similarity, once the network is trained. Alternatively, pre-trained
word embeddings, such as word2vec (Mikolov 2013) or GloVe (Pennington 2014),
can be used to save on training time.

The matrix **W** acts as a filter on the internal hidden state, which represents
the prior context.

The matrix **V** allows each word in the vocabulary to 'vote' on how likely it
thinks it will be next, based on the context (current + previous words). The
softmax layer then converts these scores into probabilities, so the top *k* most
likely words can be found for a given context.


-> show calcs and matrices for abcd example - nvocab=5, nhidden=2, incl loss vs
   accuracy, perplexity

-> explain LSTM and GRU briefly

see http://www.wildml.com/2015/10/recurrent-neural-network-tutorial-part-4-implementing-a-grulstm-rnn-with-python-and-theano/
incls diagrams
"GRUs have fewer parameters (U and W are smaller) and thus may train a bit
faster or need less data to generalize. On the other hand, if you have enough
data, the greater expressive power of LSTMs may lead to better results."

-> lstm's came in 1997 [cite], gru 2014 - simpler [cite] (Chung 2014)

-> then attention 2015? discuss briefly, cite

will use Keras (Chollet 2015), a library that simplifies the use of TensorFlow [cite], e.g. 
-> compare Keras code vs TensorFlow for same simple RNN


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

The raw Gutenberg files were downloaded, then cleaned, merged, and split into
training, validation, and test sets.

To clean the files, headers and footers with Project Gutenberg and license
information were removed via regular expression searches for the delimiter
strings. Some texts contain titlepages and tables of contents also, which were
removed similarly where possible. 

Once the files were cleaned, they were merged into a single file, which was then
split into the training, validation, and test files. This was done by parsing the
text into paragraphs, and portioning them out to the different files based on the
desired proportions (e.g. 95% training, 2.5% validation, 2.5% testing).


### Implementation

<!-- The process for which metrics, algorithms, and techniques were implemented
with the given datasets or input data has been thoroughly documented.
Complications that occurred during the coding process are discussed. -->

The texts were first preprocessed to remove punctuation and converted to
lowercase - better accuracy could be achieved by leaving text case as it is, but
this would increase the vocabulary size by a significant factor, and so require
more training time.

For the training step, the baseline trigram predictor was fed all word
triples, which were accumulated in the nested dictionaries and converted to
probabilities. For the RNN predictor, all word sequences were fed to the
network and trained for a certain number of epochs, or until the cross-entropy
loss stopped improving for a certain number of epochs.

For the testing step, the baseline and RNN predictors were fed word sequences
from the test data, and the top *k* predicted words were compared against the
actual word, and a *relevance* score tallied.

Training sets of increasing sizes were used - 1k, 10k, 100k, 1 million words,
and the results recorded for comparison. Timing and memory information were also
recorded for all processes for analysis.


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





### Improvement

<!-- Discussion is made as to how one aspect of the implementation could be
improved. Potential solutions resulting from these improvements are considered
and compared/contrasted to the current solution. -->

train longer, more vocabulary, more hidden nodes

online learning (?) - ie learn new vocab words like phone does

better training/testing - distribute text by paragraphs, not sentences


## References

(Bengio 2003) Bengio, Yoshua, et al. "A neural probabilistic language model." Journal of Machine Learning Research, Feb 2003.

(Bird 2009) Bird, Steven, Edward Loper and Ewan Klein, Natural Language Processing with Python ("NLTK Book"). O'Reilly Media Inc., 2009. http://nltk.org/book

(Carlberger 1997) Carlberger, Alice, et al. "Profet, a new generation of word prediction: An evaluation study." Proceedings, ACL Workshop on Natural Language Processing for Communication Aids, 1997.

(Chelba 2013) Chelba, Ciprian, et al. "One billion word benchmark for measuring progress in statistical language modeling." arXiv preprint arXiv:1312.3005, 2013.

(Chollet 2015) Chollet, Francois, Keras, https://github.com/fchollet/keras, 2015

(Chung 2014) Chung, Junyoung, et al. "Empirical evaluation of gated recurrent neural networks on sequence modeling." (2014) >>>>>>>>>link

(Coleman 1975) Coleman, Meri; and Liau, T. L., "A computer readability formula designed for machine scoring", Journal of Applied Psychology, Vol. 60, pp. 283 - 284, 1975.

(Gutenberg 2016) Project Gutenberg. (n.d.). Retrieved December 16, 2016, from www.gutenberg.org. 

(LeCun 2015) LeCun, Bengio, Hinton, "Deep learning", Nature 521, 436 - 444 (28 May 2015) doi:10.1038/nature14539

(Markov 1913) Markov, Andrei, "An example of statistical investigation of the text Eugene Onegin concerning the connection of samples in chains." Bulletin of the Imperial Academy of Sciences of St. Petersburg, Vol 7 No 3, 1913. English translation by Nitussov, Alexander et al., Science in Context, Vol 19 No 4, 2006

(Mikolov 2013) Mikolov, Tomas; et al. "Efficient Estimation of Word Representations in Vector Space". arXiv:1301.3781, 2013.

(Pennington 2014) Pennington, Jeffrey et al.. "GloVe: Global Vectors for Word Representation". http://nlp.stanford.edu/projects/glove/ 2014

>(Rosenblatt 1957) Rosenblatt, F. "The perceptron, a perceiving and recognizing automaton" Project Para. Cornell Aeronautical Laboratory, 1957.

(Shannon 1948) Shannon, Claude, "A Mathematical Theory of Communication." The Bell System Technical Journal, Vol. 27, July 1948.

>(shri... 2014) shri with hinton on dropout 2014




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






