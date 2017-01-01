
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
text is entered, or as an aid in resolving ambiguity in speech and handwriting
recognition.

The generation of a likely word given prior words goes back to Claude Shannon's
work on information theory [Shannon1948] based in part on Markov models
introduced by Andrei Markov [Markov1913] - counts of encountered word tuples are
used to estimate the conditional probability of seeing a word given the prior
words. These so called n-grams formed the basis of commercial word prediction
software in the 1980's, eventually supplemented with similar syntax and part of
speech predictions [Carlberger1997]. More recently, distributed representations
of words have been used in recurrent neural networks (RNNs), which can better
handle data sparsity and allow more of the context to affect the prediction
[Bengio2003].

The problem is a supervised learning task, and any text can be used to train and
evaluate the models - we'll be using a million words from books digitized by the
Gutenberg Project for evaluation. Others use larger corpora, e.g. Google's __,
__, __. [cite papers] Depending on the problem domain, different corpora might
be more appropriate, e.g. training on a chat/texting corpus would be good for a
phone text entry application.


### Problem Statement

<!-- The problem which needs to be solved is clearly defined.
A strategy for solving the problem,
including discussion of the expected solution, has been made. -->

Problem: Given a sequence of *n* words, predict the *k* most likely next words
and their probabilities.

For example, for the sequence "The dog", a solution might be
(barked 10%, slept 9%, ran 8%).

We'll use some different Recurrent Neural Network (RNN) architectures to find
the most likely next words - a standard RNN, a Long Short-Term Memory (LSTM)
RNN, and a GRU (Gated R__ Unit) RNN - and compare these against some baseline
n-gram models. Based on the results in the literature, the GRU RNN is expected
to offer the best performance for a given amount of computation [cite!].

-->>>diagrams of rnn, lstm, gru - grab from good site?


### Metrics

<!-- Metrics used to measure performance of a model or result are clearly
defined. Metrics are justified based on the characteristics of the problem. -->

will use accuracy or mean error rate
-->>>see what is used in lit



## Analysis

### Data Exploration

<!-- If a dataset is present, _features_ and _calculated statistics_ relevant to the
problem have been reported and discussed, along with a _sampling_ of the data.
_Abnormalities_ or characteristics about the data or input that need to
be addressed have been identified. -->

The training and testing data are obtained from ten books from Project
Gutenberg, totalling nearly a million words -

    | Author                 | Year | Title                         | Gutenberg # |  Words | Chars/Word | Words/Sentence |
    |------------------------+------+-------------------------------+-------------+--------+------------+----------------|
    | Victor Hugo            | 1862 | Les Miserables                |         135 | 563030 |       5.70 |           15.8 |
    | Lewis Carroll          | 1865 | Alice in Wonderland           |       28885 |  26719 |       5.55 |           16.3 |
    | Robert Louis Stevenson | 1883 | Treasure Island               |         120 |  67872 |       5.31 |           18.1 |
    | Henry James            | 1898 | The Turn of the Screw         |         209 |  42278 |       5.35 |           16.7 |
    | Joseph Conrad          | 1899 | Heart of Darkness             |         219 |  37928 |       5.51 |           15.5 |
    | M R James              | 1905 | Ghost Stories of an Antiquary |        8486 |  45268 |       5.50 |           20.6 |
    | Arthur Machen          | 1907 | The Hill of Dreams            |       13969 |  65861 |       5.52 |           27.8 |
    | Kenneth Graham         | 1908 | The Wind in the Willows       |         289 |  58366 |       5.51 |           18.1 |
    | P G Woodhouse          | 1919 | My Man Jeeves                 |        8164 |  50834 |       5.38 |           10.8 |
    | M R James              | 1920 | A Thin Ghost and Others       |       20387 |  31295 |       5.30 |           22.2 |
    |------------------------+------+-------------------------------+-------------+--------+------------+----------------|
    | Total                  |      |                               |             | 989451 |            |                |

The texts can be found at, for example http://www.gutenberg.org/etext/28885.

Some sample text:

    The landscape was gloomy and deserted. He was encompassed by space.
    There was nothing around him but an obscurity in which his gaze was
    lost, and a silence which engulfed his voice.
    - Les Miserables

    From the eminence of the lane, skirting the brow of a hill, he looked down
    into deep valleys and dingles, and beyond, across the trees, to remoter
    country, wild bare hills and dark wooded lands meeting the grey still sky.
    - The Hill of Dreams

    Meantime the Rat, warm and comfortable, dozed by his fireside. His paper
    of half-finished verses slipped from his knee, his head fell back, his
    mouth opened, and he wandered by the verdant banks of dream-rivers. 
    - The Wind in the Willows


### Exploratory Visualization

<!-- A visualization has been provided that summarizes or extracts a relevant
characteristic or feature about the dataset or input data with thorough
discussion. Visual cues are clearly defined. -->

make a histogram of sentence lengths
but why?

![Sentence length distributions][1]

[1]: images/sentence_lengths.png


keyword is *relevant* - what vis would be relevant for this problem?

we're doing word prediction
maybe something more like information content?
ie how compressible the text is?
ie how predictable it is?
cf pure randomness (log2 26 ~ (log 26 2) ~ 4.7bits?)
how calculate? ngrams? 

->> information content of english - shannon paper
use to compare texts?




### Algorithms and Techniques

<!-- Algorithms and techniques used in the project are thoroughly discussed and
properly justified based on the characteristics of the problem. -->








until recently [when?] trigram was state of the art in word prediction.
can't use much bigger contexts than trigram because too many possibilities to store (calculate) and most counts would be zero.
have to back off to digrams when trigram counts are too small. 
eg if prompt is "dinosaur pizza", and you've never seen that pair before, must backoff to the bigram "pizza ___". 

but trigrams fail to use a lot of information that can be used to predict the next word.
doesn't understand similarities between words, eg cat and dog
so need to convert words into vector of syntactic and semantic features, and use the features to predict next word.
allows us to use much larger context, eg previous 10 words.
bengio pioneered this. (year? 2003?)

have huge softmax layer
skipmax connections go straight from input to output words ?
was slightly worse than trigram
since then have been improved considerably
(how?)
(this was a plain rnn?)
lstm's came in 1997
various types, incl gru 2014 - simpler
then attention 2015




### Benchmark

<!-- Student clearly defines a benchmark result or threshold for comparing
performances of solutions obtained. -->

n-grams - 1,2,3,4 trained on 1m words, 5-gram trained on billions (google)?

-> find published results, history



## Methodology

### Data Preprocessing

<!-- All preprocessing steps have been clearly documented. Abnormalities or
characteristics about the data or input that needed to be addressed have been
corrected. If no data preprocessing is necessary, it has been clearly justified.
-->

The raw Gutenberg files are downloaded, then cleaned, merged, and split into
training, validation, and test sets.

To clean the files, headers and footers with Project Gutenberg and license
information are removed via regular expression searches for the delimiter
strings. Some texts contain titlepages and tables of contents also, which are
removed similarly where possible. All non-ASCII characters are removed, as they
caused problems for NLTK in Python 2.7.

Once the files are cleaned, they are merged into a single file, which is then
split into the training, validation, and test files. This is done by parsing the
text into sentences, and apportioning them to the different files based on the
desired proportions (e.g. 80% training, 10% validation, 10% testing).


### Implementation

<!-- The process for which metrics, algorithms, and techniques were implemented
with the given datasets or input data has been thoroughly documented.
Complications that occurred during the coding process are discussed. -->




### Refinement

<!-- The process of improving upon the algorithms and techniques used is clearly
documented. Both the initial and final solutions are reported, along with
intermediate solutions, if necessary. -->



## Results

### Model Evaluation and Validation

<!-- The final model's qualities - such as parameters - are evaluated in detail.
Some type of analysis is used to validate the robustness of the model's
solution. -->

for rnns show the Loss vs epoch graphs to show increasing accuracy




### Justification

<!-- The final results are compared to the benchmark result or threshold with
some type of statistical analysis. Justification is made as to whether the final
model and solution is significant enough to have adequately solved the problem.
-->


## Conclusion

### Free-Form Visualization

<!-- A visualization has been provided that emphasizes an important quality
about the project with thorough discussion. Visual cues are clearly defined. -->

Examples of text generated by the different models:

n-gram (n=2)
Not a sudden vibration in the gate , but in books .
Hucheloup . `` and whatever , and the Italian , there was no additional contact information about the discovery
And the old stockade . The lip was indubitable -- THE FOUNDATION , your breakfast ? The woman of
Madeleine arrived there were all the neck was not the sister 's the banks of a bore a few
She opened the brim and terrible quagmire was trying to my fancy it had n't know what may be
Ballad of which sped I COULD HARDLY BELIEVE IN FAIRY LAND I was higher than you give thee From

n-gram (n=3)
LIKE SUMMER TEMPESTS CAME HIS TEARS XII . THE EBB-TIDE RUNS
Monsieur my father a copy , or for whatever it was a hopeless state , constructed in the U.S. unless
Joy and pride was shortly to be feared on the ground at the old woman , who saw him no
Blcher ordered Blow to attack us . Here was another , to such an authority in reference to what Boulatruelle
A pile of stones , destined to weep .
Dry happiness resembles the voice of the choir , as strange as anything that was easy to inspire my pupils
My spirits rose as I lay with half-closed eyes . I was not so with me out of the hand

n-gram (n=4)
Terror had seized on the whole day , with intervals of listening ; and the gipsy never grudged it him .
The glass must be violet for iron jewellery , and black for gold jewellery . 
Dry happiness resembles dry bread . 
Just like a rocket too , it burst in the air of splendidly coloured fire-flies , which sped hither and thither .


ffnn

rnn

lstm

?



### Reflection

<!-- Student adequately summarizes the end-to-end problem solution and discusses
one or two particular aspects of the project they found interesting or
difficult. -->

it took a while to set up a good test harness and data preprocessing pipeline.



### Improvement

<!-- Discussion is made as to how one aspect of the implementation could be
improved. Potential solutions resulting from these improvements are considered
and compared/contrasted to the current solution. -->



## References




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






