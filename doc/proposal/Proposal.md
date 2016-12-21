
# Capstone Proposal

<!-- this is the initial draft with the rubric in comments - the final version
was put into in libreoffice -->


**Machine Learning Engineer Nanodegree**

Brian Burns  
December 16, 2016  
Proposal: **Word Prediction using Recurrent Neural Networks**  

<!-- (approx. 2-3 pages) -->


## Domain Background

<!-- (approx. 1-2 paragraphs) -->
<!--
In this section, provide brief details on the background information of the
domain from which the project is proposed.

Historical information relevant to the project should be included.

It should be clear how or why a problem in the domain can or should be solved.

Related academic research should be appropriately cited in this section,
including why that research is relevant.

Additionally, a discussion of your personal motivation for investigating a
particular problem in the domain is encouraged but not required.
-->

Word prediction is the task of predicting the most likely words following the
preceding text. This has many applications, such as suggesting the next word as
text is entered, as an aid in speech and handwriting recognition, or generating
text to help fight spam.

The generation of a likely word given prior words goes back to Claude Shannon's
paper on Information Theory [Shannon1948] which was based on Markov models
introduced by Andrei Markov [Markov1913]. These n-grams formed the basis of
commercial word prediction software in the 1980's, eventually supplemented with
similar syntax and part of speech predictions [Carlberger1997]. More recently,
distributed representations of words have been used in recurrent neural networks
(RNNs), which can better handle data sparsity and allow for more of the context
to affect the prediction [Bengio2003].

My interest in the task stems from the desire to learn more about machine
learning with Natural Language Processing (NLP), Markov models, the Python
Natural Language Toolkit (NLTK), distributed word representations, word2vec,
RNNs, LSTM, and Deep Learning with Keras and TensorFlow.


## Problem Statement

<!-- (approx. 1 paragraph) -->
<!--
In this section, clearly describe the problem that is to be solved.

The problem described should be well defined and should have at least one relevant potential solution.

Additionally, describe the problem thoroughly such that it is clear that the problem is
- quantifiable (the problem can be expressed in mathematical or logical terms) ,
- measurable (the problem can be measured by some metric and clearly observed), and
- replicable (the problem can be reproduced and occurs more than once).
-->

Problem: Given a sequence of *n* words, what are the *k* most likely words that
might follow?

For instance, given the sequence "Alice went to the", the program should return
a list of words like "forest", "library", and "mountains", depending on the
sentences the program was trained on.


## Datasets and Inputs

<!-- (approx. 2-3 paragraphs) -->
<!--
In this section, the dataset(s) and/or input(s) being considered for the
project should be thoroughly described, such as how they relate to the problem
and why they should be used.

Information such as how the dataset or input is (was) obtained, and the
characteristics of the dataset or input, should be included with relevant
references and citations as necessary.

It should be clear how the dataset(s) or input(s) will be used in the project
and whether their use is appropriate given the context of the problem.
-->

Any text can be used, so long as it has words separated by spaces. For this
project texts from the Gutenberg project [Gutenberg2016] will be used, as an
arbitrary number of words can be easily obtained without copyright or licensing
issues (texts before 1922 are out of copyright). More modern corpora could also
be used, such as Twitter posts or chat logs, if the purpose were to predict more
contemporary writing.

The models will be trained with up to approximately a million words, using the
following texts:

    | Author                 | Year | Title                         | Gutenberg # |   Words |
    |------------------------+------+-------------------------------+-------------+---------|
    | George MacDonald       | 1858 | Phantastes                    |         325 |   72334 |
    | Victor Hugo            | 1862 | Les Miserables                |         135 |  568690 |
    | Lewis Carroll          | 1865 | Alice in Wonderland           |       28885 |   30355 |
    | Robert Louis Stevenson | 1883 | Treasure Island               |         120 |   71611 |
    | Henry James            | 1898 | The Turn of the Screw         |         209 |   45294 |
    | M R James              | 1905 | Ghost Stories of an Antiquary |        8486 |   47891 |
    | Arthur Machen          | 1907 | The Hill of Dreams            |       13969 |   68849 |
    | Kenneth Graham         | 1908 | The Wind in the Willows       |         289 |   61461 |
    | P G Woodhouse          | 1919 | My Man Jeeves                 |        8164 |   53858 |
    | M R James              | 1920 | A Thin Ghost and Others       |       20387 |   34293 |
    |------------------------+------+-------------------------------+-------------+---------|
    | Total                  |      |                               |             | 1054636 |

The texts can be found at, for example http://www.gutenberg.org/etext/28885.


## Solution Statement

<!-- (approx. 1 paragraph) -->
<!--
In this section, clearly describe a solution to the problem.

The solution should be applicable to the project domain and appropriate for the
dataset(s) or input(s) given.

Additionally, describe the solution thoroughly such that it is clear that the solution is
- quantifiable (the solution can be expressed in mathematical or logical terms) ,
- measurable (the solution can be measured by some metric and clearly observed), and
- replicable (the solution can be reproduced and occurs more than once).
-->

A Recurrent Neural Network (RNN) will be used to predict the next word in a
sequence. In such networks, a sequence of words is encoded as a set of word
vectors in a high-dimensional space (e.g. 300), and the network is trained until
the output is within a certain distance of the actual word. Then on testing, a
sequence of words will be fed into the network and the output used to search the
vector space for the closest *k* words.


## Benchmark Model

<!-- (approximately 1-2 paragraphs) -->
<!--
In this section, provide the details for a benchmark model or result that
relates to the domain, problem statement, and intended solution.

Ideally, the benchmark model or result contextualizes existing methods or known
information in the domain and problem given, which could then be objectively
compared to the solution.

Describe how the benchmark model or result is measurable (can be measured by
some metric and clearly observed) with thorough detail. <<<<<<<<
-->

For the benchmark model a simple n-gram model will be used - this is a standard
approach for next word prediction. A nested dictionary is created based on the
training data, which counts occurrences of n-tuples of words. These are then
normalized to get a probability distribution, which can be used to predict the
most likely words following a sequence.


## Evaluation Metrics

<!-- (approx. 1-2 paragraphs) -->
<!--
In this section, propose at least one evaluation metric that can be used to
quantify the performance of both the benchmark model and the solution model.

The evaluation metric(s) you propose should be appropriate given the context of
the data, the problem statement, and the intended solution.

Describe how the evaluation metric(s) are derived and provide an example of
their mathematical representations (if applicable).

Complex evaluation metrics should be clearly defined and quantifiable (can be
expressed in mathematical or logical terms).

-->

For evaluation of both approaches, the accuracy score will be reported, for
increasing training set sizes. The dataset will be split into training and test
sets - after training, a sequence of words from the test set will be chosen at
random, then fed to the predictor, and the most likely *k* words compared to the
actual following word.

Accuracy: # correct predictions / # total predictions

A prediction will be considered *correct* if the actual word is in the list of
*k* most likely words - this is relevant to the task of presenting the user with
a list of most likely next words as they are entering text.


## Project Design

<!-- (approx. 1 page) -->
<!--
In this final section, summarize a theoretical workflow for approaching a
solution given the problem.

Provide thorough discussion for what strategies you may consider employing, what
analysis of the data might be required before being used, or which algorithms
will be considered for your implementation.

The workflow and discussion that you provide should align with the qualities of
the previous sections.

Additionally, you are encouraged to include small visualizations, pseudocode, or
diagrams to aid in describing the project design, but it is not required.

The discussion should clearly outline your intended workflow of the capstone
project.

-->

The texts will first be preprocessed to remove punctuation and converted to
lowercase.

For the training step, the baseline trigram predictor will be fed all word
triples, which will be accumulated in the nested dictionaries and converted to
probabilities. For the RNN predictor, all word sequences or skip-grams will be
fed to the network and trained until its output is close to the correct final
word.

For the testing step, the baseline predictor will be fed random tuples of words,
and the top *k* predicted words will be compared against the actual word, and an
accuracy score tallied. For the RNN predictor, the same process will be used.

Training sets of increasing sizes will be used - 1k, 10k, 100k, 1 million words,
and the results recorded for comparison. Timing and memory information will also
be recorded for all processes for later analysis.

<!-- --- -->

<!-- Before submitting your proposal, ask yourself. . . -->

<!-- Does the proposal you have written follow a well-organized structure similar to that of the project template? -->
<!-- Is each section (particularly Solution Statement and Project Design) written in a clear, concise and specific fashion? Are there any ambiguous terms or phrases that need clarification? -->
<!-- Would the intended audience of your project be able to understand your proposal? -->
<!-- Have you properly proofread your proposal to assure there are minimal grammatical and spelling mistakes? -->
<!-- Are all the resources used for this project correctly cited and referenced? -->


## References

<!-- [TD03] Erik F. Tjong Kim Sang and Fien De Meulder, Introduction to the CoNLL-2003 Shared Task: Language-Independent Named Entity Recognition. In: Proceedings of CoNLL-2003, Edmonton, Canada, 2003, pp. 142-147. -->

[Bengio2003] Bengio, Yoshua, et al. "A neural probabilistic language model." Journal of Machine Learning Research, Feb 2003.

[Carlberger1997] Carlberger, Alice, et al. "Profet, a new generation of word prediction: An evaluation study." Proceedings, ACL Workshop on Natural Language Processing for Communication Aids, 1997.

[Gutenberg2016] Project Gutenberg. (n.d.). Retrieved December 16, 2016, from www.gutenberg.org. 

[Markov1913] Markov, Andrei, "An example of statistical investigation of the text Eugene Onegin concerning the connection of samples in chains." Bulletin of the Imperial Academy of Sciences of St. Petersburg, Vol 7 No 3, 1913. English translation by Nitussov, Alexander et al., Science in Context, Vol 19 No 4, 2006

[Shannon1948] Shannon, Claude, "A Mathematical Theory of Communication." The Bell System Technical Journal, Vol. 27, July 1948.

