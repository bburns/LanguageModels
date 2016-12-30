
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

<!-- If a dataset is present, features and calculated statistics relevant to the
problem have been reported and discussed, along with a sampling of the data.
Abnormalities or characteristics about the data or input that need to
be addressed have been identified. -->

The training and testing data are supplied by ten books from Project Gutenberg,
totalling nearly a million words - the texts are cleaned up (headers and
footers/licenses removed, non-ASCII characters removed, and table of contents
removed where possible), and merged, then split into training, validation, and
test sets.

dataset
ncharacters
nletters/word
nwords
nsentences
nwords/sentence, do histogram


    | Author                 | Year | Title                         | Gutenberg # |  Words | 
    |------------------------+------+-------------------------------+-------------+--------|
    | Victor Hugo            | 1862 | Les Miserables                |         135 | 563030 |
    | Lewis Carroll          | 1865 | Alice in Wonderland           |       28885 |  26719 |
    | Robert Louis Stevenson | 1883 | Treasure Island               |         120 |  67872 |
    | Henry James            | 1898 | The Turn of the Screw         |         209 |  42278 |
    | Joseph Conrad          | 1899 | Heart of Darkness             |         219 |  37928 |
    | M R James              | 1905 | Ghost Stories of an Antiquary |        8486 |  45268 |
    | Arthur Machen          | 1907 | The Hill of Dreams            |       13969 |  65861 |
    | Kenneth Graham         | 1908 | The Wind in the Willows       |         289 |  58366 |
    | P G Woodhouse          | 1919 | My Man Jeeves                 |        8164 |  50834 |
    | M R James              | 1920 | A Thin Ghost and Others       |       20387 |  31295 |
    |------------------------+------+-------------------------------+-------------+--------|
    | Total                  |      |                               |             | 989451 |


The texts can be found at, for example http://www.gutenberg.org/etext/28885.








samples

    The landscape was gloomy and deserted. He was encompassed by space.
    There was nothing around him but an obscurity in which his gaze was
    lost, and a silence which engulfed his voice.
    -lesmiserables

    Meantime the Rat, warm and comfortable, dozed by his fireside. His paper
    of half-finished verses slipped from his knee, his head fell back, his
    mouth opened, and he wandered by the verdant banks of dream-rivers. 
    -wind

cleaning/abnormalities/characteristics
oddities like . . . . ., ***, * * * *, table of contents, gutenberg preface, hyphens, different quotation marks, unicode characters?
how clean them up

count occurrences of . . . . , ***, quotation mark types, non-ascii characters

gutenberg preface, eg - 

    The Project Gutenberg EBook of Les Misérables, by Victor Hugo

    This eBook is for the use of anyone anywhere at no cost and with almost
    no restrictions whatsoever. You may copy it, give it away or re-use
    it under the terms of the Project Gutenberg License included with this
    eBook or online at www.gutenberg.org

    Title: Les Misérables Complete in Five Volumes
    Author: Victor Hugo
    Translator: Isabel F. Hapgood
    Release Date: June 22, 2008 [EBook #135]
    Last Updated: January 18, 2016
    Language: English
    Character set encoding: UTF-8
    *** START OF THIS PROJECT GUTENBERG EBOOK LES MISÉRABLES ***
    Produced by Judith Boss and David Widger

title page, eg

    LES MISÉRABLES
    By Victor Hugo
    Translated by Isabel F. Hapgood
    Thomas Y. Crowell & Co. No. 13, Astor Place New York
    Copyright 1887
    Enlarge
    Enlarge
    Enlarge
    Enlarge
    Enlarge
    Enlarge

table of contents, eg

    Contents
    LES MISÉRABLES
    VOLUME I.--FANTINE.
    PREFACE
    BOOK FIRST--A JUST MAN
    CHAPTER I--M. MYRIEL
    CHAPTER II--M. MYRIEL BECOMES M. WELCOME
    CHAPTER III--A HARD BISHOPRIC FOR A GOOD BISHOP
    CHAPTER IV--WORKS CORRESPONDING TO WORDS
    CHAPTER V--MONSEIGNEUR BIENVENU MADE HIS CASSOCKS LAST TOO LONG
    CHAPTER VI--WHO GUARDED HIS HOUSE FOR HIM
    CHAPTER VII--CRAVATTE
    CHAPTER VIII--PHILOSOPHY AFTER DRINKING
    CHAPTER IX--THE BROTHER AS DEPICTED BY THE SISTER
    CHAPTER X--THE BISHOP IN THE PRESENCE OF AN UNKNOWN LIGHT
    CHAPTER XI--A RESTRICTION
    CHAPTER XII--THE SOLITUDE OF MONSEIGNEUR WELCOME
    CHAPTER XIII--WHAT HE BELIEVED
    CHAPTER XIV--WHAT HE THOUGHT

gutenberg license at end, eg

    End of the Project Gutenberg EBook of My Man Jeeves, by P. G. Wodehouse
    *** END OF THIS PROJECT GUTENBERG EBOOK MY MAN JEEVES ***
    ***** This file should be named 8164-8.txt or 8164-8.zip *****
    This and all associated files of various formats will be found in:
    http://www.gutenberg.org/8/1/6/8164/
    Produced by Suzanne L. Shell, Charles Franks and the Online
    Distributed Proofreading Team
    Updated editions will replace the previous one--the old editions
    will be renamed.
    [and 15 page license...]


gutenberg header and footer markers - not consistent, but regexp should capture diffs. eg


    *** START OF THIS PROJECT GUTENBERG EBOOK PHANTASTES ***
    *** START OF THIS PROJECT GUTENBERG EBOOK LES MISÉRABLES ***
    *** START OF THIS PROJECT GUTENBERG EBOOK ALICE'S ADVENTURES IN WONDERLAND ***
    *** START OF THIS PROJECT GUTENBERG EBOOK TREASURE ISLAND ***
    *** START OF THIS PROJECT GUTENBERG EBOOK THE TURN OF THE SCREW ***
    *** START OF THE PROJECT GUTENBERG EBOOK GHOST STORIES OF AN ANTIQUARY ***
    ***START OF THE PROJECT GUTENBERG EBOOK THE HILL OF DREAMS***
    *** START OF THIS PROJECT GUTENBERG EBOOK THE WIND IN THE WILLOWS ***
    *** START OF THIS PROJECT GUTENBERG EBOOK MY MAN JEEVES ***
    ***START OF THE PROJECT GUTENBERG EBOOK A THIN GHOST AND OTHERS***
    
    *** END OF THIS PROJECT GUTENBERG EBOOK PHANTASTES ***
    *** END OF THIS PROJECT GUTENBERG EBOOK LES MISÉRABLES ***
    *** END OF THIS PROJECT GUTENBERG EBOOK ALICE'S ADVENTURES IN WONDERLAND ***
    *** END OF THIS PROJECT GUTENBERG EBOOK TREASURE ISLAND ***
    *** END OF THIS PROJECT GUTENBERG EBOOK THE TURN OF THE SCREW ***
    *** END OF THE PROJECT GUTENBERG EBOOK GHOST STORIES OF AN ANTIQUARY ***
    ***END OF THE PROJECT GUTENBERG EBOOK THE HILL OF DREAMS***
    *** END OF THIS PROJECT GUTENBERG EBOOK THE WIND IN THE WILLOWS ***
    *** END OF THIS PROJECT GUTENBERG EBOOK MY MAN JEEVES ***
    ***END OF THE PROJECT GUTENBERG EBOOK A THIN GHOST AND OTHERS***




### Exploratory Visualization

<!-- A visualization has been provided that summarizes or extracts a relevant
characteristic or feature about the dataset or input data with thorough
discussion. Visual cues are clearly defined. -->




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
bengio pioneered this. (year?)
huge softmax layer
skipmax connections go straight from input to output words
was slightly worse than trigram
since then have been improved considerably
>how?
>this was a plain rnn?
lstm's came in 1997
various types, incl gru 2014 - simpler
then attention 2015




### Benchmark

<!-- Student clearly defines a benchmark result or threshold for comparing
performances of solutions obtained. -->

n-grams - 1,2,3,4 trained on 1m words, 5-gram trained on billions (google)?

published results? 



## Methodology

### Data Preprocessing

<!-- All preprocessing steps have been clearly documented. Abnormalities or
characteristics about the data or input that needed to be addressed have been
corrected. If no data preprocessing is necessary, it has been clearly justified.
-->


cleanup gutenberg preface, titlepages, table of contents with code, or just leave in as noise

01-raw
02-cleaned
03-merged
04-split





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

### Improvement

<!-- Discussion is made as to how one aspect of the implementation could be
improved. Potential solutions resulting from these improvements are considered
and compared/contrasted to the current solution. -->




--------------------------------------------------------------------------------

## Quality

### Presentation

<!-- Project report follows a well-organized structure and would be readily
understood by its intended audience. Each section is written in a clear, concise
and specific manner. Few grammatical and spelling mistakes are present. All
resources used to complete the project are cited and referenced. -->

### Functionality

<!-- Code is formatted neatly with comments that effectively explain complex
implementations. Output produces similar results and solutions as to those
discussed in the project. -->






