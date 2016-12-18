
# Word Prediction using Recurrent Neural Networks

Udacity Machine Learning Nanodegree Capstone Project


## About

Word prediction is the task of predicting the most likely words following the
preceding text. This has many applications, such as suggesting the next word as
text is entered, as an aid in speech and handwriting recognition, or generating
text to help fight spam.


## Goals

* Implement an n-gram predictor as a baseline
* Implement a recurrent neural network (RNN) to try to improve on the baseline
* Compare accuracy of the two methods for different training set sizes
* Compare time and space complexity and usage of the two algorithms


## Data

The texts used to train the models can be obtained from Project Gutenberg at the
following URLs:

http://www.gutenberg.org/ebooks/325.txt.utf-8
http://www.gutenberg.org/ebooks/135.txt.utf-8
http://www.gutenberg.org/ebooks/28885.txt.utf-8
http://www.gutenberg.org/ebooks/120.txt.utf-8
http://www.gutenberg.org/ebooks/209.txt.utf-8
http://www.gutenberg.org/ebooks/8486.utf-8
http://www.gutenberg.org/ebooks/13969.txt.utf-8
http://www.gutenberg.org/ebooks/289.txt.utf-8
http://www.gutenberg.org/ebooks/8164.txt.utf-8
http://www.gutenberg.org/ebooks/20387.txt.utf-8

The word2vec project page is at https://code.google.com/archive/p/word2vec/ -
the word2vec word vectors, trained on 100 billion words from Google News, can be
obtained from https://docs.google.com/uc?id=0B7XkCwpI5KDYNlNUTTlSS21pQmM&export=download
(1.5GB).


## Usage

Preprocess the texts - this cleans up the files in the `texts` directory and
writes them to the `processed` directory.

    > wp preprocess

Train the models

    > wp train ngram
    > wp train rnn

Test the models

    > wp test ngram
    > wp test rnn


## License

GPL
