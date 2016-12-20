
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
* Generate some text using both models and compare subjectively


## Data

The texts used to train the models can be obtained from Project Gutenberg at the
following URLs:

better - 
http://www.gutenberg.org/files/325/325-0.txt
http://www.gutenberg.org/files/135/135-0.txt
http://www.gutenberg.org/files/28885/28885-0.txt
http://www.gutenberg.org/files/120/120-0.txt
http://www.gutenberg.org/files/209/209-0.txt
http://www.gutenberg.org/files/8486/8486-0.txt
http://www.gutenberg.org/files/13969/13969-0.txt
http://www.gutenberg.org/files/289/289-0.txt
http://www.gutenberg.org/files/8164/8164-0.txt
http://www.gutenberg.org/files/20387/20387-0.txt

The word2vec project page is at https://code.google.com/archive/p/word2vec/ -
the word2vec word vectors, trained on 100 billion words from Google News, can be
obtained from https://docs.google.com/uc?id=0B7XkCwpI5KDYNlNUTTlSS21pQmM&export=download
(1.5GB).


## Usage

Show list of tasks
    > make

Run tests
    > make test

Run all tasks
    > make all
    
Train the models

    > make train-ngram
    > make train-rnn

Test the models

    > make test-ngram
    > make test-rnn


## License

GPL
