
# Word Prediction using Recurrent Neural Networks

Udacity Machine Learning Nanodegree Capstone Project


## About

Word prediction, or *language modeling*, is the task of predicting the most
likely words following the preceding text. It has many applications, such as
suggesting the next word as text is entered, as an aid in resolving ambiguity in
speech and handwriting recognition, and in machine translation.


## Goals

* Implement a recurrent neural network (RNN) to try to improve on a baseline model
* Implement an n-gram predictor as a baseline model
* Compare accuracy of the two methods for different training set sizes
* Compare time and space complexity and usage of the two algorithms
* Generate some text using both models for qualitative comparison


## Usage

* Clone this repository - `git clone http://github.com/bburns/LanguageModels`
* Run `make download` to download and unzip word vectors
* Run `make rnn` or `make ngram` to train and test the different models
* Run `make wordplot` to show plot of sample word embeddings


## Libraries

<!-- include version numbers for all libraries -->

The base Python used is the [Anaconda3 4.2 distribution](https://www.continuum.io/downloads) with Python 3.5 - the following libraries are included:

- numpy
- matplotlib
- nltk
- pandas
- tabulate

The NLTK Punkt tokenizer may need to be installed, in which case:

    $ python
    >>> import nltk
    >>> nltk.download('punkt')

Additional libraries may need to be installed with pip install:

- keras - wrapper around TensorFlow
- tensorflow

<!-- - textstat - calculates Coleman-Liau Index for texts (grade level readability) -->
<!-- - pydot - to visualize Keras models -->

Additional programs used:

<!-- - dot - to visualize Keras models -->
- nbstripout - removes output from Jupyter notebooks before commiting them to
  the git repository - pip install, then add to repo with `nbstripout --install`

Additional data used:

- [GloVe](http://nlp.stanford.edu/projects/glove/ ) - Wikipedia 2014 + Gigaword
5 (6B tokens, 400K vocab, uncased, 50d, 100d, 200d, & 300d vectors, 822 MB
download): [glove.6B.zip](http://nlp.stanford.edu/data/wordvecs/glove.6B.zip)

