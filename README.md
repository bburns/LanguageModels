
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


## Libraries

>>include version numbers for all libraries

The base Python used is the Anaconda 3 distribution with Python 3.5 - the following libraries are included:

- numpy
- matplotlib
- nltk
- pandas
- tabulate
- h5py

Additional libraries installed with pip install:

- tensorflow
- keras - wrapper around TensorFlow
- textstat - calculates Coleman-Liau Index for texts (grade level readability)
- pydot - to visualize Keras models
- nbstripout - removes output from Jupyter notebooks before commiting them to the git repository - add to repo with `nbstripout --install`

Additional programs used:

- dot - to visualize Keras models

Additional data used:

- GloVe - Wikipedia 2014 + Gigaword 5 (6B tokens, 400K vocab, uncased, 50d, 100d, 200d, & 300d vectors, 822 MB download): glove.6B.zip
http://nlp.stanford.edu/projects/glove/



## Datasets

There are several datasets available for testing -

> list sizes etc

- abcd
- alphabet
- alice1
- gutenbergs

To make a new one, create a folder in the `data` folder, then a subfolder called
`1-raw`, put any text files in it, then run the following code, with appropriate
p values -

    data = Data('abcd')
    data.prepare(ptrain=0.8, pvalidate=0.1, ptest=0.1)

This will clean the texts, merge them, and split the text up by sentence and
distribute them among the different sets according to the percentages given -
the resulting files will be in the `4-split` subfolder.


## License

GPL
