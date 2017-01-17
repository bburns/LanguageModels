
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


## License

GPL
