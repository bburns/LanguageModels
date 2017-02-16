
# Word Prediction using Recurrent Neural Networks

Udacity Machine Learning Nanodegree Capstone Project


## About

Word prediction, or *language modeling*, is the task of predicting the most
likely words following the preceding text. It has many applications, such as
suggesting the next word as text is entered, as an aid in resolving ambiguity in
speech and handwriting recognition, and in machine translation.


## Goals

* Implement a recurrent neural network (RNN) language model
* Implement an n-gram predictor as a baseline model
* Compare accuracy of the two methods for different training set sizes
* Compare time and space complexity and usage of the two algorithms
* Generate some text using both models for qualitative comparison


## Report

The final report is available [here](docs/report/report.pdf).


## Usage

The project will work on a default Amazon EC2 AMI distribution if you'd like a
clean slate - the base Python used is the [Anaconda3 4.2.0 distribution](https://www.continuum.io/downloads)
with Python 3.5.2.

* Select the micro instance (free tier) and launch with a key pair.
* Login with username ec2-user
* Download Anaconda3 - `wget https://repo.continuum.io/archive/Anaconda3-4.2.0-Linux-x86_64.sh`
* Make executable - `chmod +x Anaconda3-4.2.0-Linux-x86_64.sh`
* Run installer - `./Anaconda3-4.2.0-Linux-x86_64.sh`
* Add path - `source .bashrc`
* Test it - `ipython`
* Install git - `sudo yum install git`

Once you have a Python 3.5 system set up,

* Clone this repository - `git clone https://github.com/bburns/LanguageModels`
* CD into the project - `cd LanguageModels`
* Install Python packages - `pip install -r requirements.txt`
* Run `make download` to download and unzip the GloVe word vectors (~5 minutes)
* Run `make rnn` or `make ngram` to train and test the different models (the RNN
  is set up to train on 1% of the data for evaluation purposes)
* Run `make plots` to show a plot of accuracy vs train amount for ngram and RNN



## Additional data used

Automatically downloaded by `make download` -

* [GloVe](http://nlp.stanford.edu/projects/glove/ ) - Wikipedia 2014 + Gigaword
5 (6B tokens, 400K vocab, uncased, 50d, 100d, 200d, & 300d vectors, 822 MB
download): [glove.6B.zip](http://nlp.stanford.edu/data/wordvecs/glove.6B.zip)

