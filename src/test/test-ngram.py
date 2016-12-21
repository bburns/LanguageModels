
"""
Test the n-gram models using a simple test dataset.
"""

from __future__ import print_function, division
import random
from pprint import pprint

# this assumes we're running the test from the root folder with 'make test'
import sys; sys.path.append('src')
import ngram


infile = 'test/data/test-train.txt'
modelfile = 'test/data/ngram-model-basic.pickle'
n = 3

print("read file")
f = open(infile, 'r')
s = f.read()
s = s.strip()
s = s.lower()
f.close()

print("train model")
model = ngram.Ngram(n)
model.train(s)

print("model:")
print(model)

print("predict following words")
words = ['a','b'] # for trigram model, we give it 2 words
word = model.predict(words)
print(words, word)

# test pickling
model.save(modelfile)
model = ngram.Ngram.load(modelfile)

words = ['.','a']
word = model.predict(words)
print(words, word)


