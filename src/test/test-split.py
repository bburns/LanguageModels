
"""
"""

from __future__ import print_function, division
import random
from pprint import pprint

# this assumes we're running the test from the root folder with 'make test-split'
import sys; sys.path.append('src')
import split


# filename = "test/data/test.txt"
filename = "data/raw/other/1920 M R James A Thin Ghost and Others.txt"
print("splitting " + filename)

random.seed(0)
split.split(filename, 0.8, 0.1)









