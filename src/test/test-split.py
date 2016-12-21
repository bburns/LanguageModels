
"""
Test split.py
"""

from __future__ import print_function, division
import random
from pprint import pprint

# this assumes we're running the test from the root folder with 'make test-split'
import sys; sys.path.append('src')
import split


random.seed(0)
filename = "src/test/data/test.txt"
split.split(filename)

#. now test split files size and contents









