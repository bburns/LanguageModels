
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
output_folder = "src/test/data" # same folder
split.split(filename, output_folder)

# should have created 3 files in same dir: test-train.txt, -validate, -test








