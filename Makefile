
# Makefile for word prediction


# default task - comment out for release version
all: test


# Gutenberg texts to download
gutenbergs = 325 135 28885 120 209 8486 13969 289 8164 20387

# Word vectors to download
# word_vectors=

help:
	@echo "Makefile for word prediction"
	@echo ""
	@echo "Usage:"
	@echo "  make data           get texts, preprocess, split, get vectors etc"
	@echo "  make get-texts      download gutenberg texts"
	@echo "  make preprocess     preprocess texts"
	@echo "  make split          split texts into train, refine, test sets"
	@echo "  make get-vectors    download word2vec vectors (1.5gb)"
	@echo "  make unzip-vectors  unzip word2vec vectors"
	@echo ""
	@echo "  make test           test the ngram and rnn models"

test:
	python test/test.py

# data: get-texts preprocess split get-vectors unzip-vectors


# can't do this -
# Are you human? You have used Project Gutenberg quite a lot today or clicked
# through it really fast. To make sure you are human, we ask you to resolve this
# captcha.

get-texts: $(foreach gnum,$(gutenbergs),data/raw/$(gnum)-0.txt)
# get-texts: data/raw/325-0.txt data/raw/135-0.txt data/raw/28885-0.txt data/raw/120-0.txt

# note: % is a string matching operator, $* inserts the match
# creates rules like the following:
# data/raw/325-0.txt:
# 	wget -O data/raw/325-0.txt http://www.gutenberg.org/files/325/325-0.txt
# 	touch data/raw/325-0.txt
data/raw/%-0.txt:
	wget -O data/raw/$*-0.txt http://www.gutenberg.org/files/$*/$*-0.txt
	touch data/raw/$*-0.txt

# %.txt:
# 	wget -O data/raw/%.txt $(gutenbergs).txt
# 	http://www.gutenberg.org/files/325/325-0.txt


# split:
# unzip: foo.gz
# foo.gz:
# train
# test
# all


.PHONY: test

