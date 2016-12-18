
# Makefile for word prediction

# Gutenberg texts to download
# gutenbergs:=123,234

# Word vectors to download
# word_vectors:=


help:
	@echo "Makefile for word prediction"
	@echo ""
	@echo "Usage:"
	@echo "  make data           get texts, preprocess, split, get vectors etc"
	@echo "  make get-texts      download gutenberg texts"
	@echo "  make preprocess     preprocess texts"
	@echo "  make split          split texts into "
	@echo "  make get-vectors    download word2vec vectors (1.5gb)"
	@echo "  make unzip-vectors  unzip word2vec vectors"
	@echo ""
	@echo "  make test           test the ngram and rnn models"


test:
	@echo "Hello!"

# data: get-texts preprocess split get-vectors unzip-vectors

# get-texts: data/raw/%.txt

# %.txt:
# 	wget $(gutenbergs).txt

# split:

# unzip: foo.gz
# foo.gz:


# train
# test
# all



.PHONY: test

