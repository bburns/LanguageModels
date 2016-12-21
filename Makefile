
# Makefile for word prediction


# default task - comment out for release version
all: test

# --------------------------------------------------------------------------------
# * Help
# --------------------------------------------------------------------------------
help:
	@echo "Makefile for word prediction"
	@echo ""
	@echo "Usage:"
	@echo "  make data         do the following:"
	@echo "  make download     download word2vec word vectors (1.5gb)"
	@echo "  make unzip        unzip word2vec vectors"
	@echo "  make split        split texts into train, refine, test sets"
	@echo ""
	@echo "  make test         do the following:"
	@echo "  make test-split   test the splitter"
	@echo "  make test-ngram   test the ngram model"
	@echo "  make test-rnn     test the rnn model"
	@echo ""
	@echo "  make model          "

# --------------------------------------------------------------------------------
# * Data
# --------------------------------------------------------------------------------
data: download unzip split

# word vectors
word_vectors_folder  = data/raw
word_vectors_url     = https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing
word_vectors_zipfile = $(word_vectors_folder)/GoogleNews-vectors-negative300.bin.gz
word_vectors_file    = $(word_vectors_folder)/GoogleNews-vectors-negative300.bin

unzip: $(word_vectors_file)
$(word_vectors_file): $(word_vectors_zipfile)
	gzip -k -d $(word_vectors_zipfile)
	touch $(word_vectors_file)

download: $(word_vectors_zipfile)
$(word_vectors_zipfile):
	wget -O $(word_vectors_zipfile) $(word_vectors_url)
	touch $(word_vectors_zipfile)

split:
	python src/split.py --ptrain 0.8 --pvalidate 0.1 --ptest 0.1 data/raw/all.txt data/split



# --------------------------------------------------------------------------------
# * Test
# --------------------------------------------------------------------------------
test: test-split test-ngram test-rnn

test-split:
	python src/test/test-split.py

test-ngram:
	python src/test/test-ngram.py

test-rnn:
	python src/test/test-rnn.py

# --------------------------------------------------------------------------------
# * Train
# --------------------------------------------------------------------------------
train: train-ngram

train-ngram:
	python src/train.py

models/model-ngram-basic.pickle: data/split/all-train.txt
	python src/train.py


# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------


# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------

.PHONY: data download unzip split \
        test test-split test-ngram test-rnn \
        train train-ngram train-rnn

