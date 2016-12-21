
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
	@echo "  make data           do the following:"
	@echo "    make download     download word2vec word vectors (1.5gb)"
	@echo "    make unzip        unzip word2vec vectors"
	@echo "    make split        split texts into train, refine, test sets"
	@echo ""
	@echo "  make test           do the following:"
	@echo "    make test-split   test the splitter"
	@echo "    make test-ngram   test the ngram model"
	@echo "    make test-rnn     test the rnn model"
	@echo ""
	@echo "  make model          "

# --------------------------------------------------------------------------------
# * Data
# --------------------------------------------------------------------------------
data: download unzip split

# Gutenberg text files
# gutenbergs = 325 135 28885 120 209 8486 13969 289 8164 20387
# split: $(foreach gnum,$(gutenbergs),data/train/$(gnum)-train.txt)

# Word vectors
# word_vectors_url     = https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing
# word_vectors_zipfile = data/raw/GoogleNews-vectors-negative300.bin.gz
# word_vectors_file    = data/raw/GoogleNews-vectors-negative300.bin

# test download and unzip with a small gutenberg file
word_vectors_url     = http://www.gutenberg.org/files/5616/5616.zip
word_vectors_folder  = _scratch
word_vectors_zipfile = $(word_vectors_folder)/5616.zip
word_vectors_file    = $(word_vectors_folder)/5616.txt

download: $(word_vectors_zipfile)
unzip: $(word_vectors_file)

$(word_vectors_zipfile):
	wget -O $(word_vectors_zipfile) $(word_vectors_url)
	touch $(word_vectors_zipfile)

$(word_vectors_file): $(word_vectors_zipfile)
	unzip -d $(word_vectors_folder) $(word_vectors_zipfile)
	touch $(word_vectors_file)



split: data/split/train.txt

data/split/train.txt: data/raw/all.txt
	python src/split.py

data/raw/all.gz: _scratch/foo.zip
	gunzip -k _scratch/foo.zip
	mv _scratch/5616.txt _scratch/foo.txt



# download: _scratch/foo.txt

# _scratch/foo.txt: _scratch/foo.zip
# 	cd _scratch
# 	unzip foo.zip
# 	mv 5616.txt foo.txt
# 	cd ..

# _scratch/foo.zip:
# 	wget -O _scratch/foo.zip http://www.gutenberg.org/files/5616/5616.zip
# 	touch _scratch/foo.zip

# --------------------------------------------------------------------------------
# * Test
# --------------------------------------------------------------------------------
test: test-split test-ngram test-rnn

test-split:
	python test/test-split.py

test-ngram:
	python test/test-ngram.py

test-rnn:
	echo "nop"

# split:
# train
# test
# all

# --------------------------------------------------------------------------------
# * Train
# --------------------------------------------------------------------------------
train: train-ngram

train-ngram:
	python src/train.py

models/model-ngram-basic.pickle: data/split/all_train.txt
	python src/train.py


# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------


# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------

.PHONY: data download unzip split \
        test test-split test-ngram test-rnn \
        train train-ngram train-rnn

