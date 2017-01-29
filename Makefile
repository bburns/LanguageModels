
# Makefile for word prediction project
# Use make -B <target> to force building a target


# default task - comment out for release version
all: test

# --------------------------------------------------------------------------------
# * Help
# --------------------------------------------------------------------------------
help:
	@echo "Makefile for word prediction project"
	@echo ""
	@echo "Usage:"
	@echo "  make data         do the following:"
	@echo "  make download     download word2vec word vectors (1.5gb)"
	@echo "  make unzip        unzip word2vec vectors"
	@echo "  make split        split texts into train, refine, test sets"
	@echo ""
	@echo "  make train        "
	@echo "  make validate     "
	@echo "  make test         "
	@echo ""
	@echo "  make report       make pdf of report - doc/report/report.pdf"
	@echo ""
	@echo "Unit tests:"
	@echo "  make test-split   test the splitter"
	@echo "  make test-ngram   test the ngram model"
	@echo "  make test-rnn     test the rnn model"

# --------------------------------------------------------------------------------
# * Report
# --------------------------------------------------------------------------------

report: doc/report/report.pdf
doc/report/report.pdf: doc/report/report.md
	cd doc/report && pandoc report.md -o report.pdf && start report.pdf


# --------------------------------------------------------------------------------
# * Notebook server
# --------------------------------------------------------------------------------
server:
#	start \"foo\" /MAX dir
#	start \"foo\" dir
	start jupyter notebook



# --------------------------------------------------------------------------------
# * Data
# --------------------------------------------------------------------------------

# download, unzip, and prepare data files for training, validating, and testing


data: folders download unzip split

# word vectors
# word_vectors_folder  = _vectors
# word_vectors_url     = https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing
# word_vectors_zipfile = $(word_vectors_folder)/GoogleNews-vectors-negative300.bin.gz
# word_vectors_file    = $(word_vectors_folder)/GoogleNews-vectors-negative300.bin
# unzip: $(word_vectors_file)
# $(word_vectors_file): $(word_vectors_zipfile)
# 	gzip -k -d $(word_vectors_zipfile)
# 	touch $(word_vectors_file)
# download: $(word_vectors_zipfile)
# $(word_vectors_zipfile):
# 	wget -O $(word_vectors_zipfile) $(word_vectors_url)
# 	touch $(word_vectors_zipfile)

glove_vectors_folder  = _vectors/glove.6B
glove_vectors_url     = http://nlp.stanford.edu/data/wordvecs/glove.6B.zip
glove_vectors_zipfile = $(glove_vectors_folder)/glove.6B.zip
glove_vectors_file    = $(glove_vectors_folder)/glove.6B.50d.txt

MKDIR_P = mkdir -p
.PHONY: folders
# all: folders
folders: $(glove_vectors_folder)
$(glove_vectors_folder):
	$(MKDIR_P) $(glove_vectors_folder)

#	gzip -k -d $(glove_vectors_zipfile)
unzip: $(glove_vectors_file)
$(glove_vectors_file): $(glove_vectors_zipfile)
	unzip $(glove_vectors_zipfile) -d $(glove_vectors_folder)
	touch $(glove_vectors_file)

download: $(glove_vectors_zipfile)
$(glove_vectors_zipfile):
	wget -O $(glove_vectors_zipfile) $(glove_vectors_url)
	touch $(glove_vectors_zipfile)

# split:
# 	python src/split.py --ptrain 0.8 --pvalidate 0.1 --ptest 0.1 data/processed/all.txt data/split



# --------------------------------------------------------------------------------
# * Train
# --------------------------------------------------------------------------------

train: train-ngram

train-ngram: data/models/model-ngram-basic.pickle
data/models/model-ngram-basic.pickle: data/split/all-train.txt
	python src/train-ngram.py


# --------------------------------------------------------------------------------
# * Test
# --------------------------------------------------------------------------------

# test: test-ngram

# test-ngram: data/models/model-ngram-basic.pickle
# data/models/model-ngram-basic.pickle: data/split/all-test.txt
# 	python src/test-ngram.py

# --------------------------------------------------------------------------------
# * Unit Tests
# --------------------------------------------------------------------------------

# test: test-split test-ngram test-rnn

test-split:
	python src/test/test-split.py

test-ngram:
	python src/test/test-ngram.py

test-rnn:
	python src/test/test-rnn.py

# --------------------------------------------------------------------------------
# * other
# --------------------------------------------------------------------------------


# --------------------------------------------------------------------------------
# * End
# --------------------------------------------------------------------------------

.PHONY: data download unzip split \
        train train-ngram train-rnn \
        test test-ngram test-rnn \
        test-split test-ngram test-rnn
