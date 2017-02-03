
# Makefile for RNN word prediction project
# Note: can use make -B <target> to force building a target


# --------------------------------------------------------------------------------
# * Help
# --------------------------------------------------------------------------------
help:
	@echo "Makefile for RNN word prediction project"
	@echo ""
	@echo "Usage:"
	@echo "  make download     download and unzip GloVe word vectors (~1gb)"
	@echo "  make ngram        run ngram.py"
	@echo "  make rnn          run rnn.py"
	@echo "  make plots        run plots.py"
	@echo "  make report       make and view pdf of report - doc/report/report.pdf"

# --------------------------------------------------------------------------------
# * Download
# --------------------------------------------------------------------------------

MKDIRP = mkdir -p

# word vectors
# word2vec_url     = https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing
# word2vec_zipfile = $(word2vec_folder)/GoogleNews-vectors-negative300.bin.gz
# word2vec_file    = $(word2vec_folder)/GoogleNews-vectors-negative300.bin
# unzip: $(word2vec_file)
# $(word2vec_file): $(word2vec_zipfile)
# 	gzip -k -d $(word2vec_zipfile)
# 	touch $(word2vec_file)
# download: $(word2vec_zipfile)
# $(word2vec_zipfile):
# 	wget -O $(word2vec_zipfile) $(word2vec_url)
# 	touch $(word2vec_zipfile)

glove_folder  = _vectors/glove.6B
glove_url     = http://nlp.stanford.edu/data/wordvecs/glove.6B.zip
glove_zipfile = $(glove_folder)/glove.6B.zip
glove_file    = $(glove_folder)/glove.6B.50d.txt

download: unzip get

unzip: $(glove_file)
$(glove_file): $(glove_zipfile)
	unzip $(glove_zipfile) -d $(glove_folder)
	touch $(glove_file)

get: $(glove_zipfile)
$(glove_zipfile):
	$(MKDIRP) $(glove_folder)
	wget -O $(glove_zipfile) $(glove_url)
	touch $(glove_zipfile)


# --------------------------------------------------------------------------------
# * Run
# --------------------------------------------------------------------------------

ngram:
	ipython -i src/ngram.py

rnn:
	ipython -i src/rnn.py

plots:
	ipython -i src/plots.py


# --------------------------------------------------------------------------------
# * Report
# --------------------------------------------------------------------------------

report: doc/report/report.pdf

doc/report/report.pdf: doc/report/report.md
	cd doc/report && pandoc report.md -o report.pdf && start report.pdf


# --------------------------------------------------------------------------------
# * End
# --------------------------------------------------------------------------------

.PHONY: help download ngram rnn plots report
