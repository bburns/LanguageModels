
"""
Plots
"""



print('Importing libraries (~10sec)...')

# import sys
# print(sys.version)
# import os
# import os.path
# import random
# import re
# import heapq

# import numpy as np
# import pandas as pd

# from nltk import tokenize

# #from keras.preprocessing.text import Tokenizer
# from keras.utils.np_utils import to_categorical
# from keras.layers import Dense, Activation, Dropout
# from keras.models import Model
# from keras.models import Sequential
# #from keras.models import load_model
# from keras.layers.embeddings import Embedding
# from keras.layers.recurrent import SimpleRNN, LSTM, GRU
# from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
# from keras.metrics import top_k_categorical_accuracy

import matplotlib.pyplot as plt
import seaborn as sns

# # local modules
# # import sys; sys.path.append('../src')
# import data as datamodule
# import util



# --------------------------------------------------------------------------------
# Plot Results
# --------------------------------------------------------------------------------
if 1:

    h = {'acc': [0.15175669431175184,
  0.16687797233818058,
  0.17081706331599847,
  0.17310337250074301,
  0.17452861432039662,
  0.17543799496764942,
  0.17637386999521526,
  0.17715155798647567,
  0.17769936826489846,
  0.17837029948767355],
 'loss': [5.4186651027795145,
  5.1610167792829182,
  5.1034065738217853,
  5.0783216575504202,
  5.0669597474576991,
  5.0567085283876736,
  5.0473167150160094,
  5.0400304591740701,
  5.0317346811268733,
  5.0271773664752777],
 'val_acc': [0.16647381007482837,
  0.17341664738100748,
  0.17796806294839157,
  0.17781377767721995,
  0.1772737792200727,
  0.1782766334976319,
  0.1772737792200727,
  0.18205662269536374,
  0.18460232970992843,
  0.18437090179972246],
 'val_loss': [5.1304264735138219,
  5.0385535729296516,
  5.0138801407558375,
  5.0090141165433986,
  5.0079125920480889,
  5.002276728773599,
  4.9905472688263677,
  4.9856565054171265,
  4.9789387782874561,
  4.978088331362315]}


    # plot loss vs epoch
    plt.plot(h['loss'], label='Training')
    plt.plot(h['val_loss'], label='Validation')
    plt.xlabel('epoch-1')
    plt.ylabel('loss')
    plt.title("Training and Validation Loss vs Epoch")
    plt.legend()
    plt.show()

    # plot accuracy vs epoch
    plt.plot(h['acc'], label='Training')
    plt.plot(h['val_acc'], label='Validation')
    plt.xlabel('epoch-1')
    plt.ylabel('accuracy')
    plt.title("Training and Validation Accuracy vs Epoch")
    plt.legend()
    plt.show()


# --------------------------------------------------------------------------------
# Visualize Embeddings
# --------------------------------------------------------------------------------
if 0:

    from sklearn.decomposition import PCA

    words = 'alice rabbit mouse said was fell small white gray'.split()
    print('words',words)
    iwords = [data.word_to_iword[word] for word in words]
    print('iwords',iwords)
    vecs = [E[iword] for iword in iwords]
    print('word embedding for alice',vecs[1])

    # now want to reduce dims of these vectors
    pca = PCA(n_components=2)
    pca.fit(vecs)
    vecnew = pca.transform(vecs)
    print('some projections',vecnew[:3])

    # now plot the new vectors with labels
    x = [vec[0] for vec in vecnew]
    y = [vec[1] for vec in vecnew]
    plt.scatter(x, y)

    for i, word in enumerate(words):
        plt.annotate(word, (x[i]+0.1,y[i]+0.1))

    plt.title("Word embeddings projected to 2D")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.show()



