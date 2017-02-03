
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

    h = {'val_loss': [5.0073622123681769, 4.9213166299411046, 4.8868918108423305, 4.8769821063112833, 4.8617463204574323, 4.8751997876698834, 4.8664345612158462, 4.8636721231627869, 4.8577702023186067, 4.8612525289297599], 'acc': [0.16324044974990665, 0.18173664429972991, 0.1874352738389497, 0.19103383410314542, 0.19307000516628808, 0.19434407316801178, 0.19592438503819065, 0.19672311268451956, 0.19712442462390042, 0.19760911593708858], 'val_acc': [0.18105376841780452, 0.18930803055078319, 0.19424515930184388, 0.19864228959345831, 0.19686800894854586, 0.2008794260587827, 0.20111085396898867, 0.19925943068734089, 0.2045051299853429, 0.19871943223249267], 'loss': [5.3040714398719233, 5.0340584175586907, 4.9652212234196886, 4.9287768145667217, 4.9077094437146824, 4.8950618595178579, 4.8850289214918137, 4.8801989651525872, 4.876108366390504, 4.8731554642162651]}



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



