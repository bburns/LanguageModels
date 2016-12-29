
"""
Analyze models - load/train/save/test models on given data.
"""

from __future__ import print_function, division
import os
import os.path

import pandas as pd


def init_model_table(model_specs, model_folder, data, nchars_list):
    """
    Initialize models
    """
    model_table = []
    # model_table = {}
    for nchars in nchars_list:
        print('ntraining_chars', nchars)
        # load/train/save model
        models = init_models(model_specs, model_folder, data, nchars=nchars) # load/train models
        models = [nchars] + models
        model_table.append(models)
        # model_table[nchars] = models
    return model_table

def init_models(model_specs, model_folder, data, nchars=None):
    """
    Initialize models from the given model list and data - load/train/save as needed.
    """
    train_tokens = None
    models = []
    for (model_class, model_params) in model_specs:
        # print("create model object")
        model = model_class(model_folder=model_folder, nchars=nchars, **model_params) # __init__ method
        model = model.load() # load model if available
        if not model.trained:
            # get sequence of training tokens if needed (slow)
            if not train_tokens:
                print("get complete stream of training tokens, nchars=%d" % nchars)
                train_tokens = data.tokens('train', nchars)
            print("train model")
            model.train(train_tokens)
            print("save model")
            model.save()
        models.append(model)
    return models


def test_model_table(model_table, data, ntest_chars=10000, npredictions_max=1000, k=3):
    """
    Test all models, returning results in a pandas dataframe.
    """
    # cols = ['nchars'] + [model.name for model in models]
    models = model_table[0]
    cols = ['nchars'] + [model.name for model in models[1:]]
    table = []
    for models in model_table:
        ntrain_chars = models[0]
        # scores = test_models(models, data, ntest_chars, npredictions_max, k)
        scores = test_models(models[1:], data, ntest_chars, npredictions_max, k)
        # print()
        row = [ntrain_chars] + scores
        table.append(row)
    # return as a transposed pandas df
    df = pd.DataFrame(table, columns=cols)
    # df = pd.DataFrame(table)
    df = df.transpose()
    nchars_list = [models[0] for models in model_table]
    df.columns = nchars_list
    df = df.drop('nchars',axis=0)
    return df

def test_models(models, data, ntest_chars=None, npredictions_max=1000, k=3):
    """
    Test the given models on nchars of the given data's test tokens.
    Returns list of accuracy scores for each model
    """
    # get the test tokens
    print('get complete stream of test tokens, nchars=%d' % ntest_chars)
    test_tokens = data.tokens('test', ntest_chars)
    ntokens = len(test_tokens)
    # run test on the models
    scores = []
    for model in models:
        n = model.n
        nright = 0
        for i in range(ntokens-n):
            prompt = test_tokens[i:i+n-1]
            actual = test_tokens[i+n]
            tokprobs = model.predict(prompt, k) # eg [('barked',0.031),('slept',0.025)...]
            if tokprobs: # can be None
                predicted_tokens = [tokprob[0] for tokprob in tokprobs]
                if actual in predicted_tokens:
                    nright += 1
            if i > npredictions_max: break
        npredictions = i
        accuracy = nright / npredictions
        print("%s: accuracy = nright/total = %d/%d = %f" % (model.name, nright, npredictions, accuracy))
        scores.append(accuracy)
    return scores


if __name__ == '__main__':
    pass


