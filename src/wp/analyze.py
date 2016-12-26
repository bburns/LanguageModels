
"""
Analyze models - load/train/save/test models on given data.
"""

from __future__ import print_function, division
import os
import os.path


def init_models(modelspecs, modelfolder, data, nchars=None):
    """
    Initialize models from the given model list and data, loading/training/saving as needed.
    """
    train_tokens = None
    models = []
    for (modelclass, modelparams) in modelspecs:
        print("create model object")
        model = modelclass(modelfolder=modelfolder, nchars=nchars, **modelparams) # __init__ method
        model = model.load()
        if not model.trained():
            # get sequence of training tokens if needed (slow)
            if not train_tokens:
                print("obtaining training tokens")
                train_tokens = data.tokens('train', nchars)
            print("train model")
            model.train(train_tokens)
            print("save model")
            model.save()
        models.append(model)
    return models


#> move to data.py?
def get_tuples(tokens, ntokens_per_tuple):
    """
    Group sequences of tokens together.
    e.g. ['the','dog','barked',...] => [['the','dog'],['dog','barked'],...]
    """
    tokenlists = [tokens[i:] for i in range(ntokens_per_tuple)]
    tuples = zip(*tokenlists)
    return tuples


def test_models(models, data, npredictions_max=1000, k=3, nchars=None):
    """
    Test the given models on nchars of the given data's test tokens.
    returns list of accuracy scores for each model
    """

    # get the test tokens
    test_tokens = data.tokens('test', nchars)

    # run test on the models
    scores = []
    for model in models:
        n = model.n
        test_tuples = get_tuples(test_tokens, n) # group tokens into sequences
        i = 0
        nright = 0
        for tuple in test_tuples:
            prompt = tuple[:-1]
            actual = tuple[-1]
            prediction = model.predict(prompt, k)
            if prediction: # can be None
                predicted_tokens = [pair[0] for pair in prediction]
                if actual in predicted_tokens:
                    nright += 1
            i += 1
            if i > npredictions_max: break
        npredictions = i
        accuracy = nright / npredictions
        print("%s: accuracy = nright/total = %d/%d = %f" % (model.name, nright, npredictions, accuracy))
        scores.append(accuracy)

    return scores


