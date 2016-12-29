
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
        # print("create model object")
        model = modelclass(modelfolder=modelfolder, nchars=nchars, **modelparams) # __init__ method
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


# #> move to data.py?
# def get_tuples(tokens, ntokens_per_tuple):
#     """
#     Group sequences of tokens together.
#     e.g. ['the','dog','barked',...] => [['the','dog'],['dog','barked'],...]
#     """
#     tokenlists = [tokens[i:] for i in range(ntokens_per_tuple)]
#     tuples = zip(*tokenlists)
#     return tuples


def test_models(models, data, npredictions_max=1000, k=3, nchars=None):
    """
    Test the given models on nchars of the given data's test tokens.
    returns list of accuracy scores for each model
    """

    # get the test tokens
    print('get complete stream of test tokens, nchars=%d' % nchars)
    test_tokens = data.tokens('test', nchars)
    ntokens = len(test_tokens)

    # run test on the models
    scores = []
    for model in models:

        n = model.n
        # print('group tokens into tuples, n=%d' % n)
        # test_tuples = get_tuples(test_tokens, n) # group tokens into sequences
        # i = 0
        nright = 0
        # for tuple in test_tuples:
        for i in range(ntokens-n):
            # prompt = tuple[:-1]
            # actual = tuple[-1]
            prompt = test_tokens[i:i+n-1]
            actual = test_tokens[i+n]
            tokprobs = model.predict(prompt, k)
            if tokprobs: # can be None
                predicted_tokens = [tokprob[0] for tokprob in tokprobs]
                if actual in predicted_tokens:
                    nright += 1
            # i += 1
            if i > npredictions_max: break
        npredictions = i
        accuracy = nright / npredictions
        print("%s: accuracy = nright/total = %d/%d = %f" % (model.name, nright, npredictions, accuracy))
        scores.append(accuracy)

    return scores


if __name__ == '__main__':
    pass


