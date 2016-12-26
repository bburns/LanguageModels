
"""
Analyze models
"""

from __future__ import print_function, division
import os
import os.path


modelfolder_default = '../../data/models'


# def create_models(model_list, nchars=None):
def create_models(model_list, nchars=None, modelfolder=None):
    """
    Create models from the given model list with class and params.
    """
    models = []
    for modelclass, modelparams in model_list:
        # params = modelparams.copy()
        # params['nchars'] = nchars
        # sparams = encode_params(params)
        # modelfile = 'models/' + modelclass.__name__ + '-' + sparams + '.pickle'
        # if os.path.isfile(modelfile):
        #     print("load model: " + modelfile)
        #     model = modelclass.load(modelfile) # static method
        # else:
        print("create model object")
        if modelfolder is None:
            modelfolder = modelfolder_default
        # model = modelclass(**modelparams) # __init__ method
        # model = modelclass(nchars=nchars, **modelparams) # __init__ method
        model = modelclass(nchars=nchars, modelfolder=modelfolder, **modelparams) # __init__ method
        # model = modelclass(**params) # __init__ method
        models.append(model)
    return models


def load_models(models):
    """
    Load models from file if available.
    """
    models2 = []
    for model in models:
        # params['nchars'] = nchars
        # sparams = encode_params(params)
        # modelfile = 'models/' + modelclass.__name__ + '-' + sparams + '.pickle'
        # if os.path.isfile(model.filename):
        #     print("load model: " + model.filename)
        #     modelclass = type(model)
        #     model = modelclass.load(model.filename) # static method
        model = model.load()
        models2.append(model)
    return models2


# def train_models(model_list, data, nchars=None):
# def train_models(models, data):
# def train_empty_models(models, data):
def train_empty_models(models, data, nchars=None):
    """
    # Train models on the training tokens, or load them if they're saved in files.
    Train empty models on the training tokens.
    """

    # get sequence of training tokens (slow)
    train_tokens = data.tokens('train', nchars)

    # iterate over models
    # models = []
    # for modelclass, modelparams in model_list:
    models2 = []
    for model in models:

        # load existing model, or create, train, and save one
        # params = modelparams.copy()
        # params['nchars'] = nchars
        # sparams = encode_params(params)
        # modelfile = 'models/' + modelclass.__name__ + '-' + sparams + '.pickle'
        # if os.path.isfile(modelfile):
            # print("load model: " + modelfile)
            # model = modelclass.load(modelfile) # static method
        # else:
        if not model.trained():
            # print("create model object")
            # model = modelclass(**modelparams)

            print("train model")
            model.train(train_tokens)

            # print("save model: " + modelfile)
            # model.save(modelfile)
            print("save model")
            model.save()

        models2.append(model)

    print("done")
    return models2


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
    # npredictions = 1000
    # k = 3 # number of tokens to predict
    scores = []
    for model in models:
        print(model.name)
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
        print("accuracy = nright/total = %d/%d = %f" % (nright, npredictions, accuracy))
        print()
        scores.append(accuracy)

    return scores


