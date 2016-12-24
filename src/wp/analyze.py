
"""
Analyze models
"""

from __future__ import print_function, division
import os
import os.path




#. might need to use an alist instead of dict, to preserve order
def encode_params(params):
    """
    Encode a dictionary of parameters as a string to be stored in a filename.
    e.g. {'n':3,'b':1.2} => 'n-3,b-1.2'
    """
    s = str(params)
    s = s.replace(':','-')
    s = s.replace("'",'')
    s = s.replace('{','')
    s = s.replace('}','')
    s = s.replace(' ','')
    s = '(' + s + ')'
    return s


def train_models(model_list, data, nchars=None):
    """
    Train models on the training tokens, or load them if they're saved in files.
    """

    # get sequence of training tokens (slow)
    train_tokens = data.tokens('train', nchars)

    # iterate over models
    models = []
    for modelclass, modelparams in model_list:

        # load existing model, or create, train, and save one
        params = modelparams.copy()
        params['nchars'] = nchars
        sparams = encode_params(params)
        modelfile = 'models/' + modelclass.__name__ + '-' + sparams + '.pickle'
        if os.path.isfile(modelfile):
            print("load model: " + modelfile)
            model = modelclass.load(modelfile) # static method
        else:
            print("create model object")
            model = modelclass(**modelparams)

            print("train model")
            model.train(train_tokens)

            print("save model: " + modelfile)
            model.save(modelfile)

        models.append(model)

    print("done")
    return models


#> move to data.py?
def get_tuples(tokens, ntokens_per_tuple):
    """
    Group sequences of tokens together.
    e.g. ['the','dog','barked',...]=>[['the','dog'],['dog','barked'],...]
    """
    tokenlists = [tokens[i:] for i in range(ntokens_per_tuple)]
    tuples = zip(*tokenlists)
    return tuples


def test_models(models, data, nchars=None):
    """
    """

    # get the test tokens
    test_tokens = data.tokens('test', nchars)

    # run test on the models
    npredictions = 1000
    k = 3 # number of tokens to predict
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
            if i>npredictions: break
        print("nright/total=%d/%d = %f" % (nright, npredictions, nright/npredictions))
        print()


