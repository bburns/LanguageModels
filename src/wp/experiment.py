
"""
Experiments - manage experiments on different models.
Load/train/save/test/time models with given data.
"""

import os
import os.path
from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot') # nicer style

from benchmark import benchmark
import transcript
import util


class Experiment(object):
    """
    Run an experiment on a set of models across a set of parameters.
    """
    def __init__(self, name, model_specs, data, params, test_amount=1.0):
        """
        Construct an experiment.
        name        - name of experiment
        model_specs - a list of model classes and parameters to train and test
        data        - a data object (eg Data('gutenbergs'))
        params      - additional params for the models (eg {'train_amount':1000})
        test_amount - amount of test text to use (percent or nchars)
        """
        self.name = name
        self.model_specs = model_specs
        self.data = data
        self.params = params
        self.test_amount = test_amount
        # calculated values
        self.stime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.logfolder = '../../logs'
        self.caption = 'Experiment %s: %s, dataset %s' % (self.stime, self.name, self.data.name)
        self.plotfile_prefix = self.logfolder + '/' + self.caption.replace(':','')
        self.logfile = self.logfolder + "/experiments.org"
        assert(len(params)==1) # only handles one param at a time
        # self.data.prepare() # make sure the data is cleaned and split up

    def run(self, force_training=False):
        """
        Train and test the models with different parameters.
        """
        transcript.start(self.logfile)
        print('-'*80)
        print('* ' + self.caption)
        print('-'*80)
        print()
        param_name = list(self.params.keys())[0]
        param_values = self.params[param_name]
        train_losses = pd.DataFrame(columns=['Epoch'])
        train_times = []
        test_times = []
        test_scores = []
        cols = []
        rows = []
        for param_value in param_values:
            print('** Parameter:', param_name, param_value)
            print()
            # row_name = param_name + '=' + str(param_value)
            # rows.append(row_name)
            rows.append(param_value)
            train_time_row = []
            test_time_row = []
            test_score_row = []
            cols = []
            #. get data using this param value here?
            for (model_class, model_params) in self.model_specs:
                # get a dict with all parameter names and values
                # name_includes = model_params.keys() # eg ['train_amount']
                name_includes = list(model_params.keys()) # eg ['train_amount']
                params = model_params.copy()
                params[param_name] = param_value
                print(params)
                model = model_class(self.data, name_includes=name_includes, **params)
                cols.append(model.name)
                model.train(force_training) # loads/saves model as needed
                # model.train_results # dataframe with loss vs epoch etc - want to save this to another table
                tr = model.train_results[['Epoch','Loss']]
                tr = tr.rename(columns={'Loss': model.name}) # rename Loss column to name of model
                # print(tr)
                train_losses = pd.merge(train_losses, tr, how='outer', on='Epoch')
                # print(train_losses)
                # print(util.table(model.train_results)) # print losses by epoch nicely
                model.test(test_amount=self.test_amount) # will set properties, e.g. model.test_score
                # print(util.table(model.test_samples))
                print('Score', model.test_score)
                print()
                train_time_row.append(model.train_time)
                test_time_row.append(model.test_time)
                test_score_row.append(model.test_score)
            train_times.append(train_time_row)
            test_times.append(test_time_row)
            test_scores.append(test_score_row)
            print()
        # make pandas tables
        self.train_losses = train_losses
        self.train_times = pd.DataFrame(train_times, index=rows, columns=cols)
        self.test_times = pd.DataFrame(test_times, index=rows, columns=cols)
        self.test_scores = pd.DataFrame(test_scores, index=rows, columns=cols)
        # print and plot results
        self.print_results() # goes to transcript
        # self.plot()
        transcript.stop()

    def print_results(self):
        print('** Results')
        print()
        print('Train Losses')
        print(self.train_losses)
        print()
        print('Train Times')
        print(self.train_times)
        print()
        print('Test Times')
        print(self.test_times)
        print()
        print('Test Scores')
        print(self.test_scores)
        print()

    #. refactor
    def plot(self, show=False):
        param_name = list(self.params.keys())[0]
        line_styles = ['-', '--', '-.', ':']

        # plot training loss curves
        self.train_losses.plot(x='Epoch')
        plt.title('Training Losses vs Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        if show: plt.show()
        plt.savefig(self.plotfile_prefix + ' train_losses.png')
        plt.close()

        # plot test scores
        self.test_scores.plot(kind='line', style=line_styles)
        # plt.suptitle(self.caption)
        plt.title('Relevance vs ' + param_name)
        plt.xlabel(param_name)
        plt.ylabel('relevance')
        if show: plt.show()
        plt.savefig(self.plotfile_prefix + ' relevance.png')
        plt.close()

        # plot train times
        self.train_times.plot(kind='line', style=line_styles)
        # plt.suptitle(self.caption)
        plt.title('Train time vs ' + param_name)
        plt.xlabel(param_name)
        plt.ylabel('train_time (sec)')
        if show: plt.show()
        plt.savefig(self.plotfile_prefix + ' train_time.png')
        plt.close()

        # plot test times
        self.test_times.plot(kind='line', style=line_styles)
        # plt.suptitle(self.caption)
        plt.title('Test time vs ' + param_name)
        plt.xlabel(param_name)
        plt.ylabel('test_time (sec)')
        if show: plt.show()
        plt.savefig(self.plotfile_prefix + ' test_time.png')
        plt.close()





if __name__ == '__main__':

    # need to import the same way we load/save models in the notebook for pickle to work
    import sys; sys.path.append('../')
    import wp


    # # 2017-01-04 1031
    # name = 'RNN hidden layer sizes'
    # specs = [
    #     [wp.rnn_python.RnnPython, {'nhidden':10}],
    #     [wp.rnn_python.RnnPython, {'nhidden':20}],
    #     [wp.rnn_python.RnnPython, {'nhidden':50}],
    #     [wp.rnn_python.RnnPython, {'nhidden':100}],
    # ]
    # data = wp.data.Data('gutenbergs')
    # params = {'train_amount':[1000,2000,5000,10000,20000,40000,80000]}
    # exper = Experiment(name, specs, data, params, test_amount=1000)
    # exper.run()


    # # 2017-01-04 1035
    # # surprisingly, accuracy went down as vocab went up, even with 40k training chars
    # name = 'RNN vocab sizes'
    # specs = [
    #     [wp.rnn_python.RnnPython, {'nhidden':10, 'train_amount':40000}],
    #     [wp.rnn_python.RnnPython, {'nhidden':20, 'train_amount':40000}],
    #     [wp.rnn_python.RnnPython, {'nhidden':50, 'train_amount':40000}],
    #     [wp.rnn_python.RnnPython, {'nhidden':100, 'train_amount':40000}],
    # ]
    # data = wp.data.Data('gutenbergs')
    # params = {'nvocab':[100,200,500,1000,2000]}
    # exper = Experiment(name, specs, data, params, test_amount=1000)
    # exper.run()


    # # 2017-01-04 1100
    # name = "ngrams vs rnn"
    # specs = [
    #     [wp.ngram.Ngram, {'n':1}],
    #     [wp.ngram.Ngram, {'n':2}],
    #     [wp.ngram.Ngram, {'n':3}],
    #     [wp.rnn_python.RnnPython, {}],
    # ]
    # data = wp.data.Data('gutenbergs')
    # params = {'train_amount':[1000,2000,5000,10000,20000,40000,80000]}
    # exper = Experiment(name, specs, data, params, test_amount=1000)
    # exper.run()


    # # 2017-01-05 0700
    # name = "rnn n values"
    # specs = [
    #     [wp.rnn_python.RnnPython, {'train_amount':10000}],
    # ]
    # data = wp.data.Data('animals')
    # # data = wp.data.Data('gutenbergs')
    # params = {'n':[1,2,3,4,5]}
    # exper = Experiment(name, specs, data, params, test_amount=10000)
    # # exper.run()
    # exper.run(force_training=True)
    # exper.plot(True)


    # # 2017-01-16 4pm
    # name = "keras rnn n values"
    # specs = [
    #     [wp.rnn_keras.RnnKeras, {'train_amount':10000}],
    # ]
    # data = wp.data.Data('animals')
    # # data = wp.data.Data('gutenbergs')
    # params = {'n':[1,2,3,4,5]}
    # exper = Experiment(name, specs, data, params, test_amount=10000)
    # # exper.run()
    # exper.run(force_training=True)
    # # exper.plot(True)


    # # 2017-01-18 12pm
    # name = "alice1 n"
    # specs = [
    #     # [wp.rnn_keras.RnnKeras, {'train_amount':10000,'nvocab':100,'nhidden':25,'nepochs':10}],
    #     # [wp.rnn_keras.RnnKeras, {'train_amount':1.0,'nvocab':100,'nhidden':25,'nepochs':10}],
    #     [wp.rnn_keras.RnnKeras, {'train_amount':0.25,'nvocab':100,'nhidden':25,'nepochs':10}],
    #     [wp.rnn_keras.RnnKeras, {'train_amount':0.5,'nvocab':100,'nhidden':25,'nepochs':10}],
    #     [wp.rnn_keras.RnnKeras, {'train_amount':1.0,'nvocab':100,'nhidden':25,'nepochs':10}],
    # ]
    # data = wp.data.Data('alice1')
    # # params = {'n':[2,3,4,5,6]}
    # params = {'n':[3,4,5,6]}
    # exper = Experiment(name, specs, data, params)
    # exper.run()
    # # exper.run(force_training=True)
    # exper.plot(True)


    # # 2017-01-18 5pm
    # name = "train_amounts"
    # specs = [
    #     [wp.rnn_keras.RnnKeras, {'nvocab':100,'nhidden':25,'nepochs':10}],
    # ]
    # data = wp.data.Data('alice1')
    # params = {'train_amount':[0.1,0.25,0.5,1.0]}
    # # params = {'train_amount':[0.1]}
    # exper = Experiment(name, specs, data, params)
    # exper.run()
    # # exper.run(force_training=True)
    # exper.plot(True)


    # # 2017-01-18 504pm alice1
    # name = "nvocab and nhidden"
    # specs = [
    #     [wp.rnn_keras.RnnKeras, {'nvocab':50,'nhidden':12,'nepochs':20}],
    #     [wp.rnn_keras.RnnKeras, {'nvocab':100,'nhidden':25,'nepochs':20}],
    # ]
    # data = wp.data.Data('alice1')
    # params = {'train_amount':[1.0]}
    # exper = Experiment(name, specs, data, params)
    # exper.run()
    # exper.plot(True)


    # # 2017-01-18 11pm alice1
    # name = "nhidden"
    # specs = [
    #     [wp.rnn_keras.RnnKeras, {'nvocab':500,'nhidden':6,'nepochs':10}],
    #     [wp.rnn_keras.RnnKeras, {'nvocab':500,'nhidden':12,'nepochs':10}],
    #     [wp.rnn_keras.RnnKeras, {'nvocab':500,'nhidden':25,'nepochs':10}],
    # ]
    # data = wp.data.Data('alice1')
    # params = {'train_amount':[1.0]}
    # exper = Experiment(name, specs, data, params)
    # exper.run()
    # exper.plot(True)


    # # 2017-01-18 11pm gut - roughly 1+ min per percent of training data -
    # #   nepochs=2 is fine - mostly predicting unknowns though.
    # name = "train amount"
    # specs = [
    #     [wp.rnn_keras.RnnKeras, {'nvocab':500,'nhidden':25,'nepochs':5}],
    # ]
    # data = wp.data.Data('gutenbergs')
    # params = {'train_amount':[0.05,0.1]}
    # exper = Experiment(name, specs, data, params)
    # exper.run()
    # exper.plot(True)


    # # 2017-01-19 1am gut - more vocab and hidden, fewer epochs
    name = "train amount"
    specs = [
        [wp.rnn_keras.RnnKeras, {'nvocab':1000,'nhidden':50,'nepochs':3}], # kills cpu & memory!
    ]
    data = wp.data.Data('gutenbergs')
    # params = {'train_amount':[0.05,0.1]}
    params = {'train_amount':[0.05]}
    exper = Experiment(name, specs, data, params)
    exper.run()
    exper.plot(True)

