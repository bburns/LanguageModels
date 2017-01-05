
"""
Experiments - manage experiments on different models.
Load/train/save/test/time models with given data.
"""

from __future__ import print_function, division
import os
import os.path
from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt

from benchmark import benchmark
import transcript


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
        self.time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.logfolder = '../../logs'
        self.caption = 'Experiment %s: %s' % (self.time, self.name)
        self.plotfile_prefix = self.logfolder + '/' + self.caption.replace(':','')
        self.model_specs = model_specs
        self.data = data
        self.params = params
        self.test_amount = test_amount
        self.logfile = self.logfolder + "/experiments.org"
        assert(len(params)==1) # only handles one param at a time
        # self.data.prepare() # make sure the data is cleaned and split up

    def run(self):
        """
        Train and test the models with different parameters.
        """
        transcript.start(self.logfile)
        print('-'*80)
        print('* ' + self.caption)
        print('-'*80)
        print()
        param_name = self.params.keys()[0]
        param_values = self.params[param_name]
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
                name_includes = model_params.keys() # eg ['train_amount']
                params = model_params.copy()
                params[param_name] = param_value
                print(params)
                model = model_class(self.data, name_includes=name_includes, **params)
                cols.append(model.name)
                model.train() # loads/saves model as needed
                model.test(test_amount=self.test_amount) #. should this return score, or access it as below?
                print('Score', model.test_score)
                train_time_row.append(model.train_time)
                test_time_row.append(model.test_time)
                test_score_row.append(model.test_score)
                print()
            train_times.append(train_time_row)
            test_times.append(test_time_row)
            test_scores.append(test_score_row)
            print()
        # make pandas tables
        self.train_times = pd.DataFrame(train_times, index=rows, columns=cols)
        self.test_times = pd.DataFrame(test_times, index=rows, columns=cols)
        self.test_scores = pd.DataFrame(test_scores, index=rows, columns=cols)
        # print and plot results
        self.print_results()
        self.plot_results()
        transcript.stop()

    def print_results(self):
        print('** Results')
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

    def plot_results(self):

        param_name = self.params.keys()[0]
        plt.style.use('ggplot') # nicer style
        line_styles = ['-', '--', '-.', ':']

        self.test_scores.plot(kind='line', style=line_styles)
        # plt.suptitle(self.caption)
        plt.title('Accuracy vs ' + param_name)
        plt.xlabel(param_name)
        plt.ylabel('accuracy')
        # plt.show()
        plt.savefig(self.plotfile_prefix + ' accuracy.png')
        plt.close()

        self.train_times.plot(kind='line', style=line_styles)
        # plt.suptitle(self.caption)
        plt.title('Train time vs ' + param_name)
        plt.xlabel(param_name)
        plt.ylabel('train_time (sec)')
        # plt.show()
        plt.savefig(self.plotfile_prefix + ' train_time.png')
        plt.close()

        self.test_times.plot(kind='line', style=line_styles)
        # plt.suptitle(self.caption)
        plt.title('Test time vs ' + param_name)
        plt.xlabel(param_name)
        plt.ylabel('test_time (sec)')
        # plt.show()
        plt.savefig(self.plotfile_prefix + ' test_time.png')
        plt.close()



if __name__ == '__main__':

    # need to import the same way we load/save models in the notebook for pickle to work
    import sys; sys.path.append('../')
    import wp

    # # 2017-01-04 1031
    # name = 'RNN hidden layer sizes'
    # specs = [
    #     [wp.rnn.Rnn, {'nhidden':10}],
    #     [wp.rnn.Rnn, {'nhidden':20}],
    #     [wp.rnn.Rnn, {'nhidden':50}],
    #     [wp.rnn.Rnn, {'nhidden':100}],
    # ]
    # data = wp.data.Data('gutenbergs')
    # params = {'train_amount':[1000,2000,5000,10000,20000,40000,80000]}
    # exper = Experiment(name, specs, data, params, test_amount=1000)
    # exper.run()

    # # 2017-01-04 1035
    # # surprisingly, accuracy went down as vocab went up, even with 40k training chars
    # name = 'RNN vocab sizes'
    # specs = [
    #     [wp.rnn.Rnn, {'nhidden':10, 'train_amount':40000}],
    #     [wp.rnn.Rnn, {'nhidden':20, 'train_amount':40000}],
    #     [wp.rnn.Rnn, {'nhidden':50, 'train_amount':40000}],
    #     [wp.rnn.Rnn, {'nhidden':100, 'train_amount':40000}],
    # ]
    # data = wp.data.Data('gutenbergs')
    # params = {'nvocab':[100,200,500,1000,2000]}
    # exper = Experiment(name, specs, data, params, test_amount=1000)
    # exper.run()

    # 2017-01-04 1100
    name = "ngrams vs rnn"
    specs = [
        [wp.ngram.Ngram, {'n':1}],
        [wp.ngram.Ngram, {'n':2}],
        [wp.ngram.Ngram, {'n':3}],
        [wp.rnn.Rnn, {}],
    ]
    data = wp.data.Data('gutenbergs')
    params = {'train_amount':[1000,2000,5000,10000,20000,40000,80000]}
    exper = Experiment(name, specs, data, params, test_amount=1000)
    exper.run()

