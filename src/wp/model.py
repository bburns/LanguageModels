
"""
Base Model class for use by n-gram and RNN classes.
"""

from __future__ import print_function, division
import os
import os.path
import cPickle as pickle # faster version of pickle

from benchmark import benchmark
import util


class Model(object):
    """
    Base model for n-gram and RNN classes.
    """

    def save(self):
        """
        Save the model to the default filename.
        """
        #. time this? but can't save it with the object
        folder = os.path.dirname(self.filename)
        util.mkdir(folder)
        with benchmark("Save model " + self.name) as b:
            with open(self.filename, 'wb') as f:
                # pickle.dump(self, f)
                pickle.dump(self.__dict__, f, 2)
        # self.save_time = b.time
        # return self.save_time

    def load(self):
        """
        Load model from the default filename.
        """
        if os.path.isfile(self.filename):
            with benchmark("Load model " + self.name) as b:
                with open(self.filename, 'rb') as f:
                    d = pickle.load(f)
                    self.__dict__.update(d)
            self.load_time = b.time
        else:
            self.load_time = None
        return self.load_time



