
"""
Base Model class for use by n-gram and RNN classes.
"""

import os
import os.path
import cPickle as pickle # faster version of pickle


class Model(object):
    """
    Base model for n-gram and RNN classes.
    """

    def save(self, filename=None):
        """
        Save the model to the given or default filename.
        """
        if filename is None:
            filename = self.filename()
        try:
            folder = os.path.dirname(filename)
            os.mkdir(folder)
        except:
            pass
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    def load(self, filename=None):
        """
        Load model from the given or default filename.
        """
        if filename is None:
            filename = self.filename()
        if os.path.isfile(filename):
            print("load model")
            with open(filename, 'rb') as f:
                model = pickle.load(f)
                return model
        else:
            return self





