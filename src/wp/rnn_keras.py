
"""
RNN implemented with Keras
"""

import numpy as np
import keras


class RnnKeras():
    """
    """
    pass




if __name__=='__main__':

    from keras.models import Sequential
    from keras.layers import Dense, Activation

    # create a model of sequential layers
    model = Sequential()

    # add layers
    model.add(Dense(output_dim=5, input_dim=5))
    model.add(Activation("relu"))
    model.add(Dense(output_dim=5))
    model.add(Activation("softmax"))

    # configure learning process
    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

    # define some data
    X_train = np.array([[1,2,3,4,5]])
    Y_train = np.array([[2,3,4,5,6]])
    X_test = np.array([[1,2,3,4,5]])
    Y_test = np.array([[2,3,4,5,6]])

    # iterate on training data in batches
    model.fit(X_train, Y_train, nb_epoch=5, batch_size=32)

    # evaluate model on test data
    loss_and_metrics = model.evaluate(X_test, Y_test, batch_size=32)

    # generate predictions on test data
    classes = model.predict_classes(X_test, batch_size=32)
    proba = model.predict_proba(X_test, batch_size=32)



