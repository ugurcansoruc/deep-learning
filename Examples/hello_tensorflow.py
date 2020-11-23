#!/usr/bin/env python3

from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


def create_model(_input_shape=1, _x=1, _y=3, _epochs=100, _verbose=0):
    """It creates a basic model, compile and fit."""
    _model = keras.Sequential([
        keras.layers.Dense(1, input_shape=[_input_shape])
    ])
    _model.compile(loss=keras.losses.mean_squared_error, optimizer=keras.optimizers.SGD())
    _history = _model.fit(_x, _y, epochs=_epochs, verbose=_verbose)
    return _model, _history


def create_data(_start_range=0, _end_range=100):
    """"Creates sample data for the network."""
    _x = np.arange(_start_range, _end_range)
    _y = functions(_x)
    return _x, _y


def functions(_x):
    """Used to create target values in create_data function."""
    return 2 * _x + 1


def plot_loss(_history):
    """Plot the model loss."""
    epochs = np.arange(0, len(_history.history['loss']))
    plt.plot(epochs, _history.history['loss'])
    plt.show()
    plt.clf()


def plot_prediction(_prediction, _target):
    """It creates a graph to compare the estimates made by the network."""
    x_axis = np.arange(len(_prediction))
    plt.plot(x_axis, _prediction, 'ro')
    plt.plot(x_axis, _target, 'bo')
    plt.show()
    plt.clf()


if __name__ == '__main__':
    x_train, y_train = create_data(1, 6)
    x_predict, y_predict = create_data(20, 30)
    model, history = create_model(x_train.ndim, x_train, y_train, 50, _verbose=1)
    predictions = model.predict(x_predict)

    plot_loss(history)
    plot_prediction(predictions, y_predict)
